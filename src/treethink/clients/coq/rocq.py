from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from loguru import logger

from ..base import (
    CheckResponse,
    ProofAssistantClient,
    ProofStateInfo,
    SnippetResult,
)

if TYPE_CHECKING:
    from rocq_ml_toolbox.inference.client import PytanqueExtended

# Back-compat alias — Rocq now returns the shared client result types.
RocqSnippetResult = SnippetResult


class RocqClient(ProofAssistantClient):
    backend_name = "rocq"

    def __init__(
        self,
        host: str,
        port: int,
    ) -> None:
        self.host = host
        self.port = port
        self._pet: Optional["PytanqueExtended"] = None

    def close(self) -> None:
        if self._pet is not None and hasattr(self._pet, "close"):
            self._pet.close()
        self._pet = None

    def _ensure_client(self) -> "PytanqueExtended":
        if self._pet is None:
            from rocq_ml_toolbox.inference.client import PytanqueExtended

            self._pet = PytanqueExtended(self.host, self.port)
            self._pet.connect()
        return self._pet

    def _split_commands(self, code: str) -> List[str]:
        commands: List[str] = []
        buf: List[str] = []
        in_string = False
        comment_depth = 0
        i = 0
        while i < len(code):
            ch = code[i]
            next_ch = code[i + 1] if i + 1 < len(code) else ""

            if in_string:
                buf.append(ch)
                if ch == '"' and (i == 0 or code[i - 1] != "\\"):
                    in_string = False
                i += 1
                continue

            if comment_depth > 0:
                buf.append(ch)
                if ch == "(" and next_ch == "*":
                    comment_depth += 1
                    buf.append(next_ch)
                    i += 2
                    continue
                if ch == "*" and next_ch == ")":
                    comment_depth -= 1
                    buf.append(next_ch)
                    i += 2
                    continue
                i += 1
                continue

            if ch == '"':
                in_string = True
                buf.append(ch)
                i += 1
                continue

            if ch == "(" and next_ch == "*":
                comment_depth = 1
                buf.append(ch)
                buf.append(next_ch)
                i += 2
                continue

            if ch == "." and (not next_ch or next_ch.isspace()):
                buf.append(ch)
                cmd = "".join(buf).strip()
                if cmd:
                    commands.append(cmd)
                buf.clear()
                i += 1
                continue

            buf.append(ch)
            i += 1

        tail = "".join(buf).strip()
        if tail:
            commands.append(tail)
        return commands

    def _run_command(
        self,
        pet: "PytanqueExtended",
        state,
        cmd: str,
        timeout: Optional[float] = None,
    ):
        stripped = cmd.lstrip().lower()
        use_timeout = timeout if timeout is not None else None
        if stripped.startswith("timeout "):
            use_timeout = None
        return pet.run(state, cmd, timeout=use_timeout)

    def _parse_theorem_name(self, proof_string: str) -> str:
        """Extract the theorem name from a whole Rocq proof string."""
        import re

        matches = re.findall(
            r"(?:Theorem|Lemma|Proposition|Corollary|Example|Fact|Remark)\s+([\w']+)",
            proof_string,
        )
        if not matches:
            logger.error(
                "Could not find a theorem name in the proof string. "
                "Expected one of: Theorem, Lemma, Proposition, Corollary, "
                "Example, Fact, Remark."
            )
            return ""
        return matches[-1]

    def _extract_proof_commands(self, code: str) -> List[str]:
        """Extract proof-body commands (Proof. through Qed./Admitted./Defined./Abort.)."""
        commands = self._split_commands(code)
        proof_start = None
        terminator = None

        for i, cmd in enumerate(commands):
            stripped = cmd.strip()
            if stripped == "Proof." or stripped.startswith("Proof "):
                proof_start = i
                break

        if proof_start is None:
            raise ValueError("Could not find 'Proof.' in the proof string")

        for i in range(proof_start + 1, len(commands)):
            stripped = commands[i].strip()
            if stripped in ("Qed.", "Admitted.", "Defined.", "Abort."):
                terminator = i
                break

        if terminator is not None:
            return commands[proof_start : terminator + 1]
        return commands[proof_start:]

    def _extract_error_message(self, state) -> Optional[str]:
        feedback = getattr(state, "feedback", None)
        if not feedback:
            return None
        messages = []
        for item in feedback:
            if isinstance(item, tuple) and len(item) >= 2:
                messages.append(str(item[1]))
            elif isinstance(item, dict):
                messages.append(
                    str(item.get("message", item.get("data", item)))
                )
            else:
                messages.append(str(item))
        return "\n".join(messages) if messages else None

    # -- Proof-state extraction helpers ---------------------------------------

    @staticmethod
    def _format_goals(goals: list) -> str:
        """Pretty-print a list of ``Goal`` objects into a readable string.

        Each goal is rendered as::

            ---
            Context:
              h1 : type1
              h2 : type2
            Goal:
              <goal type>

        When no goals are present returns an empty string.
        """
        if not goals:
            return ""
        parts: list[str] = []
        for g in goals:
            hyps_str = "\n".join(
                f"  {':'.join(h.names)} : {h.ty}"
                for h in getattr(g, "hyps", [])
            )
            goal_str = getattr(g, "pp", str(g))
            block = ""
            if hyps_str:
                block += f"Context:\n{hyps_str}\n"
            block += f"Goal:\n{goal_str}"
            parts.append(block)
        return "\n\n".join(parts)

    @staticmethod
    def _compute_closed_goals(before: list, after: list) -> list:
        """Return goals present in *before* but absent in *after*.

        Comparison is by pretty-printed type (``.pp``) since evar IDs differ
        between states.  This is best-effort — bullet/stack manipulation can
        make exact comparison tricky.
        """
        before_pps = {g.pp for g in before if g.pp is not None}
        after_pps = {g.pp for g in after if g.pp is not None}
        closed_pps = before_pps - after_pps
        return [g for g in before if g.pp in closed_pps]

    def extract_proof_state(
        self,
        proof_string: str,
        response: Any = None,
    ) -> ProofStateInfo:
        """Extract tactic/goal information from a Rocq proof string.

        Replays the proof command-by-command against the ``rocq-ml-server``,
        capturing open goals and computing closed goals at each step.
        Returns the information for the **last** successfully applied tactic.
        """
        pet = self._ensure_client()
        theorem_name = self._parse_theorem_name(proof_string)
        if not theorem_name:
            return ProofStateInfo(error_message="Could not parse theorem name")
        tmp_path = pet.tmp_file(content=proof_string, root=None)

        from pytanque import PetanqueError

        try:
            state = pet.start(file=str(tmp_path), thm=theorem_name)
        except PetanqueError as exc:
            return ProofStateInfo(error_message=f"Failed to start proof: {exc}")
        except Exception as exc:
            return ProofStateInfo(
                error_message=f"Unexpected start error: {exc}"
            )

        try:
            proof_commands = self._extract_proof_commands(proof_string)
        except ValueError as exc:
            return ProofStateInfo(error_message=str(exc))

        last_applied_tactic: str | None = None
        last_open_goals: str | None = None
        last_closed_goals: str | None = None
        last_error: str | None = None

        _TERMINATORS = {"Qed.", "Admitted.", "Defined.", "Abort."}

        for cmd in proof_commands:
            stripped_cmd = cmd.strip()
            # Skip terminator commands — they close the proof
            if stripped_cmd in _TERMINATORS:
                continue

            # Capture goals before the tactic
            try:
                resp_before = pet.complete_goals(state)
                goals_before: list = list(
                    getattr(resp_before, "goals", []) or []
                )
            except Exception:
                goals_before = []

            # Run the tactic
            try:
                state = self._run_command(pet, state, cmd)
            except PetanqueError as exc:
                last_error = str(exc)
                break
            except Exception as exc:
                last_error = f"Unexpected error: {exc}"
                break

            # Capture goals after the tactic
            try:
                resp_after = pet.complete_goals(state)
                # If proof finished, complete_goals may raise or return empty
                goals_after: list = list(getattr(resp_after, "goals", []) or [])
            except Exception:
                goals_after = []

            last_applied_tactic = stripped_cmd
            last_open_goals = self._format_goals(goals_after)
            closed_goals = self._compute_closed_goals(goals_before, goals_after)
            last_closed_goals = self._format_goals(closed_goals)

        # If no tactic was applied (e.g. only terminator commands), return empty
        if last_applied_tactic is None and last_error is None:
            return ProofStateInfo()

        return ProofStateInfo(
            applied_tactic=last_applied_tactic,
            open_goals=last_open_goals,
            closed_goals=last_closed_goals,
            error_message=last_error,
        )

    def verify_whole_proof(
        self,
        proof: str,
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        pet = self._ensure_client()
        theorem_name = self._parse_theorem_name(proof)
        tmp_path = pet.tmp_file(content=proof, root=None)

        from pytanque import PetanqueError

        try:
            state = pet.start(file=str(tmp_path), thm=theorem_name)
            proof_commands = self._extract_proof_commands(proof)
            for cmd in proof_commands:
                state = self._run_command(pet, state, cmd, timeout=timeout)

            # We expect the model to put "Qed." or end the proof, no need to force it.
            # if not getattr(state, "proof_finished", False):
            #     state = self._run_command(pet, state, "Qed.", timeout=timeout)

            success = bool(getattr(state, "proof_finished", False))
            return {
                "backend": self.backend_name,
                "proof_finished": success,
                "error": None
                if success
                else self._extract_error_message(state),
                "messages": list(getattr(state, "feedback", []) or []),
            }
        except PetanqueError as exc:
            logger.error(f"Rocq evaluation | PetanqueError: {exc}")
            return {
                "backend": self.backend_name,
                "proof_finished": False,
                "error": str(exc),
                "messages": [],
            }
        except Exception as exc:
            logger.error(f"Unexpected Rocq evaluation: {exc}")
            return {
                "backend": self.backend_name,
                "proof_finished": False,
                "error": str(exc),
                "messages": [],
            }

    def check(
        self,
        *,
        snips: List[str],
        timeout: Optional[float] = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> CheckResponse:
        results = [
            SnippetResult(
                response=self.verify_whole_proof(snip, timeout=timeout)
            )
            for snip in snips
        ]
        return CheckResponse(results=results)

    def is_success_response(self, response: dict[str, Any]) -> bool:
        return bool(response.get("proof_finished")) and not response.get(
            "error"
        )
