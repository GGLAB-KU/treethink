from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, List, Optional

from loguru import logger

from ..base import (
    AsyncProofAssistantClient,
    CheckResponse,
    ProofAssistantClient,
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
        batch_size: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self._pets: List[Optional["PytanqueExtended"]] = [None] * batch_size

    def close(self) -> None:
        for pet in self._pets:
            if pet is not None and hasattr(pet, "close"):
                pet.close()
        self._pets = [None] * self.batch_size

    def _ensure_client(self, idx: int = 0) -> "PytanqueExtended":
        if self._pets[idx] is None:
            from rocq_ml_toolbox.inference.client import PytanqueExtended

            self._pets[idx] = PytanqueExtended(self.host, self.port)
            self._pets[idx].connect()
        return self._pets[idx]

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

    def verify_whole_proof(
        self,
        proof: str,
        *,
        timeout: Optional[float] = None,
        _conn_idx: int = 0,
    ) -> dict[str, Any]:
        pet = self._ensure_client(_conn_idx)
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
        if not snips:
            return CheckResponse(results=[])

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Partition snips into groups; each group goes to one connection.
        groups = [
            snips[i : i + batch_size] for i in range(0, len(snips), batch_size)
        ]
        num_groups = len(groups)
        num_workers = min(self.batch_size, num_groups)

        def _process_group(
            group: List[str], conn_idx: int
        ) -> List[SnippetResult]:
            return [
                SnippetResult(
                    response=self.verify_whole_proof(
                        s, timeout=timeout, _conn_idx=conn_idx
                    )
                )
                for s in group
            ]

        ordered_results: List[Optional[SnippetResult]] = [None] * len(snips)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for i, group in enumerate(groups):
                conn_idx = i % self.batch_size
                fut = executor.submit(_process_group, group, conn_idx)
                futures[fut] = i

            for fut in as_completed(futures):
                group_idx = futures[fut]
                start = group_idx * batch_size
                for j, r in enumerate(fut.result()):
                    ordered_results[start + j] = r

        return CheckResponse(results=ordered_results)

    def is_success_response(self, response: dict[str, Any]) -> bool:
        return bool(response.get("proof_finished")) and not response.get(
            "error"
        )


class AsyncRocqClient(AsyncProofAssistantClient):
    """Async wrapper around the synchronous :class:`RocqClient`.

    Holds a pre-initialised pool of ``RocqClient`` instances (each with
    its own ``PytanqueExtended`` connection).  ``check()`` partitions
    the input snippets into groups and assigns each group to a different
    pooled client so that concurrent ``asyncio.to_thread`` calls do not
    corrupt shared proof state.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        batch_size: int = 4,
    ) -> None:
        self._host = host
        self._port = port
        self._batch_size = batch_size
        self._pool: List[RocqClient] = [
            RocqClient(host=host, port=port, batch_size=1)
            for _ in range(batch_size)
        ]

    async def check(
        self,
        *,
        snips: List[str],
        timeout: Optional[float] = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> CheckResponse:
        """Async batch-verify proof snippets in parallel.

        Partitions *snips* into groups of *batch_size* and verifies
        each group on a different :class:`RocqClient` from the pool.
        """
        if not snips:
            return CheckResponse(results=[])

        # Partition snips into groups; each group goes to one connection.
        groups = [
            snips[i : i + batch_size] for i in range(0, len(snips), batch_size)
        ]

        async def _process_group(
            group: List[str], conn_idx: int
        ) -> List[SnippetResult]:
            client = self._pool[conn_idx % self._batch_size]
            results = []
            for snip in group:
                r = await asyncio.to_thread(
                    client.verify_whole_proof, snip, timeout=timeout
                )
                results.append(SnippetResult(response=r))
            return results

        group_coros = [_process_group(groups[i], i) for i in range(len(groups))]
        group_results = await asyncio.gather(*group_coros)

        # Flatten, preserving order.
        ordered_results = []
        for gr in group_results:
            ordered_results.extend(gr)

        return CheckResponse(results=ordered_results)

    def is_success_response(self, response: dict[str, Any]) -> bool:
        return bool(response.get("proof_finished")) and not response.get(
            "error"
        )

    async def close(self) -> None:
        for client in self._pool:
            client.close()
