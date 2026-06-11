from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional

from loguru import logger

from ..base import ProofAssistantClient


@dataclass
class RocqSnippetResult:
    response: dict[str, Any]


class RocqBatchClient(ProofAssistantClient):
    backend_name = "rocq"

    def __init__(
        self,
        host: str,
        port: int,
        *,
        workspace_dir: Optional[str] = None,
        theorem_name: str = "__eval",
        statement: str = "True",
        prelude: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.workspace_dir = workspace_dir
        self.theorem_name = theorem_name
        self.statement = statement
        self.prelude = prelude
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

    def _build_file_text(self) -> str:
        parts: List[str] = []
        if self.prelude:
            parts.append(self.prelude.rstrip())
        parts.append(f"Theorem {self.theorem_name} : {self.statement}.")
        return "\n".join(parts) + "\n"

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

    def verify_snippet(
        self,
        snippet: str,
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        pet = self._ensure_client()
        tmp_path = pet.tmp_file(
            content=self._build_file_text(), root=self.workspace_dir
        )

        from pytanque import PetanqueError

        try:
            state = pet.start(file=str(tmp_path), thm=self.theorem_name)
            for cmd in self._split_commands(snippet):
                state = self._run_command(pet, state, cmd, timeout=timeout)
            if not getattr(state, "proof_finished", False):
                state = self._run_command(pet, state, "Qed.", timeout=timeout)

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
            logger.error("Rocq evaluation error: %s", exc)
            return {
                "backend": self.backend_name,
                "proof_finished": False,
                "error": str(exc),
                "messages": [],
            }
        except Exception as exc:
            logger.error("Unexpected Rocq evaluation error: %s", exc)
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
    ):
        results = [
            RocqSnippetResult(
                response=self.verify_snippet(snip, timeout=timeout)
            )
            for snip in snips
        ]
        return SimpleNamespace(results=results)

    def is_success_response(self, response: dict[str, Any]) -> bool:
        return bool(response.get("proof_finished")) and not response.get(
            "error"
        )
