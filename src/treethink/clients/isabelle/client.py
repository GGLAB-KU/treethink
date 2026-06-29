"""Isabelle proof-assistant client backed by ``isabelle-client``.

Wraps Isabelle's server protocol (started programmatically) so treethink can
verify Isabelle/HOL proof snippets through the language-agnostic
:class:`~treethink.clients.base.ProofAssistantClient` interface — the same
way the Lean and Rocq clients plug in.

Each snippet handed to :meth:`IsabelleClient.check` is treated as the *body*
of a theory; the client supplies the ``theory <name> imports <imports>
begin … end`` scaffold and processes it via ``use_theories``.  A snippet is
successful when its theory finishes with no ``error`` messages.
"""

import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

from loguru import logger

from ..base import (
    CheckResponse,
    ProofAssistantClient,
    ProofStateInfo,
    SnippetResult,
)

# Back-compat aliases — Isabelle now returns the shared client result types.
# Each ``SnippetResult.response`` is a dict
# ``{"ok": bool, "errors": List[str], "theory": Optional[str]}``.
IsabelleSnippetResult = SnippetResult
IsabelleCheckResponse = CheckResponse

# Isabelle embeds the remaining goal state inside failure messages, e.g.
# ``Failed to apply proof method:\ngoal (1 subgoal):\n 1. ...``.
_GOAL_BLOCK_RE = re.compile(r"goal\s*\(\d+\s+subgoal[s]?\):.*", re.DOTALL)


class IsabelleClient(ProofAssistantClient):
    """Verify Isabelle/HOL proof snippets via the Isabelle server.

    On construction this starts an Isabelle server and a session (default
    logic ``HOL``).  Snippets are split into batches of ``batch_size``
    theories (one ``use_theories`` call each) processed across
    ``max_workers`` threads; concurrent processing on a single session is
    supported by the server (~linear speedup measured).

    Parameters
    ----------
    session : str
        Isabelle session/logic to start (default ``"HOL"``).
    imports : str
        Imports clause inserted into each generated theory (default
        ``"Main"``; may list several, e.g. ``'Main "HOL-Library.Multiset"'``).
    server_name : str
        Name passed to the Isabelle server.
    server_log : str, optional
        Path for the Isabelle server log file.
    session_dirs : list[str], optional
        Extra session directories (``-d``) for theory lookup.
    """

    def __init__(
        self,
        session: str = "HOL",
        imports: str = "Main",
        server_name: str = "treethink",
        server_log: Optional[str] = None,
        session_dirs: Optional[List[str]] = None,
    ) -> None:
        from isabelle_client import get_isabelle_client, start_isabelle_server

        self.session = session
        self.imports = imports
        self._server_info, self._server_process = start_isabelle_server(
            name=server_name, log_file=server_log
        )
        self._client = get_isabelle_client(self._server_info)
        self._session_id = self._start_session(session_dirs)
        logger.info(
            f"Isabelle session '{session}' ready (id={self._session_id[:8]})."
        )

    def _start_session(self, dirs: Optional[List[str]]) -> str:
        session_id = None
        for resp in self._client.session_start(session=self.session, dirs=dirs):
            session_id = getattr(resp.response_body, "session_id", None) or (
                session_id
            )
        if session_id is None:
            raise RuntimeError(
                f"Failed to start Isabelle session '{self.session}'."
            )
        return session_id

    def _wrap(self, name: str, snippet: str) -> str:
        """Wrap a snippet body into a complete, name-matched theory file."""
        return (
            f"theory {name}\n  imports {self.imports}\nbegin\n{snippet}\nend\n"
        )

    def _run_batch(
        self,
        indexed_snips: List[Tuple[int, str]],
        timeout: Optional[int],
    ) -> List[Tuple[int, dict]]:
        """Process one batch of ``(index, snippet)`` via a single
        ``use_theories`` call; returns ``(index, response_dict)`` pairs."""
        master_dir = tempfile.mkdtemp(prefix="treethink_isa_")
        name_to_idx = {}
        for idx, snippet in indexed_snips:
            name = f"TT_{idx}"
            name_to_idx[name] = idx
            with open(os.path.join(master_dir, f"{name}.thy"), "w") as fh:
                fh.write(self._wrap(name, snippet))

        kwargs = {}
        if timeout is not None:
            kwargs["watchdog_timeout"] = float(timeout)

        try:
            responses = self._client.use_theories(
                session_id=self._session_id,
                theories=list(name_to_idx.keys()),
                master_dir=master_dir,
                **kwargs,
            )
        except Exception as exc:  # library/server-level failure
            logger.error(f"Isabelle use_theories failed: {exc}")
            return [
                (idx, {"ok": False, "errors": [str(exc)], "theory": None})
                for idx, _ in indexed_snips
            ]

        nodes = getattr(responses[-1].response_body, "nodes", []) or []
        by_idx: dict = {}
        for node in nodes:
            short_name = node.theory_name.split(".")[-1]  # strip "Draft." etc.
            idx = name_to_idx.get(short_name)
            if idx is None:
                continue
            errors = [m.message for m in node.messages if m.kind == "error"]
            by_idx[idx] = {
                "ok": not errors,
                "errors": errors,
                "theory": node.theory_name,
            }

        return [
            (
                idx,
                by_idx.get(
                    idx,
                    {"ok": False, "errors": ["no node result"], "theory": None},
                ),
            )
            for idx, _ in indexed_snips
        ]

    def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,  # accepted for interface parity
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> CheckResponse:
        """Batch-verify proof snippets, preserving input order."""
        indexed = list(enumerate(snips))
        if not indexed:
            return CheckResponse(results=[])

        batches = [
            indexed[i : i + max(1, batch_size)]
            for i in range(0, len(indexed), max(1, batch_size))
        ]

        collected: dict = {}
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for batch_result in executor.map(
                lambda batch: self._run_batch(batch, timeout), batches
            ):
                for idx, resp in batch_result:
                    collected[idx] = resp

        results = [
            SnippetResult(response=collected[i]) for i in range(len(snips))
        ]
        return CheckResponse(results=results)

    def is_success_response(self, response: Any) -> bool:
        """``True`` when the snippet's theory closed with no errors."""
        if isinstance(response, dict):
            return bool(response.get("ok"))
        return bool(getattr(response, "ok", False))

    # ------------------------------------------------------------------
    # Proof-state extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_markup(text: str) -> str:
        """Drop Isabelle's position marker; keep math notation intact."""
        return text.replace("\\<^here>", "").strip()

    @staticmethod
    def _last_proof_step(snippet: str) -> Optional[str]:
        """Best-effort applied tactic: the last non-empty line of the body."""
        for line in reversed(snippet.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    def extract_proof_state(
        self,
        proof_string: str,
        response: Any = None,
    ) -> ProofStateInfo:
        """Extract tactic / goal / error info from an Isabelle proof.

        Isabelle's batch ``use_theories`` API does not expose intermediate
        goal states on success, but its **failure** messages embed the
        remaining goal (``goal (N subgoals): ...``).  So this is
        error-focused: on failure it returns the failing ``error_message``
        and the ``open_goals`` parsed from it; ``applied_tactic`` is the last
        proof step of *proof_string*.  ``closed_goals`` is not derivable from
        this API and is left ``None``.

        ``response`` is the per-snippet dict from :meth:`check`
        (``{"ok", "errors", "theory"}``); if not given, the proof is
        re-verified.
        """
        if isinstance(response, dict):
            errors = response.get("errors") or []
        else:
            try:
                resp = self.check(snips=[proof_string]).results[0].response
                errors = resp.get("errors") or []
            except Exception as exc:
                logger.warning(f"Isabelle proof-state extraction failed: {exc}")
                return ProofStateInfo(error_message=str(exc))

        applied_tactic = self._last_proof_step(proof_string)

        if not errors:
            # No errors → theory processed cleanly, no remaining goals.
            return ProofStateInfo(applied_tactic=applied_tactic, open_goals="")

        descriptions: List[str] = []
        open_goals: Optional[str] = None
        for err in errors:
            match = _GOAL_BLOCK_RE.search(err)
            if match and open_goals is None:
                open_goals = self._clean_markup(match.group(0))
                desc = self._clean_markup(err[: match.start()]).rstrip(":")
                descriptions.append(desc)
            else:
                descriptions.append(self._clean_markup(err))

        error_message = "\n".join(d for d in descriptions if d) or None
        return ProofStateInfo(
            applied_tactic=applied_tactic,
            open_goals=open_goals,
            error_message=error_message,
        )

    def close(self) -> None:
        try:
            if getattr(self, "_session_id", None):
                self._client.session_stop(session_id=self._session_id)
        except Exception as exc:
            logger.warning(f"Failed to stop Isabelle session: {exc}")
        finally:
            process = getattr(self, "_server_process", None)
            if process is not None:
                process.terminate()


__all__ = ["IsabelleClient", "IsabelleSnippetResult", "IsabelleCheckResponse"]
