import importlib.util
import shutil

import pytest

_ISABELLE_AVAILABLE = (
    shutil.which("isabelle") is not None
    and importlib.util.find_spec("isabelle_client") is not None
)

pytestmark = pytest.mark.skipif(
    not _ISABELLE_AVAILABLE,
    reason="Isabelle binary and/or isabelle-client not available.",
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def isabelle_client():
    """Create an :class:`AsyncIsabelleClient` with an active server session.

    The server is started during ``__init__`` (via ``_init_server``) so no
    explicit ``await client.start()`` is needed.
    """
    from treethink.clients.isabelle.client import AsyncIsabelleClient

    client = AsyncIsabelleClient(
        session="HOL", server_name="pytest_server", max_workers=2
    )
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Regression: ``__init__`` starts the server in *any* asyncio context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_started_in_init():
    """The server is started during ``__init__`` inside a running event loop
    (regression guard for the ``asyncio.run()``-inside-event-loop bug)."""
    from treethink.clients.isabelle.client import AsyncIsabelleClient

    client = AsyncIsabelleClient(
        session="HOL", server_name="pytest_server", max_workers=2
    )
    assert client._client is not None
    assert client._session_id is not None
    client.close()


@pytest.mark.asyncio
async def test_start_is_idempotent(isabelle_client):
    """Calling ``start()`` after init is a safe no-op."""
    await isabelle_client.start()
    assert isabelle_client._client is not None


def test_server_started_in_sync_context():
    """The server is also started during ``__init__`` in a plain sync context."""
    from treethink.clients.isabelle.client import AsyncIsabelleClient

    client = AsyncIsabelleClient(
        session="HOL", server_name="pytest_server", max_workers=2
    )
    assert client._client is not None
    assert client._session_id is not None
    client.close()


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_valid_proof(isabelle_client):
    """Verifies that a mathematically sound snippet yields a success response."""
    valid_snippet = 'lemma valid_proof: "A ⟶ A"\n  by auto'

    response = await isabelle_client.check(snips=[valid_snippet])

    assert len(response.results) == 1
    result = response.results[0].response
    assert isabelle_client.is_success_response(result) is True
    assert not result.get("errors")


@pytest.mark.asyncio
async def test_check_invalid_proof(isabelle_client):
    """Verifies that an incorrect mathematical statement is flagged as an error."""
    invalid_snippet = 'lemma invalid_proof: "True = False"\n  by auto'

    response = await isabelle_client.check(snips=[invalid_snippet])

    assert len(response.results) == 1
    result = response.results[0].response
    assert isabelle_client.is_success_response(result) is False
    assert len(result.get("errors", [])) > 0


@pytest.mark.asyncio
async def test_extract_proof_state_on_failure(isabelle_client):
    """Ensures goal states and error messages are extracted from failed proofs."""
    invalid_snippet = 'lemma invalid_proof: "True = False"\n  by auto'

    state = await isabelle_client.extract_proof_state(invalid_snippet)

    assert state.applied_tactic == "by auto"
    assert state.error_message is not None
    assert "Failed to finish proof" in state.error_message
