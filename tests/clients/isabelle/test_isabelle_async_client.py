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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def isabelle_client():
    """Create an :class:`AsyncIsabelleClient` with an active server session.

    The server is started via the explicit ``await client.start()`` path
    (regression guard for the ``asyncio.run()``-inside-event-loop bug).
    """
    from treethink.clients.isabelle.client import AsyncIsabelleClient

    client = AsyncIsabelleClient(
        session="HOL", server_name="pytest_server", max_workers=2
    )
    await client.start()
    yield client
    client.close()


@pytest.fixture
def isabelle_client_unstarted():
    """Create an :class:`AsyncIsabelleClient` **without** starting the server.

    Use this to test lazy-initialisation behaviour.
    """
    from treethink.clients.isabelle.client import AsyncIsabelleClient

    client = AsyncIsabelleClient(
        session="HOL", server_name="pytest_server", max_workers=2
    )
    return client


# ---------------------------------------------------------------------------
# Explicit ``start()`` path (regression test for the asyncio.run() bug)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_called_explicitly(isabelle_client):
    """``start()`` can be called from within a running event loop."""
    # The fixture already calls ``start()`` — the server must be connected.
    assert isabelle_client._client is not None
    assert isabelle_client._session_id is not None


@pytest.mark.asyncio
async def test_start_is_idempotent(isabelle_client):
    """Calling ``start()`` twice is a safe no-op."""
    await isabelle_client.start()  # second call — should not raise
    assert isabelle_client._client is not None


# ---------------------------------------------------------------------------
# Lazy-initialisation path (server starts on first use)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_start_on_check(isabelle_client_unstarted):
    """Server starts automatically on first ``check()`` call."""
    client = isabelle_client_unstarted
    assert client._client is None  # not yet started

    valid_snippet = 'lemma valid_proof: "A ⟶ A"\n  by auto'
    response = await client.check(snips=[valid_snippet])

    assert client._client is not None  # now started
    assert len(response.results) == 1
    result = response.results[0].response
    assert client.is_success_response(result) is True
    assert not result.get("errors")
    client.close()


@pytest.mark.asyncio
async def test_lazy_start_on_extract_proof_state(isabelle_client_unstarted):
    """Server starts automatically on first ``extract_proof_state()`` call."""
    client = isabelle_client_unstarted
    assert client._client is None  # not yet started

    invalid_snippet = 'lemma invalid_proof: "True = False"\n  by auto'
    state = await client.extract_proof_state(invalid_snippet)

    assert client._client is not None  # now started
    assert state.applied_tactic == "by auto"
    assert state.error_message is not None
    client.close()


# ---------------------------------------------------------------------------
# Idempotent close (never started)
# ---------------------------------------------------------------------------


def test_close_without_start():
    """``close()`` is a safe no-op when the server was never started."""
    from treethink.clients.isabelle.client import AsyncIsabelleClient

    client = AsyncIsabelleClient(
        session="HOL", server_name="pytest_server", max_workers=2
    )
    # Should not raise any exception
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
