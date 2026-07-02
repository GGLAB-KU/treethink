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


@pytest.fixture(scope="module")
def isabelle_client():
    """Initializes the Isabelle client and ensures proper cleanup after tests."""

    from treethink.clients.isabelle.client import AsyncIsabelleClient

    client = AsyncIsabelleClient(
        session="HOL", server_name="pytest_server", max_workers=2
    )
    yield client
    client.close()


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
