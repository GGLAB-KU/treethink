import os

import pytest

from treethink.clients.coq.rocq import AsyncRocqClient

# Local rocq server coordinates
ROCQ_HOST = os.environ.get("ROCQ_HOST", "localhost")
ROCQ_PORT = int(os.environ.get("ROCQ_PORT", "5000"))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ROCQ_TESTS", "0") == "0",
    reason="RUN_ROCQ_TESTS not set. Rocq tests require a running rocq-ml-server.",
)


@pytest.fixture
async def async_rocq_client():
    client = AsyncRocqClient(
        host=ROCQ_HOST,
        port=ROCQ_PORT,
    )
    yield client
    await client.close()


@pytest.fixture
def valid_whole_proof() -> str:
    """A complete valid Rocq proof file."""
    return """From Coq Require Import List.
Import ListNotations.

Theorem rev_app_distr_user :
  forall (A : Type) (xs ys : list A),
    rev (xs ++ ys) = rev ys ++ rev xs.
Proof.
  intros A xs ys.
  induction xs as [|x xs IH]; simpl.
  - now rewrite app_nil_r.
  - rewrite IH, app_assoc. reflexivity.
Qed."""


@pytest.fixture
def invalid_whole_proof() -> str:
    """A complete but incorrect Rocq proof file."""
    return """From Coq Require Import List.
Import ListNotations.

Theorem rev_app_distr_user :
  forall (A : Type) (xs ys : list A),
    rev (xs ++ ys) = rev ys ++ rev xs.
Proof.
  intros A xs ys.
  induction xs as [|x xs IH]; simpl.
  - reflexivity. (* This should fail *)
  - rewrite IH, app_assoc. reflexivity.
Qed."""


@pytest.mark.asyncio
async def test_async_rocq_check_single_valid(
    async_rocq_client, valid_whole_proof
):
    """Verify a single valid proof via async check()."""
    response = await async_rocq_client.check(snips=[valid_whole_proof])

    assert len(response.results) == 1
    assert (
        async_rocq_client.is_success_response(response.results[0].response)
        is True
    )


@pytest.mark.asyncio
async def test_async_rocq_check_single_invalid(
    async_rocq_client, invalid_whole_proof
):
    """Verify a single invalid proof via async check()."""
    response = await async_rocq_client.check(snips=[invalid_whole_proof])

    assert len(response.results) == 1
    assert (
        async_rocq_client.is_success_response(response.results[0].response)
        is False
    )


@pytest.mark.asyncio
async def test_async_rocq_check_mixed_batch(
    async_rocq_client, valid_whole_proof, invalid_whole_proof
):
    """Verify a batch of mixed valid/invalid proofs concurrently via async check()."""
    response = await async_rocq_client.check(
        snips=[valid_whole_proof, invalid_whole_proof]
    )

    assert len(response.results) == 2
    assert (
        async_rocq_client.is_success_response(response.results[0].response)
        is True
    )
    assert (
        async_rocq_client.is_success_response(response.results[1].response)
        is False
    )


@pytest.mark.asyncio
async def test_async_rocq_response_shape(async_rocq_client, valid_whole_proof):
    """Check the response dict shape produced by the async client."""
    response = await async_rocq_client.check(snips=[valid_whole_proof])

    result = response.results[0].response
    assert isinstance(result, dict)
    assert result["backend"] == "rocq"
    assert result["proof_finished"] is True
    assert result["error"] is None
    assert "messages" in result


@pytest.mark.asyncio
async def test_async_rocq_invalid_response_shape(
    async_rocq_client, invalid_whole_proof
):
    """Check the response dict shape for a failing proof."""
    response = await async_rocq_client.check(snips=[invalid_whole_proof])

    result = response.results[0].response
    assert isinstance(result, dict)
    assert result["backend"] == "rocq"
    assert result["proof_finished"] is False
    assert result["error"] is not None
    assert "messages" in result


@pytest.mark.asyncio
async def test_async_rocq_check_empty_batch(async_rocq_client):
    """Verify an empty batch returns an empty CheckResponse."""
    response = await async_rocq_client.check(snips=[])

    assert len(response.results) == 0


@pytest.mark.asyncio
async def test_async_rocq_concurrent_verification(
    async_rocq_client, valid_whole_proof, invalid_whole_proof
):
    """Verify that multiple snippets are processed concurrently (async gather)."""
    response = await async_rocq_client.check(
        snips=[valid_whole_proof, invalid_whole_proof, valid_whole_proof]
    )

    assert len(response.results) == 3
    # Valid proofs should succeed
    assert (
        async_rocq_client.is_success_response(response.results[0].response)
        is True
    )
    # Invalid proof should fail
    assert (
        async_rocq_client.is_success_response(response.results[1].response)
        is False
    )
    # Second valid proof should succeed
    assert (
        async_rocq_client.is_success_response(response.results[2].response)
        is True
    )
