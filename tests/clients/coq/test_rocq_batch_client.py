import os

import pytest

from treethink.clients.coq.rocq import RocqClient

# Local rocq server coordinates
ROCQ_HOST = os.environ.get("ROCQ_HOST", "localhost")
ROCQ_PORT = int(os.environ.get("ROCQ_PORT", "5000"))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ROCQ_TESTS", "0") == "0",
    reason="RUN_ROCQ_TESTS not set. Rocq tests require a running rocq-ml-server.",
)


@pytest.fixture
def rocq_client():
    client = RocqClient(
        host=ROCQ_HOST,
        port=ROCQ_PORT,
    )
    yield client
    client.close()


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
Qed."""


def test_rocq_batch_client(rocq_client, valid_whole_proof):
    """
    Test that the RocqClient can verify a correct whole proof.
    """
    result = rocq_client.verify_whole_proof(valid_whole_proof)

    assert result["backend"] == "rocq"
    assert result["proof_finished"] is True
    assert result["error"] is None
    assert rocq_client.is_success_response(result) is True


def test_rocq_batch_client_invalid(rocq_client, invalid_whole_proof):
    """
    Test that the RocqClient fails on an incorrect whole proof.
    """
    result = rocq_client.verify_whole_proof(invalid_whole_proof)

    assert result["backend"] == "rocq"
    assert result["proof_finished"] is False
    assert result["error"] is not None
    assert rocq_client.is_success_response(result) is False


def test_rocq_check_method(rocq_client, valid_whole_proof, invalid_whole_proof):
    """
    Test the check() method with multiple whole-proof strings.
    """
    response = rocq_client.check(snips=[valid_whole_proof, invalid_whole_proof])

    assert len(response.results) == 2
    assert rocq_client.is_success_response(response.results[0].response) is True
    assert (
        rocq_client.is_success_response(response.results[1].response) is False
    )
