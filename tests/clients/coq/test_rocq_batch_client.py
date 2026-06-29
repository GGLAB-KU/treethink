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


# ---------------------------------------------------------------------------
# extract_proof_state tests
# ---------------------------------------------------------------------------


@pytest.fixture
def half_proof() -> str:
    """A half-complete Rocq proof — only the first two tactics are applied."""
    return """From Coq Require Import List.
Import ListNotations.

Theorem rev_app_distr_user :
  forall (A : Type) (xs ys : list A),
    rev (xs ++ ys) = rev ys ++ rev xs.
Proof.
  intros A xs ys.
  induction xs as [|x xs IH]; simpl.
  now rewrite app_nil_r.
Qed."""


@pytest.fixture
def half_proof_with_error() -> str:
    """A proof that will fail at the second tactic."""
    return """From Coq Require Import List.
Import ListNotations.

Theorem rev_app_distr_user :
  forall (A : Type) (xs ys : list A),
    rev (xs ++ ys) = rev ys ++ rev xs.
Proof.
  intros A xs ys.
  induction xs as [|x xs IH]; simpl.
  reflexivity.
Qed."""


def test_extract_proof_state_valid(rocq_client, valid_whole_proof):
    """
    A complete valid proof should return the last tactic applied and no open goals.
    """
    info = rocq_client.extract_proof_state(valid_whole_proof)

    # The last tactic is "reflexivity."
    assert info.applied_tactic is not None
    assert info.applied_tactic == "reflexivity."
    # The proof is complete, so open goals should be empty (or "∎")
    assert info.open_goals == "" or info.open_goals is None
    assert info.error_message is None


def test_extract_proof_state_half(rocq_client, half_proof):
    """
    A half-complete proof should return the last tactic and show open goals.
    """
    info = rocq_client.extract_proof_state(half_proof)

    assert info.applied_tactic is not None
    # The last tactic applied is "now rewrite app_nil_r."
    assert (
        "rewrite" in info.applied_tactic or "app_nil_r" in info.applied_tactic
    )
    # There should still be open goals (the induction step remains)
    assert info.open_goals is not None and len(info.open_goals) > 0
    # Closed goals should include the base case
    assert info.closed_goals is not None and len(info.closed_goals) > 0
    assert info.error_message is None


def test_extract_proof_state_error(rocq_client, half_proof_with_error):
    """
    A proof that fails mid-way should capture the error and the last success.
    """
    info = rocq_client.extract_proof_state(half_proof_with_error)

    # The last successful tactic is "simpl."
    assert info.applied_tactic is not None
    # There should be an error message
    assert info.error_message is not None


def test_extract_proof_state_no_theorem(rocq_client):
    """
    A string without a theorem name returns an error ProofStateInfo.
    """
    info = rocq_client.extract_proof_state("some random text")

    assert info.error_message is not None
    assert info.applied_tactic is None


def test_extract_proof_state_no_proof_block(rocq_client):
    """
    A string with a theorem but no 'Proof.' block returns an error.
    """
    info = rocq_client.extract_proof_state("Theorem foo : True. Qed.")

    assert info.error_message is not None


def test_format_goals_empty():
    """
    _format_goals with an empty list returns empty string.
    """
    assert RocqClient._format_goals([]) == ""


def test_compute_closed_goals_empty():
    """
    _compute_closed_goals with empty lists returns empty list.
    """
    assert RocqClient._compute_closed_goals([], []) == []
