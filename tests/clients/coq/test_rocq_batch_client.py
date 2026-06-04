import os

import pytest

from treethink.clients.coq.rocq import RocqBatchClient

# Local rocq server coordinates
ROCQ_HOST = os.environ.get("ROCQ_HOST", "localhost")
ROCQ_PORT = int(os.environ.get("ROCQ_PORT", "5000"))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ROCQ_TESTS", "0") == "0",
    reason="RUN_ROCQ_TESTS not set. Rocq tests require a running rocq-ml-server.",
)


@pytest.fixture
def rocq_client():
    client = RocqBatchClient(
        host=ROCQ_HOST,
        port=ROCQ_PORT,
        theorem_name="rev_app_distr_user",
        statement="forall (A : Type) (xs ys : list A), rev (xs ++ ys) = rev ys ++ rev xs",
        prelude="From Coq Require Import List.\nImport ListNotations.",
    )
    yield client
    client.close()


def test_rocq_batch_client(rocq_client):
    """
    Test that the RocqBatchClient can verify a correct proof snippet.
    """
    # Excerpt from TargetGood.v
    snippet = """Proof.
  intros A xs ys.
  induction xs as [|x xs IH]; simpl.
  - now rewrite app_nil_r.
  - rewrite IH, app_assoc. reflexivity.
Qed."""

    result = rocq_client.verify_snippet(snippet)

    assert result["backend"] == "rocq"
    assert result["proof_finished"] is True
    assert result["error"] is None
    assert rocq_client.is_success_response(result) is True


def test_rocq_batch_client_invalid(rocq_client):
    """
    Test that the RocqBatchClient fails on an incorrect proof snippet.
    """
    snippet = """Proof.
  intros A xs ys.
  induction xs as [|x xs IH]; simpl.
  - reflexivity. (* This should fail *)
Qed."""

    result = rocq_client.verify_snippet(snippet)

    assert result["backend"] == "rocq"
    assert result["proof_finished"] is False
    assert result["error"] is not None
    assert rocq_client.is_success_response(result) is False
