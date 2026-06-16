"""Integration tests for the Isabelle client.

Gated on a real Isabelle install: skipped unless the ``isabelle`` binary is on
PATH and the ``isabelle_client`` package is importable. These tests start a
real Isabelle server + HOL session, so they take a few seconds.
"""

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
def client():
    from treethink.clients.isabelle.client import IsabelleClient

    c = IsabelleClient(server_name="tt-pytest")
    yield c
    c.close()


def test_valid_proof_succeeds(client):
    resp = client.check(snips=['lemma "(1::nat) + 1 = 2" by simp'])
    assert len(resp.results) == 1
    assert client.is_success_response(resp.results[0].response) is True


def test_invalid_proof_fails(client):
    resp = client.check(snips=['lemma "(1::nat) + 1 = 3" by simp'])
    assert client.is_success_response(resp.results[0].response) is False
    assert resp.results[0].response["errors"]  # non-empty


def test_batch_preserves_order_and_per_snippet_results(client):
    snips = [
        'lemma "(1::nat) + 1 = 2" by simp',  # ok
        'lemma "(1::nat) + 1 = 3" by simp',  # fail
        'lemma "True" by blast',  # ok
    ]
    resp = client.check(snips=snips, batch_size=2, max_workers=2)
    outcomes = [client.is_success_response(r.response) for r in resp.results]
    assert outcomes == [True, False, True]


def test_empty_snips(client):
    resp = client.check(snips=[])
    assert resp.results == []


def test_factory_creates_isabelle_client():
    """create_client routes FormalLanguage.ISABELLE to a working client."""
    from treethink.client_factory import create_client
    from treethink.utils.args import ClientArgs
    from treethink.utils.enums import FormalLanguage

    c = create_client(FormalLanguage.ISABELLE, ClientArgs())
    try:
        resp = c.check(snips=['lemma "True" by simp'])
        assert c.is_success_response(resp.results[0].response) is True
    finally:
        c.close()
