"""Tests for the XML ``parse_tag`` feature in policies."""

import vllm

from treethink.policies import BasePolicy

# ── Helpers ──────────────────────────────────────────────────────────────


class _ConcretePolicy(BasePolicy):
    """Minimal concrete policy used to exercise BasePolicy methods."""

    def __init__(self, parse_tag: str = "\n", *args, **kwargs):
        super().__init__(
            name="test_policy", parse_tag=parse_tag, *args, **kwargs
        )

    def __call__(self, node, method):
        pass


def _make_policy(parse_tag: str = "\n") -> _ConcretePolicy:
    return _ConcretePolicy(parse_tag=parse_tag)


# ── _derive_closing_tag ──────────────────────────────────────────────────


class TestDeriveClosingTag:
    def test_reasoning_tag(self):
        policy = _make_policy(parse_tag="<reasoning>")
        assert policy._derive_closing_tag() == "</reasoning>"

    def test_answer_tag(self):
        policy = _make_policy(parse_tag="<answer>")
        assert policy._derive_closing_tag() == "</answer>"

    def test_think_tag(self):
        policy = _make_policy(parse_tag="<think>")
        assert policy._derive_closing_tag() == "</think>"

    def test_newline_default(self):
        policy = _make_policy(parse_tag="\n")
        # ``"\n"[1:]`` is ``""``, so the derived "closing tag" is ``"</"``.
        # This is never invoked in practice because the ``"\n"`` branch
        # in ``set_sampling_params`` and ``_clean_text`` short-circuits
        # before reaching this method.
        assert policy._derive_closing_tag() == "</"


# ── _clean_text ──────────────────────────────────────────────────────────


class TestCleanText:
    def test_default_newline_returns_text_unchanged(self):
        policy = _make_policy(parse_tag="\n")
        text = "some generated content\n"
        assert policy._clean_text(text) == text

    def test_strips_opening_and_closing_tags(self):
        policy = _make_policy(parse_tag="<reasoning>")
        raw = "<reasoning>We can prove this by induction.</reasoning>"
        expected = "We can prove this by induction."
        assert policy._clean_text(raw) == expected

    def test_strips_only_leading_opening_tag(self):
        policy = _make_policy(parse_tag="<reasoning>")
        raw = "<reasoning>Start of proof."
        expected = "Start of proof."
        assert policy._clean_text(raw) == expected

    def test_strips_only_trailing_closing_tag(self):
        policy = _make_policy(parse_tag="<reasoning>")
        raw = "Continuing the proof.</reasoning>"
        expected = "Continuing the proof."
        assert policy._clean_text(raw) == expected

    def test_no_tags_present_returns_unchanged(self):
        policy = _make_policy(parse_tag="<reasoning>")
        raw = "Just plain text without any XML tags."
        assert policy._clean_text(raw) == raw

    def test_tags_in_middle_are_preserved(self):
        """Only leading/trailing occurrences are stripped."""
        policy = _make_policy(parse_tag="<tag>")
        raw = "some <tag>nested</tag> content"
        assert policy._clean_text(raw) == raw

    def test_empty_string(self):
        policy = _make_policy(parse_tag="<reasoning>")
        assert policy._clean_text("") == ""

    def test_only_tags(self):
        policy = _make_policy(parse_tag="<reasoning>")
        raw = "<reasoning></reasoning>"
        assert policy._clean_text(raw) == ""


# ── set_sampling_params ──────────────────────────────────────────────────


class TestSetSamplingParams:
    def test_default_newline_does_not_replace_stop(self):
        """When ``parse_tag == "\\n"``, ``stop`` is left as-is."""
        policy = _make_policy(parse_tag="\n")
        params = vllm.SamplingParams(stop=["\n"])
        result = policy.set_sampling_params(params)
        assert result.stop == ["\n"]

    def test_xml_tag_replaces_stop_with_closing_tag(self):
        """When ``parse_tag != "\\n"``, ``stop`` becomes ``[closing_tag]``."""
        policy = _make_policy(parse_tag="<reasoning>")
        params = vllm.SamplingParams(stop=["\n"])
        result = policy.set_sampling_params(params)
        assert result.stop == ["</reasoning>"]

    def test_xml_tag_with_no_initial_stop(self):
        policy = _make_policy(parse_tag="<answer>")
        params = vllm.SamplingParams()
        result = policy.set_sampling_params(params)
        assert result.stop == ["</answer>"]

    def test_always_sets_include_stop_str_in_output(self):
        policy = _make_policy(parse_tag="<reasoning>")
        params = vllm.SamplingParams()
        result = policy.set_sampling_params(params)
        assert result.include_stop_str_in_output is True

    def test_default_newline_sets_include_stop_str_in_output(self):
        policy = _make_policy(parse_tag="\n")
        params = vllm.SamplingParams()
        result = policy.set_sampling_params(params)
        assert result.include_stop_str_in_output is True

    def test_returns_sampling_params(self):
        policy = _make_policy(parse_tag="\n")
        params = vllm.SamplingParams()
        result = policy.set_sampling_params(params)
        assert result is params
