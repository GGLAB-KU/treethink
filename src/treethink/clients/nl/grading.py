"""Math answer grading — symbolic-equivalence checking for NL answers.

The logic here is adapted from the Hendrycks *MATH* ``math_equivalence`` release
and the PRM800K answer grader: an answer is correct if it normalises to the same
string as the ground truth, or if ``sympy`` can simplify their difference to 0.

``sympy`` and ``pylatexenc`` are used when available; if either is missing the
grader degrades gracefully to string normalisation (no crash).
"""

import re
from typing import Optional

try:  # optional — symbolic equivalence
    import sympy
    from sympy.parsing import sympy_parser

    _HAS_SYMPY = True
except Exception:  # pragma: no cover - env without sympy
    _HAS_SYMPY = False

try:  # optional — LaTeX parsing
    from pylatexenc import latex2text

    _HAS_LATEX = True
except Exception:  # pragma: no cover - env without pylatexenc
    _HAS_LATEX = False


# ---------------------------------------------------------------------------
# Boxed-answer extraction
# ---------------------------------------------------------------------------


def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last ``\\boxed{...}`` / ``\\fbox{...}`` substring, or None."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> Optional[str]:
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(answer_string: str) -> Optional[str]:
    """Extract the content of the last ``\\boxed{...}`` block (or None)."""
    boxed = last_boxed_only_string(answer_string)
    if boxed is None:
        return None
    return remove_boxed(boxed)


# ---------------------------------------------------------------------------
# Normalisation (Hendrycks MATH ``math_normalize``)
# ---------------------------------------------------------------------------


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a, b = int(a_str), int(b_str)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


# ---------------------------------------------------------------------------
# sympy-based equivalence (grader.py)
# ---------------------------------------------------------------------------

_BAD_SUBSTRINGS = ["^{", "^("]
_BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
_TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    for src, dst in (
        ("√", "sqrt"),
        ("π", "pi"),
        ("∞", "inf"),
        ("∪", "U"),
        ("·", "*"),
        ("×", "*"),
    ):
        expr = expr.replace(src, dst)
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        return abs(float(x) - int(round(float(x)))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    return int(float(x.replace(",", "")))


def _inject_implicit_mixed_number(step: str) -> str:
    return re.sub(r"([0-9]) +([0-9])", r"\1+\2", step)


def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return expr


def _normalize(expr: str) -> Optional[str]:
    if expr is None:
        return None

    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%").replace("\\$", "$")
    expr = expr.replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in (
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ):
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr and _HAS_LATEX:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def _count_unknown_letters_in_expr(expr: str) -> int:
    expr = expr.replace("sqrt", "").replace("frac", "")
    return len(set(x for x in expr if x.isalpha()))


def _should_allow_eval(expr: str) -> bool:
    if _count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad in _BAD_SUBSTRINGS:
        if bad in expr:
            return False
    for bad in _BAD_REGEXES:
        if re.search(bad, expr) is not None:
            return False
    return True


def _are_equal_under_sympy(gt_norm: str, given_norm: str) -> bool:
    if not _HAS_SYMPY:
        return False
    try:
        expr = f"({gt_norm})-({given_norm})"
        if _should_allow_eval(expr):
            if sympy.simplify(_sympy_parse(expr)) == 0:
                return True
    except Exception:
        pass
    return False


def _split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in _TUPLE_CHARS
        and expr[-1] in _TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in _TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


def grade_answer(given_answer: Optional[str], ground_truth: str) -> bool:
    """True if *given_answer* matches *ground_truth* (string or sympy-equiv)."""
    if given_answer is None:
        return False

    if normalize_answer(ground_truth) == normalize_answer(given_answer):
        return True

    gt_norm = _normalize(ground_truth)
    given_norm = _normalize(given_answer)

    if gt_norm is None:
        return False
    if gt_norm == given_norm:
        return True
    if given_norm is None or len(given_norm) == 0:
        return False

    gt_elems = _split_tuple(gt_norm)
    given_elems = _split_tuple(given_norm)

    if len(gt_elems) > 1 and (
        gt_norm[0] != given_norm[0] or gt_norm[-1] != given_norm[-1]
    ):
        return False
    if len(gt_elems) != len(given_elems):
        return False

    is_correct = False
    for gt_elem, given_elem in zip(gt_elems, given_elems):
        if _is_frac(gt_elem) and _is_frac(given_elem):
            is_correct = gt_elem == given_elem
        elif _str_is_int(gt_elem) != _str_is_int(given_elem):
            is_correct = False
        else:
            is_correct = _are_equal_under_sympy(gt_elem, given_elem)
        if not is_correct:
            break
    return is_correct


__all__ = ["grade_answer", "normalize_answer", "extract_boxed_answer"]
