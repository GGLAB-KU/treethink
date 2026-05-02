import re


def extract_result(answer_string):
    boxed_string = _last_boxed_only_string(answer_string)
    if boxed_string:
        return _remove_boxed(boxed_string)
    else:
        return "NO_BOXED_STRING_FOUND"


def _last_boxed_only_string(string):
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
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def _remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def check_tags(string, check_start_value="<code>", check_end_value="</code>"):
    # Even number of tags check
    if len(re.findall("<.*?>", string)) % 2 != 0:
        return False
    # Copy string for reverse check.
    string_reverse = string
    if check_start_value in string:
        # There might be more than one tag blocks.
        number_of_start_value = len(re.findall(check_start_value, string))
        # We should ensure that each tag block paired correctly.
        for i in range(number_of_start_value):
            start_index = string.find(check_start_value) + len(
                check_start_value
            )
            end_index = string.find(check_end_value, start_index)
            # If tag block is not ended then return False.
            if end_index == -1:
                return False
            # Check if next tag is check_end_value
            for tag in re.findall("<.*?>", string[start_index:end_index]):
                if tag != check_end_value:
                    return False
            string = string[end_index:]
        # Copy string for reverse check.

    if check_end_value in string_reverse:
        # There might be more than one tag blocks.
        number_of_end_value = len(re.findall(check_end_value, string_reverse))
        # We should ensure that each tag block paired correctly.
        for i in range(number_of_end_value):
            end_index = string_reverse.find(check_end_value)
            start_index = string_reverse[:end_index].find(check_start_value)

            # İf tag block is not started then retun False.
            if start_index == -1:
                return False
            # Check if previous tag is check_start_value
            for tag in re.findall(
                "<.*?>", string_reverse[start_index:end_index]
            ):
                if tag != check_start_value:
                    return False
            string_reverse = string_reverse[end_index + len(check_end_value) :]

    return True
