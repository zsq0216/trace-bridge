"""Translate supported path globs to regular expressions."""
import re


def _alternatives(pattern):
    start = next((i for i, ch in enumerate(pattern) if ch == "{" and (i == 0 or pattern[i - 1] != "\\")), -1)
    if start < 0:
        return [pattern]
    end = pattern.find("}", start)
    if end < 0 or "{" in pattern[start + 1:end]:
        raise ValueError("GLOB_BRACE_SYNTAX_UNSUPPORTED")
    choices = pattern[start + 1:end].split(",")
    if len(choices) < 2 or any(not part for part in choices):
        raise ValueError("GLOB_BRACE_SYNTAX_UNSUPPORTED")
    values = []
    for choice in choices:
        values += _alternatives(pattern[:start] + choice + pattern[end + 1:])
        if len(values) > 64:
            raise ValueError("GLOB_EXPANSION_LIMIT")
    return values


def _regex(pattern):
    result = "(?:[^/]+/)*" if "/" not in pattern else ""
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern[i:i + 3] == "**/" and (i == 0 or pattern[i - 1] == "/"):
                result += "(?:[^/]+/)*"
                i += 3
                continue
            if pattern[i:] == "**" and (i == 0 or pattern[i - 1] == "/"):
                result += ".*"
                break
            result += "[^/]*"
        elif ch == "?":
            result += "[^/]"
        elif ch == "\\":
            i += 1
            if i >= len(pattern):
                raise ValueError("GLOB_DANGLING_ESCAPE")
            result += re.escape(pattern[i])
        elif ch in "[]{}":
            raise ValueError("GLOB_CHARACTER_CLASS_UNSUPPORTED")
        else:
            result += re.escape(ch)
        i += 1
    return result


def glob_regex(pattern, root):
    if not pattern or pattern.startswith(("!", "/")) or "\x00" in pattern or "\x00" in root:
        raise ValueError("GLOB_PATTERN_DIALECT_UNSUPPORTED")
    patterns = _alternatives(pattern)
    return "(?s)^" + re.escape(root.rstrip("/") + "/") + "(?:" + "|".join(_regex(p) for p in patterns) + ")$"
