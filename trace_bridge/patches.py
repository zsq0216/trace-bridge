"""Parse and apply structured Codex patches."""
from __future__ import annotations

import re


class PatchError(ValueError):
    pass


def parse_patch(text: str) -> list[dict]:
    if not isinstance(text, str):
        raise PatchError("PATCH_NOT_TEXT")
    lines = text.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise PatchError("PATCH_ENVELOPE_REQUIRED")
    files, current, hunk = [], None, None
    opened, group = False, -1
    for line in lines:
        if line == "*** Begin Patch":
            if opened:
                raise PatchError("NESTED_PATCH_ENVELOPE")
            opened, current, hunk = True, None, None
            group += 1
            continue
        if line == "*** End Patch":
            if not opened:
                raise PatchError("UNMATCHED_PATCH_END")
            opened, current, hunk = False, None, None
            continue
        if not opened:
            if line.strip():
                raise PatchError("TEXT_OUTSIDE_PATCH")
            continue
        if not line and hunk is not None and hunk["at_eof"]:
            continue
        if not line and current is not None and current["kind"] == "update":
            line = " "
        if line.startswith(("*** Add File: ", "*** Update File: ", "*** Delete File: ")):
            kind, path = line[4:].split(": ", 1)
            if not path or "\x00" in path:
                raise PatchError("PATCH_PATH_INVALID")
            current = {"kind": kind.split()[0].lower(), "path": path, "hunks": [], "group": group}
            files.append(current)
            hunk = None
        elif line.startswith("*** Move to: "):
            if current is None or current["kind"] != "update" or "move_to" in current:
                raise PatchError("INVALID_MOVE")
            current["move_to"] = line[len("*** Move to: "):]
            if not current["move_to"] or "\x00" in current["move_to"]:
                raise PatchError("INVALID_MOVE_PATH")
        elif line == "*** End of File":
            if hunk is None or current["kind"] != "update":
                raise PatchError("EOF_WITHOUT_UPDATE_HUNK")
            hunk["at_eof"] = True
        elif line == "@@" or line.startswith("@@ "):
            if current is None or current["kind"] != "update":
                raise PatchError("HUNK_WITHOUT_UPDATE")
            hint = line[2:].strip()
            numeric = bool(re.fullmatch(r"[-+]?\d+(?:,\d+)?(?:\s+\+\d+(?:,\d+)?)?\s*@@.*", hint))
            hunk = {"context_hint": "" if numeric else hint, "header": line,
                    "header_kind": "line_numbers" if numeric else "context", "at_eof": False, "lines": []}
            current["hunks"].append(hunk)
        elif current is not None and line[:1] in {" ", "+", "-"}:
            if current["kind"] == "delete":
                raise PatchError("DELETE_WITH_CONTENT")
            if current["kind"] == "add" and not line.startswith("+"):
                raise PatchError("ADD_REQUIRES_PLUS_LINES")
            if hunk is None:
                hunk = {"context_hint": "", "at_eof": False, "lines": []}
                current["hunks"].append(hunk)
            if hunk["at_eof"]:
                raise PatchError("CONTENT_AFTER_EOF")
            hunk["lines"].append({"tag": line[0], "text": line[1:]})
        else:
            raise PatchError("UNSUPPORTED_PATCH_LINE:" + line[:60])
    if not files:
        raise PatchError("EMPTY_PATCH")
    if opened:
        raise PatchError("UNTERMINATED_PATCH")
    for change in files:
        if change["kind"] == "update" and not change["hunks"]:
            raise PatchError("EMPTY_UPDATE")
        for chunk in change["hunks"]:
            chunk["old_lines"] = [v["text"] for v in chunk["lines"] if v["tag"] in {" ", "-"}]
            chunk["new_lines"] = [v["text"] for v in chunk["lines"] if v["tag"] in {" ", "+"}]
            if change["kind"] == "update" and not chunk["old_lines"]:
                raise PatchError("CONTEXTLESS_UPDATE")
        if change["kind"] == "add":
            added = [v["text"] for h in change["hunks"] for v in h["lines"]]
            change["content"] = "\n".join(added) + ("\n" if added else "")
    return files


def apply_update(before: str, hunks: list[dict]) -> str:
    """Apply exact line hunks to a supplied file snapshot."""
    if "\r" in before:
        raise PatchError("PATCH_NEWLINE_POLICY_UNVERIFIED")
    lines = before.splitlines()
    cursor = 0
    for hunk in hunks:
        old, new = hunk["old_lines"], hunk["new_lines"]
        hint = hunk["context_hint"]
        if hint:
            positions = [i for i in range(cursor, len(lines)) if lines[i] == hint]
            if len(positions) != 1:
                raise PatchError("PATCH_HINT_NOT_UNIQUE")
            cursor = positions[0] + 1
        candidates = [i for i in range(cursor, len(lines) - len(old) + 1)
                      if lines[i:i + len(old)] == old
                      and (not hunk["at_eof"] or i + len(old) == len(lines))]
        if len(candidates) != 1:
            raise PatchError("PATCH_CONTEXT_NOT_UNIQUE")
        start = candidates[0]
        lines[start:start + len(old)] = new
        cursor = start + len(new)
    return "\n".join(lines) + ("\n" if lines else "")
