"""Source tool parsers for six recorded harness formats."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache

import jsonschema

from .ir import Operation, SourceCall
from .patches import parse_patch

SOURCE_TOOLS = {
    "SWE-agent": {"bash", "str_replace_editor", "submit"},
    "mini-swe-agent": {"bash"},
    "OpenHands": {"execute_bash", "str_replace_editor", "finish", "think"},
    "Claude Code": {"Bash", "Read", "Edit", "Write", "Glob", "Grep"},
    "OpenCode": {"bash", "read", "edit", "write", "glob", "grep"},
    "Codex-format": {"shell", "read_file", "apply_patch", "rg", "delete"},
}
MINI_SCOPE = "Directory or environment variable changes are not persistent. Every action is executed in a new subshell."


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def content_text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(v.get("text", "") for v in value if isinstance(v, dict) and v.get("type") == "text")
    return ""


def definitions(tools):
    return {t.get("function", t)["name"]: t.get("function", t) for t in tools}


@lru_cache(maxsize=512)
def validator(schema):
    parsed = json.loads(schema)
    return jsonschema.validators.validator_for(parsed)(parsed)


def declared_root(messages):
    text = "\n".join(content_text(m.get("content")) for m in messages if m["role"] in {"user", "system"})
    text = text.replace("\\n", "\n")
    match = re.search(r"<uploaded_files>\s*(/[^\n<]+)", text) or re.search(r"Cloned path:\s*(/[^\n]+)", text)
    return match[1].strip() if match else "/testbed" if "- MODIFY: Regular source code files in /testbed" in text else None


def checked_keys(args, allowed):
    extra = set(args) - set(allowed.split())
    if extra:
        raise ValueError("UNSUPPORTED_ARGUMENTS:" + ",".join(sorted(extra)))


def string(value, name, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value) or "\x00" in value:
        raise ValueError("INVALID_TEXT:" + name)
    return value


def lower_call(source, name, args, definition, operating_text="") -> Operation:
    """Parse a source call into an operation and its declared contracts."""
    if source not in SOURCE_TOOLS or name not in SOURCE_TOOLS[source]:
        return Operation("unsupported", {"source_tool": name, "arguments": args}, {"reason": "SOURCE_TOOL_UNSUPPORTED"})
    desc = definition.get("description", "")
    path = args.get("path", args.get("file_path", args.get("filePath")))
    if name == "apply_patch":
        checked_keys(args, "patch cwd")
        return Operation("apply_patch", {"files": parse_patch(args["patch"]), "cwd": args.get("cwd")}, {
            "matching": "exact_lines", "contextless_update": "error", "syntax_check_python": "syntax-checked" in desc,
            "rollback_on_syntax_error": "rolled back" in desc, "source_format": "codex_patch",
            "newline_policy": "source_patch_text", "parent_creation": "source_runtime_unspecified"})
    if name in {"Write", "write"}:
        checked_keys(args, "file_path filePath content")
        if "overwrite the existing file" not in desc:
            raise ValueError("WRITE_MODE_UNDECLARED")
        return Operation("write_file", {"path": string(path, "path", True), "content": string(args.get("content"), "content"), "mode": "overwrite"}, {
            "encoding": "utf-8", "preserve_content_bytes": True,
            "read_before_overwrite": "Read tool first" in desc, "create_parents": False})
    if name in {"Edit", "edit"}:
        checked_keys(args, "file_path filePath old_string oldString new_string newString replace_all replaceAll")
        old = string(args.get("old_string", args.get("oldString")), "old", True)
        new = string(args.get("new_string", args.get("newString")), "new")
        return Operation("replace_text", {"path": string(path, "path", True), "old": old, "new": new,
                         "mode": "all" if args.get("replace_all", args.get("replaceAll", False)) else "unique"}, {
            "matching": "literal", "missing_match": "error_without_modification", "encoding": "utf-8",
            "read_before_edit": "Read` tool" in desc or "Read tool" in desc})
    if name in {"Glob", "glob"}:
        checked_keys(args, "pattern path")
        return Operation("find_files", {"root": args.get("path"), "pattern": string(args.get("pattern"), "pattern", True)}, {
            "dialect": source, "order": "mtime" if "modification time" in desc else "unspecified",
            "hidden_files": "unspecified", "ignore_files": "unspecified", "symlinks": "unspecified",
            "output": "file_paths"})
    if name == "delete":
        checked_keys(args, "file_path")
        return Operation("delete_file", {"path": string(path, "path", True)}, {
            "missing": "error", "kind": "regular_file", "recursive": False})
    if name in {"Read", "read", "read_file"}:
        checked_keys(args, "file_path filePath path offset limit")
        start = args.get("offset", 1)
        limit = args.get("limit", 2000 if source in {"Claude Code", "OpenCode"} else None)
        return Operation("read_file", {"path": string(path, "path", True), "start": start, "limit": limit}, {
            "line_base": 1, "negative_offset": source == "Codex-format", "presentation": "source_numbered_lines"})
    if name in {"str_replace_editor", "file_editor"}:
        checked_keys(args, "command path file_text old_str new_str insert_line view_range")
        command = args["command"]
        path = string(path, "path", True)
        if command == "create":
            return Operation("write_file", {"path": path, "content": string(args.get("file_text"), "content"), "mode": "create"}, {"existing": "error", "native_editor_semantics": True})
        if command == "str_replace":
            return Operation("replace_text", {"path": path, "old": string(args.get("old_str"), "old", True),
                             "new": string(args.get("new_str", ""), "new"), "mode": "unique"}, {"missing_match": "error_without_modification", "native_editor_semantics": True})
        if command == "view":
            view = args.get("view_range", [1, -1])
            if len(view) != 2:
                raise ValueError("INVALID_VIEW_RANGE")
            return Operation("read_file", {"path": path, "start": view[0], "limit": None if view[1] == -1 else view[1] - view[0] + 1}, {"line_base": 1, "native_editor_semantics": True})
        if command not in {"insert", "undo_edit"}:
            raise ValueError("EDITOR_COMMAND_UNSUPPORTED")
        return Operation("editor_action", dict(args), {"native_editor_semantics": True})
    if name in {"bash", "Bash", "shell", "execute_bash"}:
        checked_keys(args, "command description timeout timeout_ms command_timeout workdir mode is_input run_in_background dangerouslyDisableSandbox")
        command = string(args.get("command"), "command")
        argv = None
        if source == "Codex-format" and command.lstrip().startswith("["):
            argv = json.loads(command)
            if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and "\x00" not in v for v in argv):
                raise ValueError("INVALID_ARGV")
        scope = "per_call" if source in {"mini-swe-agent", "Codex-format"} else "persistent"
        if source == "mini-swe-agent" and MINI_SCOPE not in operating_text:
            scope = "unknown"
        is_input = args.get("is_input", False)
        if is_input not in (True, False, "true", "false"):
            raise ValueError("INVALID_IS_INPUT")
        timeout_keys = [k for k in ("timeout_ms", "timeout", "command_timeout") if args.get(k) is not None]
        if len(timeout_keys) > 1:
            raise ValueError("CONFLICTING_TIMEOUT_ARGUMENTS")
        timeout = args[timeout_keys[0]] if timeout_keys else None
        return Operation("run_command", {"command": None if argv else command, "argv": argv, "cwd": args.get("workdir"),
                         "is_input": is_input in (True, "true"), "timeout": timeout,
                         "timeout_unit": "seconds" if source == "OpenHands" else "milliseconds"}, {
            "scope": scope, "login": source == "Codex-format" and argv is None,
            "login_declared": "-lc" in desc, "background": bool(args.get("run_in_background")),
            "sandbox_override": bool(args.get("dangerouslyDisableSandbox")), "original_mode": args.get("mode")})
    if name in {"Grep", "grep", "rg"}:
        checked_keys(args, "pattern path glob include type output_mode -A -B -C -n -i multiline head_limit offset ignore_case max_count")
        return Operation("search_text", dict(args), {"source_dialect": source,
                         "order": "mtime" if "modification time" in desc else "unspecified"})
    if name in {"finish", "submit"}:
        return Operation("finish", {"message": args.get("message", "")})
    if name == "think":
        return Operation("think", {"thought": args.get("thought", "")})
    raise ValueError("SOURCE_TOOL_UNSUPPORTED")


def parse_record(record) -> list[SourceCall]:
    messages = record.get("messages")
    if messages is None:
        messages = json.loads(record["messages_json"])
    tools = record.get("tools")
    if tools is None:
        tools = json.loads(record["tools_json"])
    defs = definitions(tools)
    masks = record.get("message_loss_mask", [0] * len(messages))
    if len(masks) != len(messages):
        raise ValueError("MASK_LENGTH_MISMATCH")
    operating = "\n".join(content_text(m.get("content")) for m in messages if m["role"] in {"system", "user"})
    results = defaultdict(list)
    ids = Counter(c.get("id") for m in messages for c in m.get("tool_calls", []))
    for i, message in enumerate(messages):
        if message["role"] == "tool" and message.get("tool_call_id"):
            results[message["tool_call_id"]].append(i)
    calls = []
    for i, message in enumerate(messages):
        for j, call in enumerate(message.get("tool_calls", [])):
            function = call.get("function", {})
            name, cid = function.get("name", ""), call.get("id", "")
            definition = defs.get(name)
            try:
                if not cid or ids[cid] != 1:
                    raise ValueError("NONUNIQUE_CALL_ID")
                args = function.get("arguments")
                args = json.loads(args) if isinstance(args, str) else args
                if not isinstance(args, dict):
                    raise ValueError("ARGUMENTS_NOT_OBJECT")
                if definition is None:
                    raise ValueError("SOURCE_DEFINITION_MISSING")
                validator(dumps(definition.get("parameters", {}))).validate(args)
                operation = lower_call(record["source_scaffold"], name, args, definition, operating)
                if operation.kind == "finish" and any(m["role"] == "assistant" for m in messages[i + 1:]):
                    operation.contracts["nonterminal"] = True
            except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as error:
                operation = Operation("unsupported", {"source_tool": name}, {"reason": str(error).splitlines()[0][:240]})
            calls.append(SourceCall(cid, record["source_scaffold"], name, i, j, operation, call,
                                    sha(definition) if definition else None, results.get(cid, []), bool(masks[i])))
    groups = defaultdict(list)
    for call in calls:
        groups[call.message_index].append(call)
    for group in groups.values():
        if len(group) > 1:
            read_only = all(c.operation.kind in {"read_file", "find_files", "search_text", "think"} for c in group)
            for call in group:
                call.operation.contracts["source_batch_size"] = len(group)
                call.operation.contracts["source_batch_order_unverified"] = not read_only
    return calls
