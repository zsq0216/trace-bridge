"""Compile operations to OpenHands and SWE-agent tools."""
from __future__ import annotations

import ast
import hashlib
import posixpath
import shlex
import sys
from dataclasses import dataclass

from .ir import CompileContext, Operation, Plan, TargetAction
from .globs import glob_regex
from .patches import apply_update


@dataclass(frozen=True)
class Capabilities:
    name: str
    shell: str
    editor: str
    finish: str
    process_input: bool
    native_editor_commands: tuple[str, ...] = ("view", "create", "str_replace", "insert", "undo_edit")
    utilities: tuple[str, ...] = ("bash", "printf", "cat", "rm", "test", "sed-gnu", "rg", "mkdir", "mv", "mktemp", "tr")


TARGETS = {
    "openhands_sdk": Capabilities("openhands_sdk", "terminal", "file_editor", "finish", True),
    "sweagent": Capabilities("sweagent", "bash", "str_replace_editor", "submit", False),
}


def quote(value):
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("SHELL_ARGUMENT_MUST_BE_NUL_FREE_TEXT")
    return shlex.quote(value)


def absolute(path, cwd):
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("INVALID_PATH")
    if posixpath.isabs(path):
        return path
    if not cwd or not posixpath.isabs(cwd):
        raise ValueError("RELATIVE_PATH_REQUIRES_SOURCE_CWD")
    # Preserve symlink traversal through .. components.
    return posixpath.join(cwd, path)


def write_command(path, content):
    """Write literal content with its original trailing newline."""
    quote(content)
    if content.endswith("\n"):
        delimiter = "TB_CONTENT_" + hashlib.sha256(content.encode()).hexdigest()[:16]
        while delimiter in content.splitlines():
            delimiter += "_X"
        return "cat > " + quote(path) + " <<'" + delimiter + "'\n" + content + delimiter
    return "printf '%s' " + quote(content) + " > " + quote(path)


def subshell_command(body):
    if not isinstance(body, str) or "\x00" in body:
        raise ValueError("INVALID_SHELL_BODY")
    if not any(line.strip() and not line.lstrip().startswith("#") for line in body.splitlines()):
        body += "\n:"
    return "(\n" + body + "\n)"


def login_shell_command(body):
    """Keep the command literal while retaining bash -lc and its original stdin."""
    if not isinstance(body, str) or "\x00" in body:
        raise ValueError("INVALID_SHELL_BODY")
    delimiter = "TB_SCRIPT"
    lines = set(body.splitlines())
    while delimiter in lines:
        delimiter += "_END"
    # Remove only the newline added before the heredoc delimiter.
    return ("(\nIFS= read -r -d '' tb_source_script <<'" + delimiter + "' || :\n"
            + body + "\n" + delimiter
            + '\nbash -lc "${tb_source_script%?}"\n)')


def sed_literal(value, replacement=False):
    """Literal GNU BRE and replacement, including embedded newlines."""
    quote(value)
    special = "\\&|" if replacement else "\\.[*^$|"
    return "".join("\\n" if ch == "\n" else "\\" + ch if ch in special else ch for ch in value)


def replace_all_command(path, old, new):
    if not old:
        raise ValueError("EMPTY_REPLACE_PATTERN")
    pattern = sed_literal(old)
    substitution = "s|" + pattern + "|" + sed_literal(new, True) + "|g"
    # Write through the original inode rather than replacing it with sed -i.
    check = "\\|" + pattern + "|q0; $q1"
    guard = "test -f " + quote(path) + " && test -s " + quote(path) + " && LC_ALL=C sed -z -n " + quote(check) + " -- " + quote(path)
    return ("( " + guard + " || exit $?; tb_replace_tmp=$(mktemp) || exit $?; "
            + "trap 'rm -f -- \"$tb_replace_tmp\"' EXIT; LC_ALL=C sed -z "
            + quote(substitution) + " -- " + quote(path) + ' > "$tb_replace_tmp"'
            + ' && cat -- "$tb_replace_tmp" > ' + quote(path) + " )")


def native_literal_equivalent(target, before, old, new):
    """Check literal replacement on a full snapshot, accounting for CRLF and SWE tab expansion."""
    if not old or old == new or before.count(old) != 1 or "\r" in before + old + new:
        return False
    return target.name != "sweagent" or "\t" not in before + old + new


def delete_command(path):
    return "test -f " + quote(path) + " && rm -- " + quote(path)


def compile_operation(operation: Operation, target: str | Capabilities,
                      context: CompileContext | None = None) -> Plan:
    target = TARGETS[target] if isinstance(target, str) else target
    ctx = context or CompileContext()
    plan = Plan(target.name)
    p, c, kind = operation.parameters, operation.contracts, operation.kind
    planned_files = dict(ctx.files)

    def require(code):
        if code not in plan.requirements:
            plan.requirements.append(code)

    def verify(property_name):
        if property_name not in plan.verified_properties:
            plan.verified_properties.append(property_name)

    def native(command, path, **args):
        if command not in target.native_editor_commands:
            raise ValueError("TARGET_EDITOR_COMMAND_UNSUPPORTED:" + command)
        if command == "create" and not isinstance(args.get("file_text"), str):
            raise ValueError("EDITOR_FILE_TEXT_REQUIRED")
        if command == "str_replace" and (not isinstance(args.get("old_str"), str) or not args["old_str"]):
            raise ValueError("EDITOR_OLD_STRING_REQUIRED")
        if command == "insert" and (not isinstance(args.get("new_str"), str)
                                    or not isinstance(args.get("insert_line"), int)
                                    or isinstance(args.get("insert_line"), bool) or args["insert_line"] < 0):
            raise ValueError("EDITOR_INSERT_ARGUMENTS_INVALID")
        plan.actions.append(TargetAction(target.editor, {"command": command, "path": path, **args}, "native"))

    def shell(command, utility=None, **args):
        if utility and utility not in target.utilities:
            raise ValueError("TARGET_UTILITY_UNAVAILABLE:" + utility)
        plan.actions.append(TargetAction(target.shell, {"command": command, **args}, "simple_shell"))

    def literal_write(path, body):
        shell(write_command(path, body), "cat" if body.endswith("\n") else "printf")

    def access_guard(path, key):
        if c.get(key) and path not in ctx.read_paths:
            if ctx.source_succeeded:
                plan.notes.append("Read-access precondition established only by this bound source success.")
            else:
                require("SOURCE_READ_ACCESS_PRECONDITION")

    def snapshot(path):
        if path in planned_files and isinstance(planned_files[path], str):
            if isinstance(ctx.files.get(path), str):
                plan.preimage_sha256[path] = hashlib.sha256(ctx.files[path].encode()).hexdigest()
            return planned_files[path]
        return None

    try:
        if c.get("source_batch_order_unverified"):
            require("SOURCE_BATCH_EFFECT_ORDER")
        if kind == "unsupported":
            raise ValueError(c.get("reason", "SOURCE_TOOL_UNSUPPORTED"))
        if kind == "write_file":
            path = absolute(p["path"], ctx.cwd)
            access_guard(path, "read_before_overwrite")
            if p["mode"] == "create":
                native("create", path, file_text=p["content"])
            elif p["mode"] == "overwrite":
                literal_write(path, p["content"])
                plan.notes.append("UTF-8; parent must exist; exact trailing newline preserved.")
            else:
                raise ValueError("WRITE_MODE_UNSUPPORTED")
        elif kind == "replace_text":
            path = absolute(p["path"], ctx.cwd)
            access_guard(path, "read_before_edit")
            before = snapshot(path)
            if p["mode"] not in {"unique", "all"} or not p["old"]:
                raise ValueError("REPLACE_MODE_OR_PATTERN_UNSUPPORTED")
            if before is not None and (before.count(p["old"]) == 0 or (p["mode"] == "unique" and before.count(p["old"]) != 1)):
                raise ValueError("REPLACE_PREIMAGE_MATCH_CONTRACT_FAILED")
            if before is not None and native_literal_equivalent(target, before, p["old"], p["new"]):
                native("str_replace", path, old_str=p["old"], new_str=p["new"])
                verify("literal_replace_on_full_preimage")
            elif p["mode"] == "unique" and before is not None:
                literal_write(path, before.replace(p["old"], p["new"], 1))
                verify("literal_replace_on_full_preimage")
                plan.notes.append("Snapshot-based write avoids the target editor's tab/newline normalization.")
            elif p["mode"] == "unique":
                native("str_replace", path, old_str=p["old"], new_str=p["new"])
                if not c.get("native_editor_semantics"):
                    require("NATIVE_EDITOR_TEXT_NORMALIZATION_EQUIVALENCE")
            elif p["mode"] == "all":
                if not {"mktemp", "cat", "rm", "test"} <= set(target.utilities):
                    raise ValueError("TARGET_REPLACE_UTILITIES_UNAVAILABLE")
                shell(replace_all_command(path, p["old"], p["new"]), "sed-gnu")
                plan.notes.append("GNU sed stdout staged then written through original pathname; no concurrent writer during the operation. Filesystem timestamps/security-policy side effects are outside byte equivalence.")
        elif kind == "delete_file":
            if c.get("recursive") or c.get("kind", "regular_file") != "regular_file":
                raise ValueError("DELETE_KIND_UNSUPPORTED")
            shell(delete_command(absolute(p["path"], ctx.cwd)), "rm")
        elif kind == "find_files":
            root = absolute(p.get("root") or ".", ctx.cwd)
            pattern = p["pattern"]
            if pattern.startswith(("/", "!")):
                raise ValueError("GLOB_PATTERN_DIALECT_UNSUPPORTED")
            argv = ["rg", "--no-config", "--files"]
            policy = ctx.glob_policy
            if policy is not None:
                if policy.engine != "ripgrep" or policy.order not in {"mtime_desc", "mtime_asc", "path"} or policy.empty_matches not in {"success_empty", "error"}:
                    raise ValueError("GLOB_POLICY_UNSUPPORTED")
                if c.get("order") == "mtime" and policy.order == "path":
                    raise ValueError("GLOB_ORDER_CONTRACT_CONFLICT")
                declared = {"hidden_files": "include" if policy.hidden else "exclude",
                            "ignore_files": "respect" if policy.respect_ignore else "ignore",
                            "symlinks": "follow" if policy.follow_symlinks else "no_follow"}
                if any(c.get(key, "unspecified") not in {"unspecified", value} for key, value in declared.items()):
                    raise ValueError("GLOB_FILE_SELECTION_CONTRACT_CONFLICT")
                if policy.hidden:
                    argv.append("--hidden")
                if not policy.respect_ignore:
                    argv.append("--no-ignore")
                if policy.follow_symlinks:
                    argv.append("--follow")
                argv += ["--sortr" if policy.order == "mtime_desc" else "--sort", "path" if policy.order == "path" else "modified"]
            elif c.get("order") == "mtime":
                argv += ["--sortr", "modified"]
            if policy is None:
                argv += ["--glob", pattern, "--", root]
                command = shlex.join(argv)
            else:
                if "tr" not in target.utilities:
                    raise ValueError("TARGET_UTILITY_UNAVAILABLE:tr")
                argv += ["--null", "--", root]
                matcher = shlex.join(["rg", "--no-config", "--null-data", "--no-line-number", "--no-heading", "--color", "never", "-e", glob_regex(pattern, root)])
                match_test = "-le 1" if policy.empty_matches == "success_empty" else "-eq 0"
                command = ("( " + shlex.join(argv) + " | " + matcher + " | tr '\\0' '\\n'; "
                           + 'tb_glob_status=("${PIPESTATUS[@]}"); test "${tb_glob_status[0]}" -le 1'
                           + ' && test "${tb_glob_status[1]}" ' + match_test
                           + ' && test "${tb_glob_status[2]}" -eq 0 )')
            shell(command, "rg")
            if policy is None:
                require("SOURCE_GLOB_DIALECT_HIDDEN_IGNORE_SYMLINK_AND_ORDER_POLICY")
            else:
                verify("explicit_ripgrep_search_policy")
            plan.notes.append("Unknown source glob policies remain explicit; local fixtures validate only the supplied search policy.")
        elif kind == "apply_patch":
            cwd = p.get("cwd")
            cwd = absolute(cwd, ctx.cwd) if cwd else ctx.cwd
            syntax_complete = True
            python_changed = False
            def check_syntax(path, body):
                nonlocal syntax_complete, python_changed
                if not path.endswith(".py") or not (c.get("syntax_check_python") or c.get("rollback_on_syntax_error")):
                    return
                python_changed = True
                if body is not None and ctx.source_patch_succeeded:
                    verify("python_syntax_from_bound_source_patch_success")
                    return
                if body is None or ctx.python_feature_version is None:
                    syntax_complete = False
                    return
                version = ctx.python_feature_version
                if version[0] != 3 or not 7 <= version[1] <= sys.version_info.minor:
                    raise ValueError("PYTHON_SYNTAX_VERSION_UNSUPPORTED")
                tree = ast.parse(body, filename=path, feature_version=version)
                compile(tree, path, "exec")
                verify("python_syntax_on_full_postimage")
            for change in p["files"]:
                path = absolute(change["path"], cwd)
                if change["kind"] == "delete":
                    shell(delete_command(path), "rm")
                    planned_files[path] = None
                elif change["kind"] == "add":
                    native("create", path, file_text=change["content"])
                    check_syntax(path, change["content"])
                    if path not in planned_files or planned_files[path] is not None or posixpath.dirname(path) not in ctx.directories:
                        require("PATCH_ADD_EXISTENCE_AND_PARENT_DIRECTORY_POLICY")
                    else:
                        verify("patch_add_absent_path_and_existing_parent")
                    planned_files[path] = change["content"]
                elif change["kind"] == "update":
                    if path in planned_files and planned_files[path] is None:
                        raise ValueError("PATCH_UPDATE_PATH_ABSENT")
                    before = snapshot(path)
                    if before is None:
                        check_syntax(change.get("move_to", path), None)
                        require("PATCH_FULL_PREIMAGE_AND_LINE_MATCH_EQUIVALENCE")
                        require("NATIVE_EDITOR_TEXT_NORMALIZATION_EQUIVALENCE")
                        for hunk in change["hunks"]:
                            old = "\n".join(hunk["old_lines"])
                            new = "\n".join(hunk["new_lines"])
                            if not old:
                                raise ValueError("PATCH_EMPTY_NATIVE_REPLACE")
                            native("str_replace", path, old_str=old, new_str=new)
                    else:
                        after = apply_update(before, change["hunks"])
                        check_syntax(change.get("move_to", path), after)
                        pairs = [("\n".join(h["old_lines"]), "\n".join(h["new_lines"])) for h in change["hunks"]]
                        working = before
                        exact = True
                        for old, new in pairs:
                            if not native_literal_equivalent(target, working, old, new):
                                exact = False
                                break
                            working = working.replace(old, new, 1)
                        if exact and working == after:
                            for old, new in pairs:
                                native("str_replace", path, old_str=old, new_str=new)
                        elif before and native_literal_equivalent(target, before, before, after):
                            native("str_replace", path, old_str=before, new_str=after)
                        else:
                            literal_write(path, after)
                        planned_files[path] = after
                        verify("patch_line_and_byte_effect_on_full_preimage")
                    if change.get("move_to"):
                        destination = absolute(change["move_to"], cwd)
                        shell(shlex.join(["mv", "--", path, destination]), "mv")
                        if destination not in planned_files or planned_files[destination] is not None or posixpath.dirname(destination) not in ctx.directories:
                            require("PATCH_MOVE_DESTINATION_AND_PARENT_POLICY")
                        else:
                            verify("patch_move_absent_destination_and_existing_parent")
                        if path in planned_files:
                            planned_files[destination] = planned_files[path]
                        planned_files[path] = None
                else:
                    raise ValueError("PATCH_OPERATION_UNSUPPORTED")
            if not syntax_complete:
                require("PATCH_SYNTAX_CHECK_AND_ROLLBACK")
            elif python_changed:
                plan.notes.append("Syntax success established for the full postimage by supplied grammar or bound source patch success; no claim of a runtime rollback capability.")
            plan.notes.append("Snapshot-derived actions require unchanged preimages and sequential execution without external writers.")
        elif kind == "read_file":
            path = absolute(p["path"], ctx.cwd)
            start, limit = p.get("start", 1), p.get("limit")
            if not isinstance(start, int) or isinstance(start, bool) or start < 1:
                raise ValueError("READ_OFFSET_REQUIRES_FILE_LENGTH")
            if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
                raise ValueError("INVALID_READ_LIMIT")
            args = {} if start == 1 and limit is None else {"view_range": [start, start + limit - 1 if limit else -1]}
            native("view", path, **args)
            if not c.get("native_editor_semantics"):
                plan.notes.append("Source display formatting is retained in IR; this compiler does not fabricate target observations.")
        elif kind == "editor_action":
            args = dict(p)
            command, path = args.pop("command"), absolute(args.pop("path"), ctx.cwd)
            native(command, path, **args)
        elif kind == "run_command":
            if c.get("background") or c.get("sandbox_override"):
                raise ValueError("SOURCE_RUNTIME_EXTENSION_REQUIRED")
            if c.get("scope") == "unknown":
                raise ValueError("SOURCE_SHELL_SCOPE_UNDECLARED")
            if p.get("is_input"):
                if not target.process_input:
                    raise ValueError("TARGET_HAS_NO_PROCESS_INPUT")
                if p.get("argv") or p.get("cwd"):
                    raise ValueError("PROCESS_INPUT_WITH_COMMAND_CONTEXT")
                plan.actions.append(TargetAction(target.shell, {"command": p["command"], "is_input": True}, "native"))
            else:
                command = shlex.join(p["argv"]) if p.get("argv") else p["command"]
                cwd = absolute(p["cwd"], ctx.cwd) if p.get("cwd") else ctx.cwd if c.get("scope") == "per_call" else None
                body = ("cd -- " + quote(cwd) + " || exit $?\n" if cwd else "") + command
                if c.get("scope") == "per_call":
                    if c.get("login") and not c.get("login_declared"):
                        raise ValueError("SOURCE_LOGIN_SHELL_UNDECLARED")
                    if c.get("login"):
                        if "bash" not in target.utilities:
                            raise ValueError("TARGET_UTILITY_UNAVAILABLE:bash")
                        command = login_shell_command(body)
                    else:
                        command = subshell_command(body)
                elif cwd:
                    command = subshell_command(body)
                plan.actions.append(TargetAction(target.shell, {"command": command}, "source_shell"))
            if p.get("timeout") is not None:
                require("SOURCE_TIMEOUT_LIFECYCLE_EQUIVALENCE")
        elif kind == "search_text":
            if c.get("source_dialect") == "Claude Code":
                mode = p.get("output_mode", "files_with_matches")
            else:
                mode = "content"
            argv = ["rg", {"content": "-n", "files_with_matches": "-l", "count": "-c"}[mode]]
            for key, flag in (("glob", "--glob"), ("include", "--glob"), ("type", "--type"), ("max_count", "--max-count"), ("-A", "-A"), ("-B", "-B"), ("-C", "-C")):
                if key in p:
                    argv += [flag, str(p[key])]
            if p.get("-i", p.get("ignore_case", False)):
                argv.append("-i")
            if p.get("multiline"):
                argv += ["-U", "--multiline-dotall"]
            if p.get("-n") is False and "-n" in argv:
                argv.remove("-n")
            argv += ["--", p["pattern"], absolute(p.get("path") or ".", ctx.cwd)]
            shell(shlex.join(argv), "rg")
            if p.get("offset") or p.get("head_limit") or c.get("order") == "mtime":
                require("SOURCE_SEARCH_ORDER_AND_PAGINATION")
        elif kind == "finish":
            if c.get("nonterminal"):
                raise ValueError("NONTERMINAL_SOURCE_SUBMISSION")
            args = {"message": p["message"]} if target.name == "openhands_sdk" else {}
            plan.actions.append(TargetAction(target.finish, args, "native"))
        elif kind == "think":
            if target.name != "openhands_sdk":
                raise ValueError("TARGET_HAS_NO_THINK_TOOL")
            plan.actions.append(TargetAction("think", dict(p), "native"))
        else:
            raise ValueError("IR_OPERATION_UNSUPPORTED")
    except (ValueError, KeyError, TypeError, SyntaxError) as error:
        plan.actions.clear()
        plan.reason = str(error).splitlines()[0][:240]
    return plan
