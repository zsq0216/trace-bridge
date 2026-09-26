import ast
import gzip
import json
import random
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trace_bridge.ir import CompileContext, Operation
from trace_bridge.patches import PatchError, apply_update, parse_patch
from trace_bridge.run import run
from trace_bridge.sources import MINI_SCOPE, lower_call, parse_record
from trace_bridge.targets import TARGETS, compile_operation, replace_all_command, write_command


def execute(command, directory):
    return subprocess.run(["bash", "--noprofile", "--norc", "-c", command], cwd=directory,
                          capture_output=True, timeout=10)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("body", ["", "no final newline", "one\n", "\n\n", "中文🙂\r\n",
    "'$HOME' `touch injected` $(touch injected) \\ %s &\n", "line1\nline2", "EOF\nTB_CONTENT\n"])
def test_overwrite_literal_bytes_and_no_code_execution(tmp_path, target, body):
    path = tmp_path / "file ' ; $name"
    path.write_text("old contents")
    plan = compile_operation(Operation("write_file", {"path": str(path), "content": body, "mode": "overwrite"}), target)
    assert plan.status == "ready"
    assert plan.actions[0].tool == TARGETS[target].shell
    result = execute(plan.actions[0].arguments["command"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == body.encode()
    assert not (tmp_path / "injected").exists()
    wire = plan.wire_calls("a")[0]
    assert json.loads(wire["function"]["arguments"]) == plan.actions[0].arguments


def test_overwrite_does_not_create_parents_or_append_newline(tmp_path):
    path = tmp_path / "missing" / "child"
    assert execute(write_command(str(path), "text"), tmp_path).returncode != 0
    assert not path.parent.exists()


@pytest.mark.parametrize("old,new,before", [
    ("a", "b", "aaa\na"), ("a\nb", "x\ny\n", "a\nba\nb"),
    (r".*[a]$^\|", r"\1&$|", r"prefix.*[a]$^\|suffix.*[a]$^\|"),
    ("'", "$(touch injected)`touch injected`", "'\n'"),
    ("中文", "🙂", "中文\r\n中文"), ("aa", "a", "aaaaa"),
    ("\r\n", "\n", "a\r\nb\r\n"), ("\\n", "\n", "a\\nb"),
    ("old", "", "old\x00old\x00tail"), ("\n", "x", "a\nb\n"),
])
def test_replace_all_matches_literal_string_semantics(tmp_path, old, new, before):
    path = tmp_path / "file ' ; $x"
    path.write_bytes(before.encode())
    result = execute(replace_all_command(str(path), old, new), tmp_path)
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == before.replace(old, new).encode()
    assert not (tmp_path / "injected").exists()


def test_random_literal_replace_cases(tmp_path):
    rng = random.Random(398)
    alphabet = "ab .*[]^$|\\&'\n\r中文🙂+-()?{}"
    path = tmp_path / "property.txt"
    for _ in range(100):
        old = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))
        new = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 10)))
        before = "prefix" + old + "middle\n" + old + "suffix"
        path.write_bytes(before.encode())
        result = execute(replace_all_command(str(path), old, new), tmp_path)
        assert result.returncode == 0, (old, new, result.stderr)
        assert path.read_bytes() == before.replace(old, new).encode(), (old, new)


@pytest.mark.parametrize("before", ["", "other\n", "first\x00second", "partial a\npartial b"])
def test_replace_missing_match_leaves_file_unchanged(tmp_path, before):
    path = tmp_path / "f"
    path.write_bytes(before.encode())
    result = execute(replace_all_command(str(path), "a\nb", "NEW"), tmp_path)
    assert result.returncode != 0
    assert path.read_bytes() == before.encode()


def test_replace_missing_file_and_symlink(tmp_path):
    path = tmp_path / "missing"
    assert execute(replace_all_command(str(path), "x", "y"), tmp_path).returncode != 0
    assert not path.exists()
    path.write_text("xx")
    link = tmp_path / "link"
    link.symlink_to(path)
    assert execute(replace_all_command(str(link), "x", "y"), tmp_path).returncode == 0
    assert link.is_symlink() and path.read_text() == "yy"


@pytest.mark.parametrize("target", TARGETS)
def test_create_uses_native_but_overwrite_does_not(tmp_path, target):
    params = {"path": str(tmp_path / "f"), "content": "x", "mode": "create"}
    plan = compile_operation(Operation("write_file", params), target)
    assert plan.actions[0].arguments["command"] == "create"
    assert plan.actions[0].implementation == "native"
    guarded = compile_operation(Operation("write_file", {**params, "mode": "overwrite"}, {"read_before_overwrite": True}), target)
    assert guarded.status == "conditional"
    with pytest.raises(ValueError, match="PLAN_NOT_READY"):
        guarded.wire_calls("x")


@pytest.mark.parametrize("target", TARGETS)
def test_delete_file_missing_directory_and_quoted_path(tmp_path, target):
    path = tmp_path / "f ' ; touch injected"
    path.write_text("x")
    plan = compile_operation(Operation("delete_file", {"path": str(path)}), target)
    cmd = plan.actions[0].arguments["command"]
    assert execute(cmd, tmp_path).returncode == 0
    assert not path.exists() and not (tmp_path / "injected").exists()
    assert execute(cmd, tmp_path).returncode != 0
    path.mkdir()
    assert execute(cmd, tmp_path).returncode != 0
    assert path.is_dir()


PATCH = "*** Begin Patch\n*** Update File: a.py\n@@\n-old = 1\n+new = 2\n*** End Patch"


def test_patch_frontend_keeps_operations_and_declared_guards():
    op = lower_call("Codex-format", "apply_patch", {"patch": PATCH, "cwd": "/repo"},
                    {"description": "Python files are syntax-checked and the patch is rolled back on error."})
    assert op.kind == "apply_patch"
    assert op.parameters["files"][0]["hunks"][0]["old_lines"] == ["old = 1"]
    assert op.contracts["syntax_check_python"] and op.contracts["rollback_on_syntax_error"]
    assert "command" not in op.parameters
    assert "python3 -c" not in json.dumps(asdict(op))
    restored = Operation(**json.loads(json.dumps(asdict(op))))
    assert restored == op


@pytest.mark.parametrize("target", TARGETS)
def test_patch_native_candidate_and_guard_not_silently_dropped(target):
    op = Operation("apply_patch", {"files": parse_patch(PATCH), "cwd": "/repo"},
                   {"syntax_check_python": True, "rollback_on_syntax_error": True})
    plan = compile_operation(op, target)
    assert plan.status == "conditional"
    assert "PATCH_SYNTAX_CHECK_AND_ROLLBACK" in plan.requirements
    assert all(a.tool == TARGETS[target].editor for a in plan.actions)
    assert plan.actions[0].arguments == {"command": "str_replace", "path": "/repo/a.py", "old_str": "old = 1", "new_str": "new = 2"}
    with pytest.raises(ValueError, match="PLAN_NOT_READY"):
        plan.wire_calls("patch1")


def test_patch_full_preimage_validates_uniqueness_eof_and_python_syntax():
    files = parse_patch(PATCH)
    ctx = CompileContext(files={"/repo/a.py": "old = 1\n"}, python_feature_version=(3, 11))
    op = Operation("apply_patch", {"files": files, "cwd": "/repo"}, {"syntax_check_python": True})
    plan = compile_operation(op, "sweagent", ctx)
    assert "PATCH_FULL_PREIMAGE_AND_LINE_MATCH_EQUIVALENCE" not in plan.requirements
    assert plan.preimage_sha256
    bad = Operation("apply_patch", {"files": parse_patch(PATCH.replace("new = 2", "new = (")), "cwd": "/repo"}, {"syntax_check_python": True})
    failed = compile_operation(bad, "sweagent", ctx)
    assert failed.status == "unsupported" and not failed.actions
    with pytest.raises(PatchError, match="NOT_UNIQUE"):
        apply_update("old = 1\nold = 1\n", files[0]["hunks"])
    eof = parse_patch(PATCH.replace("*** End Patch", "*** End of File\n*** End Patch"))
    with pytest.raises(PatchError):
        apply_update("old = 1\ntrailer\n", eof[0]["hunks"])


def test_patch_add_delete_move_and_empty_add_are_structured():
    patch = "*** Begin Patch\n*** Add File: empty\n*** Delete File: gone\n*** Update File: before\n*** Move to: after\n@@\n-a\n+b\n*** End Patch"
    files = parse_patch(patch)
    assert files[0]["content"] == ""
    assert files[1]["kind"] == "delete"
    assert files[2]["move_to"] == "after"
    plan = compile_operation(Operation("apply_patch", {"files": files, "cwd": "/repo"}), "sweagent")
    assert len(plan.actions) == 4
    assert all("trajport_patch" not in str(a.arguments) for a in plan.actions)


def test_recorded_patch_groups_numeric_headers_and_blank_context():
    patch = "*** Begin Patch\n*** Update File: a\n@@ -1,2 +1,2 @@\n\n-old\n+new\n*** End Patch\n*** Begin Patch\n*** Delete File: b\n*** End Patch"
    files = parse_patch(patch)
    assert [f["group"] for f in files] == [0, 1]
    hunk = files[0]["hunks"][0]
    assert hunk["header_kind"] == "line_numbers" and hunk["context_hint"] == ""
    assert hunk["old_lines"] == ["", "old"]
    assert apply_update("\nold\n", files[0]["hunks"]) == "\nnew\n"


def test_patch_repeated_file_uses_planned_intermediate_content():
    patch = PATCH + "\n" + PATCH.replace("old = 1", "new = 2").replace("+new = 2", "+final = 3")
    ctx = CompileContext(cwd="/repo", files={"/repo/a.py": "old = 1\n"})
    plan = compile_operation(Operation("apply_patch", {"files": parse_patch(patch)}), "sweagent", ctx)
    assert plan.reason is None and len(plan.actions) == 2
    assert plan.actions[1].arguments["old_str"] == "new = 2"
    assert ctx.files["/repo/a.py"] == "old = 1\n"


def test_mutating_batch_order_and_nonterminal_submit_stay_explicit():
    r = make_record("SWE-agent", "bash", {"command": "pwd"})
    messages = json.loads(r["messages_json"])
    messages[1]["tool_calls"].append({"id": "call2", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}})
    r["messages_json"] = json.dumps(messages)
    assert all("SOURCE_BATCH_EFFECT_ORDER" in compile_operation(c.operation, "sweagent").requirements for c in parse_record(r))
    r = make_record("SWE-agent", "submit", {})
    messages = json.loads(r["messages_json"])
    messages.append({"role": "assistant", "content": "try again"})
    r["messages_json"], r["message_loss_mask"] = json.dumps(messages), [0, 1, 0, 1]
    assert compile_operation(parse_record(r)[0].operation, "sweagent").reason == "NONTERMINAL_SOURCE_SUBMISSION"


@pytest.mark.parametrize("patch", ["", "@@\n-a\n+b", "*** Begin Patch\n*** Update File: a\n@@\n+b\n*** End Patch",
    "*** Begin Patch\n*** Delete File: a\n+bad\n*** End Patch"])
def test_invalid_patch_is_rejected_without_partial_plan(patch):
    with pytest.raises(PatchError):
        parse_patch(patch)


def test_late_failure_discards_earlier_patch_actions():
    files = parse_patch("*** Begin Patch\n*** Add File: /ok\n+x\n*** Update File: relative\n@@\n-a\n+b\n*** End Patch")
    plan = compile_operation(Operation("apply_patch", {"files": files}), "sweagent")
    assert plan.status == "unsupported" and not plan.actions


def test_glob_preserves_mtime_and_dialect_instead_of_python_glob():
    op = lower_call("Claude Code", "Glob", {"pattern": "**/*.py", "path": "/repo"}, {"description": "Returns matching file paths sorted by modification time"})
    assert op.contracts["order"] == "mtime" and op.contracts["dialect"] == "Claude Code"
    plan = compile_operation(op, "openhands_sdk")
    assert plan.status == "conditional"
    assert "--sortr modified" in plan.actions[0].arguments["command"]
    assert "python" not in plan.actions[0].arguments["command"]
    no_rg = replace(TARGETS["sweagent"], utilities=("bash",))
    assert compile_operation(op, no_rg).status == "unsupported"


def test_shell_scope_cwd_and_failed_cd(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    op = lower_call("mini-swe-agent", "bash", {"command": "pwd"}, {"description": "bash"}, MINI_SCOPE)
    plan = compile_operation(op, "sweagent", CompileContext(cwd=str(base)))
    assert execute(plan.actions[0].arguments["command"], tmp_path).stdout.decode().strip() == str(base)
    bad = Operation("run_command", {"command": "true\ntouch injected", "cwd": str(tmp_path / "missing")}, {"scope": "persistent"})
    cmd = compile_operation(bad, "sweagent").actions[0].arguments["command"]
    assert execute(cmd, tmp_path).returncode != 0 and not (tmp_path / "injected").exists()
    cmd = compile_operation(Operation("run_command", {"command": "cd /; export TB_TEST_VAR=abc"}, {"scope": "per_call"}), "sweagent").actions[0].arguments["command"]
    result = execute(cmd + "\nprintf '%s\\n' \"$PWD\" \"${TB_TEST_VAR-unset}\"", tmp_path)
    assert result.stdout.decode().splitlines() == [str(tmp_path), "unset"]


def test_process_input_and_timeout_are_not_lost():
    op = lower_call("OpenHands", "execute_bash", {"command": "C-c", "is_input": "true", "timeout": 5}, {"description": "shell"})
    oh = compile_operation(op, "openhands_sdk")
    assert oh.actions[0].arguments["is_input"] is True
    assert "SOURCE_TIMEOUT_LIFECYCLE_EQUIVALENCE" in oh.requirements
    assert compile_operation(op, "sweagent").status == "unsupported"


def make_record(source, tool, arguments, definition=None):
    definition = definition or {"name": tool, "description": "runs a bash command", "parameters": {"type": "object"}}
    messages = [{"role": "user", "content": "<uploaded_files>\n/repo\n</uploaded_files>\n" + MINI_SCOPE},
                {"role": "assistant", "tool_calls": [{"id": "call1", "function": {"name": tool, "arguments": json.dumps(arguments)}}]},
                {"role": "tool", "tool_call_id": "call1", "content": "recorded result"}]
    return {"id": source, "source_scaffold": source, "split": "train", "message_loss_mask": [0, 1, 0],
            "messages_json": json.dumps(messages), "tools_json": json.dumps([definition if source == "OpenCode" else {"type": "function", "function": definition}])}


def test_six_sources_and_two_targets_compile_without_baseline(tmp_path):
    records = [make_record(source, tool, {"command": "pwd"}, {"name": tool, "description": "runs with -lc", "parameters": {"type": "object"}})
               for source, tool in [("SWE-agent", "bash"), ("mini-swe-agent", "bash"), ("OpenHands", "execute_bash"),
                                    ("Claude Code", "Bash"), ("OpenCode", "bash"), ("Codex-format", "shell")]]
    source = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist(records), source)
    manifest = run(source, tmp_path / "out", list(TARGETS))
    assert manifest["totals"] == {"rows": 6, "source_calls": 6}
    assert set(manifest["by_source"]) == {r["source_scaffold"] for r in records}
    assert manifest["training_ready"] is False
    for target in TARGETS:
        with gzip.open(tmp_path / "out" / (target + ".plans.jsonl.gz"), "rt") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == 6 and all(r["plans"][0]["status"] == "ready" for r in rows)
        assert all(r["plans"][0]["source_observation_indices"] == [2] for r in rows)


def test_source_unknown_arguments_and_duplicate_ids_are_not_silently_dropped():
    record = make_record("Codex-format", "delete", {"file_path": "/repo/x", "recursive": True})
    call = parse_record(record)[0]
    assert call.operation.kind == "unsupported"
    assert call.original_call["function"]["arguments"]
    record = make_record("SWE-agent", "bash", {"command": "pwd"})
    messages = json.loads(record["messages_json"])
    messages[1]["tool_calls"] *= 2
    record["messages_json"] = json.dumps(messages)
    assert all(c.operation.contracts["reason"] == "NONUNIQUE_CALL_ID" for c in parse_record(record))


def test_source_frontend_has_no_codegen_or_old_pipeline_dependencies():
    import trace_bridge
    package = Path(trace_bridge.__file__).parent
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(v.startswith(("trajectory_rewrite", "trajport", "agent_data_protocol")) for v in imported)
    tree = ast.parse((package / "sources.py").read_text())
    assert not any(isinstance(n, ast.Import) and any(a.name in {"shlex", "subprocess"} for a in n.names) for n in ast.walk(tree))
