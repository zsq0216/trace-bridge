"""Small operation fixtures, not dataset trajectories or harness rollouts."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trace_bridge.ir import CompileContext, GlobPolicy, Operation
from trace_bridge.patches import parse_patch
from trace_bridge.targets import TARGETS, compile_operation, replace_all_command


def bash(command, cwd, **env):
    return subprocess.run(["bash", "--noprofile", "--norc", "-c", command], cwd=cwd,
                          env={**os.environ, **env}, capture_output=True, timeout=10)


@pytest.mark.parametrize("target", TARGETS)
def test_replace_all_preserves_inode_hardlinks_mode_and_symlink(tmp_path, target):
    file = tmp_path / "data ' ;.txt"
    file.write_bytes(b"old\nold\n")
    file.chmod(0o640)
    alias = tmp_path / "hardlink"
    os.link(file, alias)
    link = tmp_path / "symlink"
    link.symlink_to(file)
    staging = tmp_path / "staging"
    staging.mkdir()
    before = file.stat()
    op = Operation("replace_text", {"path": str(link), "old": "old", "new": "new\nline", "mode": "all"})
    plan = compile_operation(op, target)
    assert plan.status == "ready", plan.to_dict()
    result = bash(plan.actions[0].arguments["command"], tmp_path, TMPDIR=str(staging))
    assert result.returncode == 0, result.stderr
    assert file.read_bytes() == alias.read_bytes() == b"new\nline\nnew\nline\n"
    assert file.stat().st_ino == alias.stat().st_ino == before.st_ino
    assert file.stat().st_mode == before.st_mode and file.stat().st_uid == before.st_uid
    assert link.is_symlink() and not list(staging.iterdir())


def test_replace_transform_failure_does_not_touch_original_and_cleans_temp(tmp_path):
    file = tmp_path / "data"
    file.write_text("old old")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sed = bindir / "sed"
    sed.write_text('#!/bin/bash\nif [ "$2" = "-n" ]; then exec /usr/bin/sed "$@"; fi\nprintf partial\nexit 2\n')
    sed.chmod(0o755)
    staging = tmp_path / "staging"
    staging.mkdir()
    result = bash(replace_all_command(str(file), "old", "new"), tmp_path,
                  PATH=str(bindir) + ":" + os.environ["PATH"], TMPDIR=str(staging))
    assert result.returncode != 0
    assert file.read_text() == "old old" and not list(staging.iterdir())


@pytest.mark.parametrize("target", TARGETS)
def test_replace_all_single_match_prefers_proven_native_call(target):
    op = Operation("replace_text", {"path": "/repo/a", "old": "old", "new": "new", "mode": "all"})
    plan = compile_operation(op, target, CompileContext(files={"/repo/a": "prefix old suffix\n"}))
    assert plan.status == "ready" and plan.actions[0].implementation == "native"
    assert plan.actions[0].arguments["old_str"] == "old"
    assert "literal_replace_on_full_preimage" in plan.verified_properties
    assert plan.preimage_sha256


def test_swe_unique_replace_avoids_tab_expansion(tmp_path):
    file = tmp_path / "tab.txt"
    before = "\told\nunchanged\ttext\n"
    file.write_text(before)
    op = Operation("replace_text", {"path": str(file), "old": "old", "new": "new", "mode": "unique"})
    plan = compile_operation(op, "sweagent", CompileContext(files={str(file): before}))
    assert plan.status == "ready" and plan.actions[0].implementation == "simple_shell"
    assert bash(plan.actions[0].arguments["command"], tmp_path).returncode == 0
    assert file.read_bytes() == b"\tnew\nunchanged\ttext\n"


@pytest.mark.parametrize("target", TARGETS)
def test_non_python_patch_guard_is_not_a_blanket_blocker(target):
    files = parse_patch("*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new\n*** End Patch")
    op = Operation("apply_patch", {"files": files}, {"syntax_check_python": True, "rollback_on_syntax_error": True})
    plan = compile_operation(op, target, CompileContext(cwd="/repo", files={"/repo/a.txt": "old\n"}))
    assert plan.status == "ready" and plan.actions[0].implementation == "native"
    assert "patch_line_and_byte_effect_on_full_preimage" in plan.verified_properties


@pytest.mark.parametrize("invalid", ["new = (", "return 1"])
def test_python_patch_static_failure_clears_all_actions(invalid):
    files = parse_patch("*** Begin Patch\n*** Add File: first.txt\n+hello\n*** Update File: a.py\n@@\n-old = 1\n+" + invalid + "\n*** End Patch")
    ctx = CompileContext(cwd="/repo", files={"/repo/a.py": "old = 1\n", "/repo/first.txt": None},
                         directories={"/repo"}, python_feature_version=(3, sys.version_info.minor))
    plan = compile_operation(Operation("apply_patch", {"files": files}, {"syntax_check_python": True}), "sweagent", ctx)
    assert plan.status == "unsupported" and not plan.actions


@pytest.mark.parametrize("target", TARGETS)
def test_python_patch_can_be_checked_for_a_declared_version_and_preimage(target):
    files = parse_patch("*** Begin Patch\n*** Update File: a.py\n@@\n-old = 1\n+new = 2\n*** End Patch")
    ctx = CompileContext(cwd="/repo", files={"/repo/a.py": "old = 1\n"}, python_feature_version=(3, sys.version_info.minor))
    plan = compile_operation(Operation("apply_patch", {"files": files}, {"syntax_check_python": True, "rollback_on_syntax_error": True}), target, ctx)
    assert plan.status == "ready"
    assert "python_syntax_on_full_postimage" in plan.verified_properties
    assert plan.preimage_sha256
    ctx.python_feature_version = None
    assert "PATCH_SYNTAX_CHECK_AND_ROLLBACK" in compile_operation(Operation("apply_patch", {"files": files}, {"syntax_check_python": True}), target, ctx).requirements


def test_patch_with_tabs_compiles_to_literal_write(tmp_path):
    file = tmp_path / "tab.txt"
    before = "\told\nlast\n"
    file.write_text(before)
    patch = "*** Begin Patch\n*** Update File: tab.txt\n@@\n-\told\n+\tnew\n*** End Patch"
    ctx = CompileContext(cwd=str(tmp_path), files={str(file): before})
    plan = compile_operation(Operation("apply_patch", {"files": parse_patch(patch)}), "sweagent", ctx)
    assert plan.status == "ready" and plan.actions[0].implementation == "simple_shell"
    assert bash(plan.actions[0].arguments["command"], tmp_path).returncode == 0
    assert file.read_bytes() == b"\tnew\nlast\n"


def search_fixture(tmp_path):
    names = ["root.py", "src/a.py", "src/b.txt", ".hidden.py", "ignored.py", ".gitignore"]
    for i, name in enumerate(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored.py\n" if name == ".gitignore" else name)
        os.utime(path, (1700000000 + i, 1700000000 + i))
    (tmp_path / ".git").mkdir()


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("hidden,ignore,expected", [
    (False, True, ["src/a.py", "root.py"]),
    (True, True, [".hidden.py", "src/a.py", "root.py"]),
    (False, False, ["ignored.py", "src/a.py", "root.py"]),
    (True, False, ["ignored.py", ".hidden.py", "src/a.py", "root.py"]),
])
def test_glob_explicit_policy_matches_fixture_selection_and_mtime(tmp_path, target, hidden, ignore, expected):
    search_fixture(tmp_path)
    op = Operation("find_files", {"root": str(tmp_path), "pattern": "**/*.py"}, {"order": "mtime"})
    ctx = CompileContext(glob_policy=GlobPolicy(hidden=hidden, respect_ignore=ignore))
    plan = compile_operation(op, target, ctx)
    assert plan.status == "ready"
    result = bash(plan.actions[0].arguments["command"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().splitlines() == [str(tmp_path / name) for name in expected]


@pytest.mark.parametrize("target", TARGETS)
def test_glob_empty_result_and_symlinks(tmp_path, target):
    (tmp_path / "actual.txt").write_text("x")
    (tmp_path / "linked.py").symlink_to(tmp_path / "actual.txt")
    op = Operation("find_files", {"root": str(tmp_path), "pattern": "*.py"})
    no_follow = compile_operation(op, target, CompileContext(glob_policy=GlobPolicy(follow_symlinks=False)))
    result = bash(no_follow.actions[0].arguments["command"], tmp_path)
    assert result.returncode == 0 and result.stdout == b""
    follow = compile_operation(op, target, CompileContext(glob_policy=GlobPolicy(follow_symlinks=True)))
    result = bash(follow.actions[0].arguments["command"], tmp_path)
    assert result.returncode == 0 and result.stdout.decode().strip() == str(tmp_path / "linked.py")


def test_glob_policy_does_not_override_explicit_source_contract():
    op = Operation("find_files", {"root": "/repo", "pattern": "*.py"}, {"hidden_files": "include"})
    plan = compile_operation(op, "sweagent", CompileContext(glob_policy=GlobPolicy(hidden=False)))
    assert plan.status == "unsupported"


def test_glob_braces_question_mark_and_literal_newline_path(tmp_path):
    root = tmp_path / "quoted ' root\nline"
    root.mkdir()
    for name in ("a.py", "a.ts", "aa.py", "a.txt"):
        (root / name).write_text(name)
    op = Operation("find_files", {"root": str(root), "pattern": "?.{py,ts}"})
    ctx = CompileContext(glob_policy=GlobPolicy(order="path"))
    plan = compile_operation(op, "sweagent", ctx)
    assert plan.status == "ready"
    result = bash(plan.actions[0].arguments["command"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == (str(root / "a.py") + "\n" + str(root / "a.ts") + "\n").encode()


def test_glob_traversal_error_is_not_treated_as_empty_result(tmp_path):
    op = Operation("find_files", {"root": str(tmp_path / "missing"), "pattern": "**/*.py"})
    plan = compile_operation(op, "sweagent", CompileContext(glob_policy=GlobPolicy()))
    result = bash(plan.actions[0].arguments["command"], tmp_path)
    assert result.returncode != 0


def test_glob_respects_declared_empty_error_policy(tmp_path):
    op = Operation("find_files", {"root": str(tmp_path), "pattern": "*.py"})
    plan = compile_operation(op, "sweagent", CompileContext(glob_policy=GlobPolicy(empty_matches="error")))
    result = bash(plan.actions[0].arguments["command"], tmp_path)
    assert result.returncode != 0 and result.stdout == b""


@pytest.mark.parametrize("target", TARGETS)
def test_delete_symlink_and_nonregular_failure_equivalence(tmp_path, target):
    file = tmp_path / "data"
    file.write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(file)
    op = Operation("delete_file", {"path": str(link)}, {"kind": "regular_file", "missing": "error", "recursive": False})
    plan = compile_operation(op, target)
    assert bash(plan.actions[0].arguments["command"], tmp_path).returncode == 0
    assert file.read_text() == "keep" and not link.is_symlink()
    link.symlink_to(tmp_path / "absent")
    assert bash(plan.actions[0].arguments["command"], tmp_path).returncode != 0
    assert link.is_symlink()


def test_plan_metadata_does_not_require_trajectory_replay():
    plan = compile_operation(Operation("delete_file", {"path": "/repo/file"}), "sweagent")
    data = json.loads(json.dumps(plan.to_dict()))
    assert data["validation_scope"] == "static_compilation"
    assert "target_replay_required" not in data
    assert data["training_ready"] is False
