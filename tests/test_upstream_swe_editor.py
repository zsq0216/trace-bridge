"""Optional local integration with the actual pinned SWE editor executable."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trace_bridge.ir import CompileContext, Operation
from trace_bridge.patches import parse_patch
from trace_bridge.targets import compile_operation

UPSTREAM = Path(os.environ.get("SWE_AGENT_TOOLS", "vendor/swe-agent/tools"))
EDITOR = UPSTREAM / "edit_anthropic/bin/str_replace_editor"
pytestmark = pytest.mark.skipif(not EDITOR.is_file(), reason="local SWE-agent editor checkout not installed")


def invoke(action, tmp_path):
    a = action.arguments
    env = dict(os.environ, PYTHONPATH=str(UPSTREAM / "registry/lib"), SWE_AGENT_ENV_FILE=str(tmp_path / "registry.json"))
    args = [sys.executable, str(EDITOR), a["command"], a["path"]]
    args += ["--" + name + "=" + str(a[name]) for name in ("old_str", "new_str", "file_text", "insert_line") if name in a]
    return subprocess.run(args,
                          cwd=tmp_path, env=env, capture_output=True, timeout=15)


def test_native_patch_candidate_runs_in_actual_swe_editor(tmp_path):
    path = tmp_path / "example.py"
    path.write_text("value = 1\n")
    patch = "*** Begin Patch\n*** Update File: example.py\n@@\n-value = 1\n+value = 2\n*** End Patch"
    op = Operation("apply_patch", {"files": parse_patch(patch)})
    plan = compile_operation(op, "sweagent", CompileContext(cwd=str(tmp_path), files={str(path): path.read_text()}))
    result = invoke(plan.actions[0], tmp_path)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert path.read_text() == "value = 2\n"


def test_actual_swe_tab_expansion_justifies_normalization_requirement(tmp_path):
    path = tmp_path / "example.txt"
    path.write_text("\told\n")
    op = Operation("replace_text", {"path": str(path), "old": "old", "new": "new", "mode": "unique"})
    plan = compile_operation(op, "sweagent")
    assert "NATIVE_EDITOR_TEXT_NORMALIZATION_EQUIVALENCE" in plan.requirements
    result = invoke(plan.actions[0], tmp_path)
    assert result.returncode == 0, result.stderr
    assert path.read_text() == "        new\n"


def test_ready_multifile_patch_matches_expected_bytes_with_native_primitives(tmp_path):
    first, second, added = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "new.txt"
    first.write_text("start\nold\nend\n")
    second.write_text("another\n")
    patch = ("*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new\n"
             "*** Update File: b.txt\n@@\n-another\n+changed\n"
             "*** Add File: new.txt\n+中文\n*** End Patch")
    ctx = CompileContext(cwd=str(tmp_path), files={str(first): first.read_text(), str(second): second.read_text(), str(added): None}, directories={str(tmp_path)})
    plan = compile_operation(Operation("apply_patch", {"files": parse_patch(patch)}), "sweagent", ctx)
    assert plan.status == "ready"
    for action in plan.actions:
        result = invoke(action, tmp_path)
        assert result.returncode == 0, (result.stdout, result.stderr)
    assert first.read_bytes() == b"start\nnew\nend\n"
    assert second.read_bytes() == b"changed\n"
    assert added.read_bytes() == "中文\n".encode()
