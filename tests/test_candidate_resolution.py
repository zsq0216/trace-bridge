import copy
import json
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trace_bridge.export_training import export_prompts, finalize, jsonlines, prepare
from trace_bridge.ir import CompileContext
from trace_bridge.resolution import evaluate_result
from trace_bridge.rewrite_queue import result_field
from trace_bridge.run import bound_success, run
from trace_bridge.sources import dumps, parse_record, sha
from trace_bridge.targets import compile_operation
from trace_bridge.training import assemble, profiles, stage_record
from trace_bridge.validate_training import audit


def tool(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in properties},
                           "required": list(properties), "additionalProperties": False}}}


def invocation(cid, name, args, feedback):
    return [{"role": "assistant", "content": "Inspect the recorded change.", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": dumps(args)}}]},
        {"role": "tool", "tool_call_id": cid, "content": feedback}]


def source_record(source, tools, messages, rid="review"):
    messages = [{"role": "user", "content": "<uploaded_files>\n/repo\n</uploaded_files>"}] + messages
    return {"id": rid, "source_scaffold": source, "split": "validation", "task_group_key": rid,
            "messages_json": dumps(messages), "tools_json": dumps(tools),
            "message_loss_mask": [int(m["role"] == "assistant") for m in messages]}


def edit_record(between=()):
    tools = [tool("Write", "overwrite the existing file; Read tool first", ["file_path", "content"]),
             tool("Edit", "Read tool first. Replace a unique literal occurrence.", ["file_path", "old_string", "new_string"]),
             tool("Bash", "Execute a shell command", ["command"]),
             tool("Read", "Read file contents with line numbers", ["file_path"])]
    messages = invocation("write", "Write", {"file_path": "/repo/a.txt", "content": "old\n"},
                          "File created successfully at: /repo/a.txt")
    messages += list(between)
    messages += invocation("edit", "Edit", {"file_path": "/repo/a.txt", "old_string": "old", "new_string": "new"},
                           "The file /repo/a.txt has been updated")
    messages += [{"role": "assistant", "content": "Done."}]
    return source_record("Claude Code", tools, messages)


def patch_record(path="/repo/a.py"):
    tools = [tool("apply_patch", "Python files are syntax-checked after modification; rolled back on failure.", ["patch"])]
    add = "*** Begin Patch\n*** Add File: " + path + "\n+alpha = 1\n+middle = 0\n+omega = 2\n*** End Patch"
    update = ("*** Begin Patch\n*** Update File: " + path
              + "\n@@\n-alpha = 1\n+alpha = 3\n@@\n-omega = 2\n+omega = 4\n*** End Patch")
    messages = invocation("add", "apply_patch", {"patch": add}, "Execution output of [apply_patch]:\nAdded file: " + path)
    messages += invocation("update", "apply_patch", {"patch": update}, "Execution output of [apply_patch]:\nUpdated file: " + path)
    messages += [{"role": "assistant", "content": "Done."}]
    return source_record("Codex-format", tools, messages)


def stage(record, target="sweagent"):
    calls = parse_record(record)
    messages = json.loads(record["messages_json"])
    ir = {"id": record["id"], "split": record["split"], "source_row_index": 0,
          "source_payload_sha256": sha({"messages": messages, "tools": json.loads(record["tools_json"])}),
          "operations": [c.to_dict() for c in calls]}
    plans = [{"call_id": c.call_id, "source_message_index": c.message_index,
              "source_call_index": c.call_index, "operation": c.operation.kind,
              **compile_operation(c.operation, target, CompileContext(cwd="/repo", source_succeeded=bound_success(c, messages))).to_dict()}
             for c in calls]
    return stage_record(record, ir, {"id": record["id"], "target": target, "plans": plans}, profiles()[target])


def response(request, promote=False):
    event = request["payload"]["source_messages"][-1]
    fallback = {"context": "The source record describes a file operation and its recorded feedback.",
                "evidence": [{"message_index": event["message_index"], "quote": event["evidence_text"]}],
                "uncertainties": []}
    result = {"request_id": request["request_id"], "input_sha256": request["input_sha256"]}
    if result_field(request) == "rewrite":
        result["rewrite"] = fallback
    else:
        proposals = []
        if promote:
            for entry in request["payload"]["review"]["entries"]:
                actions = entry["recompiled_plan"]["actions"]
                action = actions[0] if actions else {"tool": "bash", "arguments": {"command": "true"}}
                proposals.append({"call_id": entry["call_id"],
                                  "action": {"tool": action["tool"], "arguments": copy.deepcopy(action["arguments"])},
                                  "evidence_ids": [e["evidence_id"] for e in entry["evidence"]]})
        result["resolution"] = {"decision": "promote" if promote else "context",
                                "promotions": proposals, "fallback": fallback}
    return result


@pytest.mark.parametrize("target", ["sweagent", "openhands_sdk"])
@pytest.mark.parametrize("mask", [0, 1])
def test_verified_edit_promotes_and_inherits_mask(target, mask):
    record = edit_record()
    record["message_loss_mask"][3] = mask
    item, requests = stage(record, target)
    assert len(requests) == 1
    request = requests[0]
    assert request["payload"]["review"]["promotion_eligible"]
    result = response(request, True)
    row, _ = assemble(item, profiles()[target], {request["request_id"]: request}, {request["request_id"]: result})
    assert row["split"] == "train" and row["source_split"] == "validation"
    assert row["promoted_target_calls"] == 1 and row["masked_rewrite_count"] == 0
    messages = json.loads(row["messages_json"])
    index = next(i for i, m in enumerate(messages) if m.get("tool_calls") and '"str_replace"' in m["tool_calls"][0]["function"]["arguments"])
    assert row["message_loss_mask"][index:index + 2] == [mask, 0]
    assert messages[index + 1]["content"] == "The file /repo/a.txt has been updated"
    provenance = json.loads(row["bridge_provenance_json"])
    assert any(p["origin"] == "verified_candidate_promotion" for p in provenance)


@pytest.mark.parametrize("between", [
    invocation("opaque", "Bash", {"command": "printf 'unknown effects'"}, "done"),
    invocation("delegate", "unknown_tool", {}, "done"),
    [{"role": "user", "content": "The checkout may have changed."}],
])
def test_stale_snapshots_cannot_be_confirmed_by_model(between):
    _, requests = stage(edit_record(between))
    request = requests[-1]
    assert not request["payload"]["review"]["promotion_eligible"]
    messages, masks, decision = evaluate_result(request, response(request, True), profiles()["sweagent"])
    assert decision["decision"] == "promotion_rejected_context"
    assert len(messages) == 1 and messages[0]["role"] == "user" and masks == [0]
    assert "Original source evidence" in messages[0]["content"]


def test_numbered_read_is_not_a_full_byte_snapshot():
    record = edit_record()
    messages = json.loads(record["messages_json"])
    messages[1:3] = invocation("read", "Read", {"file_path": "/repo/a.txt"}, "     1→old\n...(truncated)")
    record["messages_json"] = dumps(messages)
    _, requests = stage(record)
    entry = requests[0]["payload"]["review"]["entries"][0]
    assert entry["context"]["files"] == {} and entry["context"]["read_paths"] == ["/repo/a.txt"]
    assert "NATIVE_EDITOR_TEXT_NORMALIZATION_EQUIVALENCE" in entry["blockers"]


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: p["action"]["arguments"].update(new_str="invented"), "ACTION_NOT_VERIFIED"),
    (lambda p: p.update(evidence_ids=["invented confirmation"]), "EVIDENCE_MISMATCH"),
    (lambda p: p.update(call_id="wrong"), "SOURCE_CALL_COVERAGE"),
])
def test_arbitrary_repair_or_confirmation_falls_back(mutation, reason):
    _, requests = stage(edit_record())
    request = requests[0]
    result = response(request, True)
    mutation(result["resolution"]["promotions"][0])
    output, masks, decision = evaluate_result(request, result, profiles()["sweagent"])
    assert masks == [0] and all(not m.get("tool_calls") for m in output)
    assert any(reason in r for r in decision["rejection_reasons"])


@pytest.mark.parametrize("target", ["sweagent", "openhands_sdk"])
def test_patch_multiple_hunks_coalesces_with_one_response_and_byte_equivalence(tmp_path, target):
    path = str(tmp_path / "quoted ' file.py")
    item, requests = stage(patch_record(path), target)
    assert len(requests) == 2
    assert not requests[0]["payload"]["review"]["promotion_eligible"]
    request = requests[1]
    entry = request["payload"]["review"]["entries"][0]
    assert request["payload"]["review"]["promotion_eligible"]
    plan = entry["recompiled_plan"]
    assert "python_syntax_from_bound_source_patch_success" in plan["verified_properties"]
    assert "single_file_patch_coalesced_with_one_recorded_response" in plan["verified_properties"]
    command = plan["actions"][0]["arguments"]["command"]
    assert "python" not in command
    (tmp_path / "quoted ' file.py").write_text("alpha = 1\nmiddle = 0\nomega = 2\n")
    subprocess.run(["bash", "-c", command], check=True)
    assert (tmp_path / "quoted ' file.py").read_bytes() == b"alpha = 3\nmiddle = 0\nomega = 4\n"
    result_map = {r["request_id"]: response(r, r is request) for r in requests}
    row, _ = assemble(item, profiles()[target], {r["request_id"]: r for r in requests}, result_map)
    messages = json.loads(row["messages_json"])
    assert sum(m["role"] == "tool" for m in messages) == 1
    assert row["promoted_region_count"] == 1 and row["masked_rewrite_count"] == 1


def test_current_success_and_future_write_cannot_supply_preimage():
    r = edit_record()
    messages = json.loads(r["messages_json"])
    messages[1:5] = messages[3:5] + messages[1:3]
    r["messages_json"] = dumps(messages)
    _, requests = stage(r)
    assert not requests[0]["payload"]["review"]["promotion_eligible"]
    assert requests[0]["payload"]["review"]["entries"][0]["context"]["files"] == {}


def test_unknown_timeout_cannot_be_waived():
    r = edit_record()
    tools = json.loads(r["tools_json"])
    tools[2]["function"]["parameters"]["properties"]["timeout"] = {"type": "integer"}
    messages = json.loads(r["messages_json"])
    messages[1:5] = invocation("shell", "Bash", {"command": "true", "timeout": 5000}, "exit 0")
    r.update(tools_json=dumps(tools), messages_json=dumps(messages), message_loss_mask=[0, 1, 0, 1])
    _, requests = stage(r)
    output, masks, decision = evaluate_result(requests[0], response(requests[0], True), profiles()["sweagent"])
    assert "SOURCE_TIMEOUT_LIFECYCLE_EQUIVALENCE" in decision["rejection_reasons"]
    assert masks == [0]


def test_source_patch_failure_does_not_discharge_syntax_guard():
    r = patch_record()
    messages = json.loads(r["messages_json"])
    messages[4]["content"] = "Execution output of [apply_patch]:\nERROR: syntax check failed; rolled back"
    r["messages_json"] = dumps(messages)
    _, requests = stage(r)
    entry = requests[-1]["payload"]["review"]["entries"][0]
    assert not entry["context"]["source_patch_succeeded"]
    assert "PATCH_SYNTAX_CHECK_AND_ROLLBACK" in entry["blockers"]
    _, masks, decision = evaluate_result(requests[-1], response(requests[-1], True), profiles()["sweagent"])
    assert masks == [0] and decision["decision"] == "promotion_rejected_context"


def test_context_only_request_cannot_inject_a_promotion():
    r = source_record("Claude Code", [], invocation("unknown", "delegate", {}, "No recorded result"))
    _, requests = stage(r)
    request = requests[0]
    assert request["payload"]["route"] == "context_rewrite"
    result = response(request)
    result["resolution"] = {"decision": "promote", "promotions": [], "fallback": result["rewrite"]}
    with pytest.raises(ValueError, match="CONTEXT_ONLY_REQUEST_CANNOT_PROMOTE"):
        evaluate_result(request, result, profiles()["sweagent"])


def test_mixed_effect_batch_cannot_be_confirmed_by_position():
    r = edit_record()
    messages = json.loads(r["messages_json"])
    messages[1]["tool_calls"] += messages[3]["tool_calls"]
    del messages[3]
    r["messages_json"] = dumps(messages)
    r["message_loss_mask"] = [int(m["role"] == "assistant") for m in messages]
    _, requests = stage(r)
    assert "SOURCE_BATCH_EFFECT_ORDER" in requests[0]["payload"]["review"]["blockers"]
    _, masks, decision = evaluate_result(requests[0], response(requests[0], True), profiles()["sweagent"])
    assert masks == [0] and "SOURCE_BATCH_EFFECT_ORDER" in decision["rejection_reasons"]


def test_partial_candidate_response_is_audited_but_not_exported(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([patch_record()]), source)
    run(source, tmp_path / "compiled", ["sweagent"])
    prepare(source, tmp_path / "compiled", tmp_path / "stage", ["sweagent"])
    requests = list(jsonlines(tmp_path / "stage/sweagent.rewrite_requests.jsonl.gz"))
    results = tmp_path / "partial.jsonl"
    results.write_text(dumps(response(requests[-1], True)) + "\n")
    manifest = finalize(tmp_path / "stage", [results], tmp_path / "partial")
    counts = manifest["counts"]["sweagent"]
    assert counts["review_accepted_target_calls"] == 1 and counts["rows/pending_rewrites"] == 1
    assert not counts.get("exported_promoted_target_calls")
    assert pq.read_table(tmp_path / "partial/sweagent.parquet").num_rows == 0
    decisions = list(jsonlines(tmp_path / "partial/review.decisions.jsonl.gz"))
    assert len(decisions) == 1 and decisions[0]["decision"] == "promoted"


def test_end_to_end_review_routes_prompts_and_decision_audit(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([edit_record(), patch_record() | {"id": "patch"}]), source)
    run(source, tmp_path / "compiled", ["sweagent"])
    manifest = prepare(source, tmp_path / "compiled", tmp_path / "stage", ["sweagent"])
    assert manifest["counts"]["sweagent"]["review/promotion_eligible"] == 2
    assert audit(tmp_path / "stage")["passed"]
    queue = tmp_path / "stage/sweagent.rewrite_requests.jsonl.gz"
    export_prompts(queue, tmp_path / "prompts.jsonl")
    assert all(p["result_field"] == "resolution" for p in jsonlines(tmp_path / "prompts.jsonl"))
    requests = list(jsonlines(queue))
    results = tmp_path / "results.jsonl"
    results.write_text("".join(dumps(response(r, r["payload"]["review"]["promotion_eligible"])) + "\n" for r in requests))
    m = finalize(tmp_path / "stage", [results], tmp_path / "final")
    counts = m["counts"]["sweagent"]
    assert counts["exported_promoted_target_calls"] == 2 and counts["masked_rewrite_messages"] == 1
    assert len(list(jsonlines(tmp_path / "final/review.decisions.jsonl.gz"))) == 3
    rows = pq.read_table(tmp_path / "final/sweagent.parquet").to_pylist()
    assert len(rows) == 2 and all(r["split"] == "train" for r in rows)
