import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trace_bridge.export_training import finalize, jsonlines, prepare
from trace_bridge.ir import CompileContext
from trace_bridge.rewrite_queue import render_result, validate_result
from trace_bridge.run import run
from trace_bridge.sources import dumps, parse_record, sha
from trace_bridge.targets import compile_operation
from trace_bridge.training import assemble, bind_results, profiles, stage_record, validate_dialogue
from trace_bridge.validate_training import audit


def call(cid="c1", name="bash", arguments=None):
    return {"id": cid, "type": "function", "function": {
        "name": name, "arguments": dumps(arguments or {"command": "printf 'hello'"})}}


def assistant(*calls, text=""):
    return {"role": "assistant", "content": text, "tool_calls": list(calls)}


def result(cid="c1", text="hello"):
    return {"role": "tool", "content": text, "tool_call_id": cid}


def record(messages, masks=None, rid="row1"):
    messages = [{"role": "system", "content": "Historical source tool instructions."},
                {"role": "user", "content": "<uploaded_files>\n/repo\n</uploaded_files>\nFix the bug."}] + messages
    tools = profiles()["sweagent"]["tools"] + [{"type": "function", "function": {
        "name": "task", "description": "Delegate work", "parameters": {"type": "object"}}}]
    return {"id": rid, "source_scaffold": "SWE-agent", "split": "train", "task_group_key": "task/" + rid,
            "messages_json": dumps(messages), "tools_json": dumps(tools),
            "message_loss_mask": [0, 0] + (masks if masks is not None else [int(m["role"] == "assistant") for m in messages[2:]]),
            "reference_tokens": 123}


def compiled(record, target="sweagent"):
    calls = parse_record(record)
    ir = {"id": record["id"], "split": record["split"], "source_row_index": 0,
          "source_payload_sha256": sha({"messages": json.loads(record["messages_json"]), "tools": json.loads(record["tools_json"])}),
          "operations": [c.to_dict() for c in calls]}
    plans = [{"call_id": c.call_id, "source_message_index": c.message_index,
              "source_call_index": c.call_index, "operation": c.operation.kind,
              **compile_operation(c.operation, target, CompileContext(cwd="/repo")).to_dict()}
             for c in calls]
    return ir, {"id": record["id"], "target": target, "plans": plans}


def staged(record, target="sweagent"):
    ir, plan = compiled(record, target)
    return stage_record(record, ir, plan, profiles()[target])


def rewrite_fixture(request):
    source = request["payload"]["source_messages"][-1]
    return {"request_id": request["request_id"], "input_sha256": request["input_sha256"],
            "rewrite": {"context": "The historical record contains the following operation and feedback.",
                        "evidence": [{"message_index": source["message_index"], "quote": source["evidence_text"]}],
                        "uncertainties": []}}


@pytest.mark.parametrize("target", ["sweagent", "openhands_sdk"])
def test_ready_roundtrip_failure_feedback_and_original_mask(target):
    r = record([assistant(call(arguments={"command": "printf '%s' '\"\\\n中文'"})), result(text="exit 1\nfailed: missing file")], masks=[0, 0])
    r["split"] = "validation"
    item, requests = staged(r, target)
    assert not requests
    row, status = assemble(item, profiles()[target], {}, {})
    assert row is None and status["status"] == "no_supervised_messages"
    r["message_loss_mask"][2] = 1
    item, requests = staged(r, target)
    row, status = assemble(item, profiles()[target], {}, {})
    m = json.loads(row["messages_json"])
    assert m[-1]["content"] == "exit 1\nfailed: missing file"
    assert row["message_loss_mask"][-2:] == [1, 0]
    assert json.loads(m[-2]["tool_calls"][0]["function"]["arguments"])["command"] == "printf '%s' '\"\\\n中文'"
    assert row["source_reference_tokens"] == 123 and "reference_tokens" not in row
    assert row["split"] == "train" and row["source_split"] == "validation"
    assert not row["has_target_finish"]


def test_unsupported_is_preserved_and_rewrite_always_masked():
    r = record([assistant(call(name="task", arguments={"prompt": "modify f.py"})),
                result(text="Recorded task error: nothing changed"),
                assistant(call("c2")), result("c2")])
    item, requests = staged(r)
    assert len(requests) == 1
    assert requests[0]["payload"]["source_messages"][1]["message"]["content"] == "Recorded task error: nothing changed"
    assert requests[0]["payload"]["source_calls"][0]["original_call"]["function"]["name"] == "task"
    row, status = assemble(item, profiles()["sweagent"], {}, {})
    assert row is None and status["status"] == "pending_rewrites"
    request = requests[0]
    row, _ = assemble(item, profiles()["sweagent"], {request["request_id"]: request},
                      {request["request_id"]: rewrite_fixture(request)})
    messages = json.loads(row["messages_json"])
    bridge = next((i, m) for i, m in enumerate(messages) if "Original source evidence" in m["content"])
    assert bridge[1]["role"] == "user" and row["message_loss_mask"][bridge[0]] == 0
    assert "Recorded task error: nothing changed" in bridge[1]["content"]
    assert [c["function"]["name"] for m in messages for c in m.get("tool_calls", [])] == ["bash"]


def test_conditional_and_mixed_batch_do_not_leak_candidate_calls():
    r = record([assistant(call(), call("c2", "task", {"prompt": "inspect"})), result(), result("c2")])
    item, requests = staged(r)
    assert len(requests) == 1 and len(requests[0]["payload"]["source_calls"]) == 2
    assert not any(s["kind"] == "messages" and any(m.get("tool_calls") for m in s["messages"]) for s in item["segments"])
    r = record([assistant(call()), result()])
    ir, plans = compiled(r)
    plans["plans"][0]["status"] = "conditional"
    plans["plans"][0]["requirements"] = ["SOURCE_TIMEOUT_LIFECYCLE_EQUIVALENCE"]
    item, requests = stage_record(r, ir, plans, profiles()["sweagent"])
    assert requests[0]["payload"]["reasons"] == ["SOURCE_TIMEOUT_LIFECYCLE_EQUIVALENCE"]
    assert requests[0]["payload"]["target_candidates"][0]["actions"]


def test_one_to_many_does_not_duplicate_source_observation():
    r = record([assistant(call()), result(text="one recorded response")])
    ir, plans = compiled(r)
    plans["plans"][0]["actions"] *= 2
    item, requests = stage_record(r, ir, plans, profiles()["sweagent"])
    assert "ONE_SOURCE_CALL_MULTIPLE_TARGET_ACTIONS" in requests[0]["payload"]["reasons"]
    assert len([m for m in requests[0]["payload"]["source_messages"] if m["message"]["role"] == "tool"]) == 1


def test_unique_binding_can_use_global_constraints_but_not_position():
    calls = [call("a"), call("b")]
    results = [(3, {"role": "tool", "content": "x", "possible_tool_call_ids": ["a", "b"]}),
               (4, {"role": "tool", "content": "y", "tool_call_id": "a"})]
    assert bind_results(calls, results) == {"a": 4, "b": 3}
    results[1][1].pop("tool_call_id")
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        bind_results(calls, results)
    results[1][1]["tool_call_id"] = "stale"
    with pytest.raises(ValueError, match="NO_MATCH"):
        bind_results(calls, results)


def test_unbound_singleton_recovery_and_missing_result_queue():
    r = record([assistant(call()), {"role": "tool", "content": "hello"}])
    item, requests = staged(r)
    assert not requests and item["stats"]["recovered_result_bindings"] == 1
    r = record([assistant(call())])
    item, requests = staged(r)
    assert requests[0]["payload"]["reasons"] == ["SOURCE_RESULT_COUNT_MISMATCH"]


@pytest.mark.parametrize("target", ["sweagent", "openhands_sdk"])
def test_plain_final_text_becomes_native_finish_and_intermediate_text_is_context(target):
    r = record([{"role": "assistant", "content": "Let me inspect."}, assistant(call()), result(),
                {"role": "assistant", "content": "Done."}])
    item, requests = staged(r, target)
    row, _ = assemble(item, profiles()[target], {}, {})
    messages = json.loads(row["messages_json"])
    assert not requests
    assert messages[3]["role"] == "user" and row["message_loss_mask"][3] == 0
    assert messages[-1]["tool_calls"][0]["function"]["name"] == ("finish" if target == "openhands_sdk" else "submit")
    assert row["has_target_finish"]


@pytest.mark.parametrize("mutation,error", [
    (lambda x: x.update(input_sha256="wrong"), "INPUT_MISMATCH"),
    (lambda x: x["rewrite"].update(context="<tool_call>bad</tool_call>"), "PROTOCOL_TAG"),
    (lambda x: x["rewrite"]["evidence"][0].update(message_index=999), "OUTSIDE_SEGMENT"),
    (lambda x: x["rewrite"]["evidence"][0].update(quote="invented success not present"), "NOT_VERBATIM"),
])
def test_rewrite_validation_rejects_cross_record_future_or_fabricated_evidence(mutation, error):
    _, requests = staged(record([assistant(call(name="task")), result(text="failed")]))
    response = rewrite_fixture(requests[0])
    mutation(response)
    with pytest.raises(ValueError, match=error):
        validate_result(requests[0], response)


def test_protocol_validator_rejects_early_finish_and_bad_masks():
    tools = profiles()["sweagent"]["tools"]
    m = [assistant(call(name="submit", arguments={"command": "bad"}))]
    with pytest.raises(ValueError, match="EXTRA_TARGET_ARGUMENT"):
        validate_dialogue(m, [1], tools, "sweagent")
    m[0]["tool_calls"][0]["function"]["arguments"] = "{}"
    m.append({"role": "user", "content": "continue"})
    with pytest.raises(ValueError, match="NONTERMINAL"):
        validate_dialogue(m, [1, 0], tools, "sweagent")
    with pytest.raises(ValueError, match="SUPERVISION_ROLE"):
        validate_dialogue([{"role": "user", "content": "context"}], [1], tools, "sweagent")


def test_complete_prepare_finalize_pipeline_assigns_all_train_and_preserves_source_split(tmp_path):
    good = record([assistant(call()), result()], rid="good")
    pending = record([assistant(call(name="task")), result(text="source task failed"),
                      assistant(call("c2")), result("c2")], rid="pending")
    pending["split"] = "validation"
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([good, pending]), source)
    compiled_dir = tmp_path / "compiled"
    run(source, compiled_dir, ["sweagent", "openhands_sdk"])
    stage = tmp_path / "stage"
    m = prepare(source, compiled_dir, stage)
    for target in m["targets"]:
        assert m["counts"][target]["source_rows"] == 2
        assert m["counts"][target]["rows/exported"] == 1
        assert pq.read_table(stage / (target + ".ready.parquet"))["id"].to_pylist() == ["good"]
    assert len(list(jsonlines(stage / "source.records.jsonl.gz"))) == 2
    assert audit(stage)["passed"]
    for target in m["targets"]:
        items = list(jsonlines(stage / (target + ".staged.jsonl.gz")))
        assert [r["metadata"]["split"] for r in items] == ["train", "train"]
        assert [r["metadata"]["source_split"] for r in items] == ["train", "validation"]
        requests = list(jsonlines(stage / (target + ".rewrite_requests.jsonl.gz")))
        assert all(r["payload"]["split"] == "train" and r["payload"]["source_split"] == "validation" for r in requests)
    responses = tmp_path / "results.jsonl"
    responses.write_text("".join(dumps(rewrite_fixture(r)) + "\n" for t in m["targets"]
                                for r in jsonlines(stage / (t + ".rewrite_requests.jsonl.gz"))))
    final = tmp_path / "final"
    fm = finalize(stage, [responses], final)
    for target in m["targets"]:
        rows = pq.read_table(final / (target + ".parquet")).to_pylist()
        assert [r["id"] for r in rows] == ["good", "pending"]
        assert [r["split"] for r in rows] == ["train", "train"]
        assert [r["source_split"] for r in rows] == ["train", "validation"]
        assert fm["counts"][target]["masked_rewrite_messages"] == 1
        assert all(r["training_ready"] for r in rows)
    assert list(jsonlines(final / "pending.rows.jsonl.gz")) == []


def test_incomplete_results_remain_pending_and_conflicting_results_fail(tmp_path):
    r = record([assistant(call(name="task")), result(), assistant(call("c2")), result("c2")])
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([r]), source)
    run(source, tmp_path / "compiled", ["sweagent"])
    prepare(source, tmp_path / "compiled", tmp_path / "stage", ["sweagent"])
    m = finalize(tmp_path / "stage", [], tmp_path / "pending")
    assert m["counts"]["sweagent"]["rows/pending_rewrites"] == 1
    assert pq.read_table(tmp_path / "pending/sweagent.parquet").num_rows == 0
    request = next(jsonlines(tmp_path / "stage/sweagent.rewrite_requests.jsonl.gz"))
    response = rewrite_fixture(request)
    data = dumps(response) + "\n"
    response["rewrite"]["context"] = "different response"
    results = tmp_path / "conflicts.jsonl"
    results.write_text(data + dumps(response) + "\n")
    with pytest.raises(ValueError, match="CONFLICTING_REWRITE_RESULTS"):
        finalize(tmp_path / "stage", [results], tmp_path / "conflicts")
    assert not (tmp_path / "conflicts/manifest.json").exists()


def test_tampered_ir_or_plan_is_rejected():
    r = record([assistant(call()), result()])
    ir, plans = compiled(r)
    ir["source_payload_sha256"] = "bad"
    with pytest.raises(ValueError, match="PAYLOAD_CHANGED"):
        stage_record(r, ir, plans, profiles()["sweagent"])
