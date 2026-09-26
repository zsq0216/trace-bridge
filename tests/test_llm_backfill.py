import asyncio
import gzip
import json
import subprocess
import sys

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trace_bridge import llm_backfill as runner
from trace_bridge.export_training import jsonlines, prepare
from trace_bridge.rewrite_queue import make_request
from trace_bridge.run import file_sha, run
from trace_bridge.sources import dumps, sha
from trace_bridge.resolution import evaluate_result
from trace_bridge.training import profiles


def tiny_stage(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    record = {"id": "sample", "source_scaffold": "SWE-agent", "split": "train"}
    request = make_request(record, "openhands_sdk", [0], ["unrecognized"], [], [],
                           [{"role": "tool", "content": "Operation failed."}])
    queue = stage / "openhands_sdk.rewrite_requests.jsonl.gz"
    with gzip.open(queue, "wt") as f:
        f.write(dumps(request) + "\n")
    (stage / "manifest.json").write_text(dumps({"outputs": {queue.name: file_sha(queue)}}))
    return stage, request


def test_http_validation_retry_checkpoint_resume_and_usage(tmp_path, monkeypatch):
    stage, request = tiny_stage(tmp_path)
    out = tmp_path / "run"
    db, config = runner.prepare_db(stage, "openhands_sdk", out, "http://unit.test/v1", "fixture-model")
    count = 0

    async def respond(http_request):
        nonlocal count
        count += 1
        assert http_request.headers["Authorization"] == "Bearer test-secret"
        payload = json.loads(http_request.content)
        assert "thinking" not in payload
        body = {"context": "The recorded operation failed.", "uncertainties": [],
                "evidence": [{"message_index": 0, "quote": "invented" if count == 1 else "Operation failed."}]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": dumps(body)}}],
                                       "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(runner.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw))

    async def immediate(*args):
        pass
    monkeypatch.setattr(runner.asyncio, "sleep", immediate)
    asyncio.run(runner.execute(db, config, profiles()["openhands_sdk"], "test-secret", out, 1, 0, 2, 10))
    assert count == 2
    assert db.execute("SELECT status FROM jobs").fetchone()[0] == "done"
    assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
    assert len(list(jsonlines(out / "results.jsonl.gz"))) == 1
    report = json.loads((out / "progress.json").read_text())
    assert report["usage"]["total_tokens"] == 30
    assert "test-secret" not in (out / "config.json").read_text()
    db.close()
    db, config = runner.prepare_db(stage, "openhands_sdk", out, "http://unit.test/v1", "fixture-model")
    asyncio.run(runner.execute(db, config, profiles()["openhands_sdk"], "test-secret", out, 1, 0, 2, 10))
    assert count == 2
    db.close()


def test_json_decoder_rejects_duplicate_keys_and_handles_fence():
    assert runner.model_json('```json\n{"a":1}\n```') == {"a": 1}
    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
        runner.model_json('{"a":1,"a":2}')


def test_blocked_candidate_uses_short_summary_prompt_but_keeps_full_source(tmp_path):
    _, request = tiny_stage(tmp_path)
    payload = request["payload"]
    payload.update(route="candidate_resolution", review={"promotion_eligible": False,
                   "blockers": ["SOURCE_RESULT_COUNT_MISMATCH"], "large_candidate": "IR_ONLY"})
    request["request_id"] = request["input_sha256"] = sha(payload)
    messages = runner.messages_for(request)
    view = json.loads(messages[1]["content"])
    assert view["source_segment"][0]["message"] == payload["source_messages"][0]["message"]
    assert "IR_ONLY" not in messages[1]["content"]
    body = {"context": "The recorded operation failed.", "evidence": [{"message_index": 0, "quote": "Operation failed."}], "uncertainties": []}
    result = runner.envelope(request, body, "llm_segment_context_v2")
    output, masks, decision = evaluate_result(request, result, profiles()["openhands_sdk"])
    assert masks == [0] and decision["decision"] == "context"
    assert "Original source evidence" in output[0]["content"]


def test_only_unobserved_known_submission_can_skip_llm():
    marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    message = {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {
        "name": "bash", "arguments": dumps({"command": "echo " + marker})}}]}
    call = {"call_id": "c", "source_tool": "bash", "message_index": 0,
            "operation": {"kind": "run_command", "parameters": {"command": "echo " + marker}}}
    record = {"id": "r", "source_scaffold": "mini-swe-agent", "split": "train", "tools_json": "[]"}
    review = {"promotion_eligible": False, "blockers": ["SOURCE_RESULT_COUNT_MISMATCH"]}
    request = make_request(record, "openhands_sdk", [0], ["SOURCE_RESULT_COUNT_MISMATCH"], [call], [], [message], review)
    result = runner.mechanical_result(request)
    assert result["producer"] == "mechanical_unobserved_submission_v1"
    _, masks, decision = evaluate_result(request, result, profiles()["openhands_sdk"])
    assert masks == [0] and decision["decision"] == "context"
    request["payload"]["source_messages"].append({"message_index": 1, "message": {"role": "tool", "content": "observed"}})
    assert runner.mechanical_result(request) is None


def test_finalize_cli_exports_only_requested_target(tmp_path):
    source = tmp_path / "source.parquet"
    messages = [{"role": "assistant", "content": "Done."}]
    record = {"id": "row", "source_scaffold": "SWE-agent", "split": "train", "task_group_key": "task",
              "messages_json": dumps(messages), "message_loss_mask": [1],
              "tools_json": dumps(profiles()["sweagent"]["tools"])}
    pq.write_table(pa.Table.from_pylist([record]), source)
    run(source, tmp_path / "compiled", ["sweagent", "openhands_sdk"])
    prepare(source, tmp_path / "compiled", tmp_path / "stage")
    subprocess.run([sys.executable, "-m", "trace_bridge.export_training", "finalize", "--stage", str(tmp_path / "stage"),
                    "--targets", "openhands_sdk", "--out", str(tmp_path / "final")], check=True, capture_output=True)
    assert (tmp_path / "final/openhands_sdk.parquet").exists()
    assert not (tmp_path / "final/sweagent.parquet").exists()
    assert json.loads((tmp_path / "final/manifest.json").read_text())["targets"] == ["openhands_sdk"]
    db, _ = runner.prepare_db(tmp_path / "stage", "openhands_sdk", tmp_path / "calls", "http://unit.test/v1", "fixture")
    db.close()
    from trace_bridge.validate_backfill import audit
    checked = audit(tmp_path / "stage", tmp_path / "calls", tmp_path / "final")
    assert checked["passed"] and checked["counts"]["rows"] == 1 and checked["processed_regions"] == 0
