import copy
import json

import pytest

from trace_bridge.llm_backfill import envelope, pack, prepare_db
from trace_bridge.resolution import evaluate_result
from trace_bridge.reuse_context import Donor, PRODUCER
from trace_bridge.run import file_sha
from trace_bridge.sources import dumps, sha
from trace_bridge.training import profiles
from test_llm_backfill import tiny_stage


def donor_fixture(tmp_path):
    stage, request = tiny_stage(tmp_path)
    out = tmp_path / "donor"
    db, config = prepare_db(stage, "openhands_sdk", out, "http://unit.test/v1", "fixture")
    body = {"context": "The operation failed.", "uncertainties": [],
            "evidence": [{"message_index": 0, "quote": "Operation failed."}]}
    result = envelope(request, body, "llm_segment_context_v2")
    _, _, decision = evaluate_result(request, result, profiles()["openhands_sdk"])
    response = {"choices": [{"finish_reason": "stop", "message": {"content": dumps(body)}}]}
    db.execute("INSERT INTO attempts(job_id,driver_sha256,response) VALUES(?,?,?)",
               (request["request_id"], config["driver_sha256"], pack(response)))
    db.execute("UPDATE jobs SET status='done',result=?,decision=?", (pack(result), pack(decision)))
    db.commit()
    db.close()
    recipient = copy.deepcopy(request)
    recipient["payload"]["target"] = "sweagent"
    recipient["request_id"] = recipient["input_sha256"] = sha(recipient["payload"])
    return out, config, request, recipient


def test_identical_prompt_rebinds_hash_and_records_llm_origin(tmp_path):
    out, config, source, target = donor_fixture(tmp_path)
    donor = Donor(out, config)
    try:
        result = donor.adapt(target, source["request_id"])
        assert result["request_id"] == target["request_id"] != source["request_id"]
        assert result["producer"] == PRODUCER and result["reuse"]["new_http_call"] is False
        _, masks, decision = evaluate_result(target, result, profiles()["sweagent"])
        assert masks == [0] and decision["decision"] == "context"
        donor.verify(target, result)
        changed = copy.deepcopy(result)
        changed["rewrite"]["context"] = "The operation succeeded."
        with pytest.raises(ValueError, match="REUSE_PROVENANCE_OR_RESULT_MISMATCH"):
            donor.verify(target, changed)
    finally:
        donor.close()


def test_different_source_and_candidate_review_are_not_reused(tmp_path):
    out, config, source, target = donor_fixture(tmp_path)
    donor = Donor(out, config)
    try:
        changed = copy.deepcopy(target)
        changed["payload"]["source_messages"][0]["message"]["content"] = "Different feedback."
        with pytest.raises(ValueError, match="REUSE_PROMPT_MISMATCH"):
            donor.adapt(changed, source["request_id"])
        target["payload"]["review"] = {"promotion_eligible": True}
        with pytest.raises(ValueError, match="CANDIDATE_REVIEW_CANNOT_REUSE_CONTEXT"):
            donor.adapt(target, source["request_id"])
    finally:
        donor.close()


def test_retry_prompt_and_other_model_excluded(tmp_path):
    out, config, source, target = donor_fixture(tmp_path)
    import sqlite3
    db = sqlite3.connect(out / "journal.sqlite")
    db.execute("INSERT INTO attempts(job_id,error) VALUES(?,?)", (source["request_id"], "SCHEMA_ERROR"))
    db.commit()
    db.close()
    donor = Donor(out, config)
    try:
        assert donor.index() == {}
    finally:
        donor.close()
    with pytest.raises(ValueError, match="REUSE_SERVICE_OR_MODEL_CONFIGURATION_CHANGED"):
        Donor(out, {**config, "model": "different"})
