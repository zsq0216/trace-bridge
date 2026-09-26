"""Audit finalized data against source records and the LLM journal."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from itertools import chain, zip_longest
from pathlib import Path

import pyarrow.parquet as pq

from .export_training import jsonlines
from .llm_backfill import unpack
from .resolution import evaluate_result
from .reuse_context import Donor, PRODUCER as REUSED_CONTEXT_PRODUCER
from .run import file_sha
from .sources import sha
from .training import validate_dialogue


def audit(stage, calls, final):
    manifest = json.loads((final / "manifest.json").read_text())
    if manifest["phase"] != "finalize" or len(manifest["targets"]) != 1:
        raise ValueError("EXPECTED_SINGLE_TARGET_FINALIZE")
    for name, digest in manifest["outputs"].items():
        if file_sha(final / name) != digest:
            raise ValueError("FINAL_ARTIFACT_HASH_MISMATCH:" + name)
    target = manifest["targets"][0]
    profile = json.loads((stage / "profiles.json").read_text())[target]
    db = sqlite3.connect("file:" + str((calls / "journal.sqlite").resolve()) + "?mode=ro", uri=True)
    donors = {}
    call_config = json.loads((calls / "config.json").read_text())
    try:
        incomplete = db.execute("SELECT COUNT(*) FROM jobs WHERE status!='done'").fetchone()[0]
        if incomplete:
            raise ValueError("INCOMPLETE_CALL_JOURNAL:" + str(incomplete))
        if next(jsonlines(final / "pending.rows.jsonl.gz"), None) is not None:
            raise ValueError("FINAL_HAS_PENDING_ROWS")
        counts = Counter()
        producers = Counter()
        sources = Counter()
        decisions = Counter()
        seen_requests = set()
        exported = chain.from_iterable(b.to_pylist() for b in pq.ParquetFile(final / (target + ".parquet")).iter_batches(batch_size=8))
        for original, staged, row in zip_longest(jsonlines(stage / "source.records.jsonl.gz"),
                                                jsonlines(stage / (target + ".staged.jsonl.gz")), exported):
            if original is None or staged is None or row is None or original["id"] != staged["id"] or row["id"] != staged["id"]:
                raise ValueError("FINAL_SOURCE_ROW_COVERAGE")
            messages = json.loads(row["messages_json"])
            masks = row["message_loss_mask"]
            raw_messages = json.loads(original["messages_json"])
            if row["split"] != "train" or row["source_split"] != original["split"]:
                raise ValueError("FINAL_SPLIT_MISMATCH")
            if row["source_payload_sha256"] != staged["source_payload_sha256"]:
                raise ValueError("FINAL_SOURCE_HASH_MISMATCH")
            if json.loads(row["tools_json"]) != profile["tools"]:
                raise ValueError("FINAL_TOOL_DEFINITIONS_CHANGED")
            checks = validate_dialogue(messages, masks, profile["tools"], target)
            source_coverage, target_cursor, request_ids = [], 0, []
            provenance = json.loads(row["bridge_provenance_json"])
            for segment, origin in zip(staged["segments"], provenance, strict=True):
                start, end = origin["target_start"], origin["target_end"]
                if start != target_cursor or origin["source_indices"] != segment["source_indices"]:
                    raise ValueError("FINAL_PROVENANCE_COVERAGE")
                target_cursor = end
                source_coverage.extend(origin["source_indices"])
                if segment["kind"] == "messages":
                    expected, weights = segment["messages"], segment["masks"]
                else:
                    rid = segment["request_id"]
                    if rid in seen_requests or origin["request_id"] != rid:
                        raise ValueError("FINAL_REQUEST_DUPLICATION")
                    seen_requests.add(rid)
                    request_ids.append(rid)
                    stored = db.execute("SELECT request,result FROM jobs WHERE id=? AND status='done'", (rid,)).fetchone()
                    if stored is None:
                        raise ValueError("FINAL_REQUEST_NOT_IN_CALL_JOURNAL")
                    request, result = map(unpack, stored)
                    if result.get("producer") == REUSED_CONTEXT_PRODUCER:
                        donor_path = result["reuse"]["donor_run"]
                        if donor_path not in donors:
                            donors[donor_path] = Donor(Path(donor_path), call_config)
                        donors[donor_path].verify(request, result)
                        counts["verified_reused_llm_contexts"] += 1
                    if request["input_sha256"] != segment["input_sha256"]:
                        raise ValueError("FINAL_REQUEST_INPUT_MISMATCH")
                    expected, weights, decision = evaluate_result(request, result, profile)
                    if origin["review_decision"] != decision:
                        raise ValueError("FINAL_DECISION_CHANGED")
                    producers[result.get("producer", "llm_full_payload_v1")] += 1
                    decisions[decision["decision"]] += 1
                    counts["restored_supervised_calls"] += decision["restored_supervised_calls"]
                    counts["promoted_target_calls"] += decision["promoted_calls"]
                if messages[start:end] != expected or masks[start:end] != weights:
                    raise ValueError("FINAL_MESSAGE_OR_SUPERVISION_CHANGED")
            if target_cursor != len(messages) or source_coverage != list(range(len(raw_messages))) or request_ids != staged["pending_request_ids"]:
                raise ValueError("FINAL_SOURCE_EVENT_LOSS_OR_DUPLICATION")
            sources[original["source_scaffold"]] += 1
            counts["rows"] += 1
            counts["supervised_messages"] += checks["supervised_messages"]
            counts["messages"] += len(messages)
            counts["rows_with_finish"] += int(checks["terminal"])
            if counts["rows"] % 1000 == 0:
                print(json.dumps({"audited_rows": counts["rows"]}), flush=True)
        job_count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if len(seen_requests) != job_count:
            raise ValueError("FINAL_REQUEST_COVERAGE")
        return {"passed": True, "target": target, "counts": dict(counts), "by_source": dict(sources),
                "processed_regions": job_count, "producers": dict(producers), "decisions": dict(decisions),
                "api_attempts": db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
                "split_policy": "all_train", "trajectory_replay": "NOT_RUN",
                "llm_execution": "REAL_HTTP_CALLS_WITH_SEPARATELY_LABELLED_MECHANICAL_CONTEXT",
                "final_manifest_sha256": file_sha(final / "manifest.json"),
                "call_config_sha256": file_sha(calls / "config.json")}
    finally:
        for donor in donors.values():
            donor.close()
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.stage, args.calls, args.final)
    with args.out.open("x") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
