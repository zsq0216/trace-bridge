"""Audit staged dialogues, source coverage, and masks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import pyarrow.parquet as pq

from .export_training import jsonlines
from .run import file_sha
from .sources import sha
from .resolution import EvidenceTracker, make_review
from .training import validate_dialogue


def audit(stage: Path):
    manifest = json.loads((stage / "manifest.json").read_text())
    if manifest["phase"] != "prepare":
        raise ValueError("EXPECTED_PREPARE_MANIFEST")
    for name, digest in manifest["outputs"].items():
        if file_sha(stage / name) != digest:
            raise ValueError("ARTIFACT_HASH_MISMATCH:" + name)
    profiles = json.loads((stage / "profiles.json").read_text())
    summary = {}
    for target in manifest["targets"]:
        counts = Counter()
        queue = iter(jsonlines(stage / (target + ".rewrite_requests.jsonl.gz")))
        source = jsonlines(stage / "source.records.jsonl.gz")
        staged = jsonlines(stage / (target + ".staged.jsonl.gz"))
        ready_ids = {}
        seen = set()
        for raw, item in zip_longest(source, staged):
            if raw is None or item is None or raw["id"] != item["id"] or item["id"] in seen:
                raise ValueError("STAGE_SOURCE_ALIGNMENT")
            seen.add(item["id"])
            original = json.loads(raw["messages_json"])
            if item["source_payload_sha256"] != sha({"messages": original, "tools": json.loads(raw["tools_json"])}):
                raise ValueError("SOURCE_HASH_MISMATCH")
            if item["metadata"]["split"] != "train" or item["metadata"].get("source_split") != raw["split"]:
                raise ValueError("TRAIN_SPLIT_OR_SOURCE_PROVENANCE_INVALID")
            coverage = []
            pending_ids = []
            validation_messages, validation_masks = [], []
            source_call_count = sum(len(m.get("tool_calls", [])) for m in original)
            preserved_calls = 0
            tracker = None
            for segment in item["segments"]:
                indices = segment["source_indices"]
                coverage.extend(indices)
                if segment["kind"] == "rewrite":
                    request = next(queue)
                    rid = request["request_id"]
                    if rid != segment["request_id"] or sha(request["payload"]) != request["input_sha256"] or rid != request["input_sha256"]:
                        raise ValueError("REWRITE_PAYLOAD_MISMATCH")
                    payload = request["payload"]
                    if payload["target"] != target or payload["source_id"] != raw["id"] or payload["loss_mask"] != 0 or segment["loss_mask"] != 0:
                        raise ValueError("REWRITE_PROVENANCE_OR_MASK")
                    if payload["split"] != "train" or payload.get("source_split") != raw["split"]:
                        raise ValueError("REWRITE_SPLIT_POLICY_INVALID")
                    if [x["message_index"] for x in payload["source_messages"]] != indices:
                        raise ValueError("REWRITE_INDEX_COVERAGE")
                    for event in payload["source_messages"]:
                        if event["message"] != original[event["message_index"]]:
                            raise ValueError("REWRITE_SOURCE_EVENT_CHANGED")
                    if any(e["message_index"] >= min(indices) for e in payload["context_before"]):
                        raise ValueError("FUTURE_CONTEXT_LEAK")
                    if payload.get("route") == "candidate_resolution":
                        if tracker is None:
                            from .sources import parse_record
                            parsed_calls = [c.to_dict() for c in parse_record(raw)]
                            tracker = EvidenceTracker(parsed_calls, original)
                        selected_calls = [c for c in parsed_calls if c["message_index"] in indices]
                        if selected_calls != payload["source_calls"]:
                            raise ValueError("REVIEW_SOURCE_IR_MISMATCH")
                        if payload["review"] != make_review(selected_calls, target, tracker, indices, original, profiles[target]):
                            raise ValueError("REVIEW_NOT_DERIVED_FROM_SOURCE_EVIDENCE")
                        counts["review/promotion_eligible" if payload["review"]["promotion_eligible"] else "review/still_blocked"] += 1
                    counts["route/" + payload.get("route", "context_rewrite")] += 1
                    preserved_calls += sum(len(original[i].get("tool_calls", [])) for i in indices)
                    pending_ids.append(rid)
                    counts["rewrite_regions"] += 1
                    counts["rewrite_source_calls"] += sum(len(original[i].get("tool_calls", [])) for i in indices)
                    validation_messages.append({"role": "user", "content": "Pending historical context."})
                    validation_masks.append(0)
                else:
                    validation_messages.extend(segment["messages"])
                    validation_masks.extend(segment["masks"])
                    native = [m for m in segment["messages"] if m.get("tool_calls")]
                    if segment["origin"] == "source_observation_projection":
                        preserved_calls += len(native)
                        expected_mask = int(bool(raw["message_loss_mask"][indices[0]]))
                        for message, mask in zip(segment["messages"], segment["masks"], strict=True):
                            if mask != (expected_mask if message["role"] == "assistant" else 0):
                                raise ValueError("SOURCE_SUPERVISION_CHANGED")
                        counts["mapped_source_calls"] += len(native)
            if preserved_calls != source_call_count:
                raise ValueError("SOURCE_CALL_LOSS_OR_DUPLICATION")
            if coverage != list(range(len(original))) or pending_ids != item["pending_request_ids"]:
                raise ValueError("SOURCE_MESSAGE_LOSS_OR_DUPLICATION")
            checks = validate_dialogue(validation_messages, validation_masks, profiles[target]["tools"], target)
            if not pending_ids and checks["supervised_messages"]:
                ready_ids[item["id"]] = ("train", raw["split"], sha([validation_messages, validation_masks]))
            counts["source_rows"] += 1
            counts["source_calls"] += source_call_count
        if next(queue, None) is not None:
            raise ValueError("EXTRA_REWRITE_REQUESTS")
        exported = []
        for batch in pq.ParquetFile(stage / (target + ".ready.parquet")).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                messages, masks = json.loads(row["messages_json"]), row["message_loss_mask"]
                validate_dialogue(messages, masks, json.loads(row["tools_json"]), target)
                if (row["split"], row["source_split"], sha([messages, masks])) != ready_ids.get(row["id"]):
                    raise ValueError("READY_ROW_DIFFERS_FROM_STAGE")
                if not row["training_ready"] or row["rewrite_count"] != 0:
                    raise ValueError("PENDING_REWRITE_LEAKED_TO_READY")
                exported.append(row["id"])
                counts["ready_supervised_messages"] += sum(masks)
        if exported != list(ready_ids):
            raise ValueError("READY_ROW_COVERAGE")
        counts["ready_rows"] = len(exported)
        summary[target] = dict(counts)
    return {"passed": True, "targets": summary, "trajectory_replay": "NOT_RUN",
            "llm_execution": "NOT_RUN", "tokenizer_validation": "NOT_RUN"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = audit(args.stage)
    if args.out:
        with args.out.open("x") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
