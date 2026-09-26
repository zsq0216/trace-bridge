"""Prepare rewrite queues and export SFT Parquet files."""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import VERSION as COMPILER_VERSION
from .rewrite_queue import (PROMPT, RESULT_SCHEMA, RESOLUTION_PROMPT, RESOLUTION_SCHEMA,
                            VERSION, prompt_messages, response_schema, result_field)
from .resolution import evaluate_result
from .run import file_sha
from .sources import dumps
from .training import assemble, profiles, stage_record


SCHEMA = pa.schema([(n, pa.string()) for n in (
    "id", "source_scaffold", "task_group_key", "split", "source_split", "source_dataset", "source_revision",
    "teacher", "acceptance_basis", "target_harness", "source_payload_sha256", "messages_json",
    "tools_json", "bridge_provenance_json", "observation_policy")]
    + [("source_reference_tokens", pa.int64()), ("source_row_index", pa.int64()),
       ("message_loss_mask", pa.list_(pa.int64())), ("training_ready", pa.bool_()),
       ("has_target_finish", pa.bool_()), ("rewrite_count", pa.int64()),
       ("masked_rewrite_count", pa.int64()), ("promoted_region_count", pa.int64()),
       ("promoted_target_calls", pa.int64())])


def jsonlines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class ParquetSink:
    def __init__(self, path):
        self.writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")
        self.rows = []

    def write(self, row):
        self.rows.append(row)
        if len(self.rows) >= 16:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=SCHEMA))
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()


def verified_inputs(source, compiled, targets):
    manifest = json.loads((compiled / "manifest.json").read_text())
    if manifest.get("version") != COMPILER_VERSION:
        raise ValueError("COMPILED_VERSION_MISMATCH:recompile the source with this release")
    digest = file_sha(source)
    if digest not in manifest["input"].values():
        raise ValueError("COMPILED_SOURCE_HASH_MISMATCH")
    names = ["ir.jsonl.gz"] + [t + ".plans.jsonl.gz" for t in targets]
    hashes = {str(source.resolve()): digest, str((compiled / "manifest.json").resolve()): file_sha(compiled / "manifest.json")}
    for name in names:
        path = compiled / name
        actual = file_sha(path)
        if actual != manifest["outputs"].get(name):
            raise ValueError("COMPILED_FILE_HASH_MISMATCH:" + name)
        hashes[str(path.resolve())] = actual
    return hashes


def finish_manifest(out, data):
    data.update({"version": VERSION, "trajectory_replay": "NOT_RUN", "llm_execution": "NOT_RUN_BY_EXPORTER",
                 "split_policy": "all_train_original_label_in_source_split",
                 "observation_policy": "recorded_source_projection_no_target_execution",
                 "tokenizer_length_validation": "NOT_RUN_NO_TRAINING_TOKENIZER_SELECTED",
                 "validation": "JSON/schema, call-result pairing, source coverage, masks and provenance; not LLM semantic entailment",
                 "code": {p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                 "profile_sha256": file_sha(Path(__file__).with_name("training_profiles.json"))})
    for path in out.glob("*.partial"):
        path.rename(path.with_suffix(""))
    data["outputs"] = {p.name: file_sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(dumps(data["counts"]), flush=True)
    return data


def prepare(source: Path, compiled: Path, out: Path, targets=None, limit=0):
    selected = targets or list(profiles())
    profile_map = profiles()
    hashes = verified_inputs(source, compiled, selected)
    out.mkdir(parents=True, exist_ok=False)
    counts = {t: Counter() for t in selected}
    rows = 0
    streams = {"ir": iter(jsonlines(compiled / "ir.jsonl.gz")),
               **{t: iter(jsonlines(compiled / (t + ".plans.jsonl.gz"))) for t in selected}}
    with ExitStack() as stack:
        archive = stack.enter_context(gzip.open(out / "source.records.jsonl.gz.partial", "wt", encoding="utf-8", compresslevel=3))
        staged = {t: stack.enter_context(gzip.open(out / (t + ".staged.jsonl.gz.partial"), "wt", encoding="utf-8", compresslevel=3)) for t in selected}
        queues = {t: stack.enter_context(gzip.open(out / (t + ".rewrite_requests.jsonl.gz.partial"), "wt", encoding="utf-8", compresslevel=3)) for t in selected}
        ready = {t: ParquetSink(out / (t + ".ready.parquet.partial")) for t in selected}
        for writer in ready.values():
            stack.callback(writer.close)
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16):
            for record in batch.to_pylist():
                if limit and rows >= limit:
                    break
                ir = next(streams["ir"])
                if ir["source_row_index"] != rows:
                    raise ValueError("SOURCE_ROW_INDEX_MISMATCH")
                archive.write(dumps(record) + "\n")
                for target in selected:
                    plan_row = next(streams[target])
                    item, requests = stage_record(record, ir, plan_row, profile_map[target])
                    staged[target].write(dumps(item) + "\n")
                    for request in requests:
                        queues[target].write(dumps(request) + "\n")
                    counts[target].update(item["stats"])
                    counts[target]["source_rows"] += 1
                    row, status = assemble(item, profile_map[target], {}, {})
                    counts[target]["rows/" + status["status"]] += 1
                    if row is not None:
                        ready[target].write(row)
                        counts[target]["exported_supervised_messages"] += sum(row["message_loss_mask"])
                rows += 1
                if rows % 1000 == 0:
                    print(dumps({"staged_rows": rows, "targets": {t: {"requests": counts[t]["rewrite_regions"], "ready_rows": counts[t]["rows/exported"]} for t in selected}}), flush=True)
            if limit and rows >= limit:
                break
        if not limit:
            for name, stream in streams.items():
                if next(stream, None) is not None:
                    raise ValueError("EXTRA_COMPILED_ROWS:" + name)
    (out / "profiles.json").write_text(dumps(profile_map) + "\n")
    (out / "rewrite_spec.json").write_text(json.dumps({"version": VERSION, "routes": {
        "context_rewrite": {"system_prompt": PROMPT, "response_schema": RESULT_SCHEMA, "result_field": "rewrite"},
        "candidate_resolution": {"system_prompt": RESOLUTION_PROMPT, "response_schema": RESOLUTION_SCHEMA, "result_field": "resolution"}},
        "result_envelope": {"request_id": "from request", "input_sha256": "from request",
                            "<result_field>": "object matching the route response_schema"}}, ensure_ascii=False, indent=2) + "\n")
    return finish_manifest(out, {"phase": "prepare", "inputs": hashes, "source_rows": rows,
                                "counts": counts, "targets": selected, "limit": limit})


def finalize(stage: Path, result_paths: list[Path], out: Path, targets=None):
    manifest = json.loads((stage / "manifest.json").read_text())
    if manifest["phase"] != "prepare":
        raise ValueError("FINALIZE_REQUIRES_PREPARED_STAGE")
    if manifest.get("version") != VERSION:
        raise ValueError("STAGE_VERSION_MISMATCH:prepare the data with this release")
    selected = targets or manifest["targets"]
    if len(set(selected)) != len(selected) or not set(selected) <= set(manifest["targets"]):
        raise ValueError("FINALIZE_TARGET_SELECTION_INVALID")
    profile_map = json.loads((stage / "profiles.json").read_text())
    for name, expected in manifest["outputs"].items():
        if file_sha(stage / name) != expected:
            raise ValueError("STAGE_HASH_MISMATCH:" + name)
    out.mkdir(parents=True, exist_ok=False)
    counts = {t: Counter() for t in selected}
    db = sqlite3.connect(out / "rewrite_results.sqlite")
    db.execute("CREATE TABLE results (id TEXT PRIMARY KEY, body TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0)")
    try:
        for path in result_paths:
            for result in jsonlines(path):
                rid, body = result["request_id"], dumps(result)
                existing = db.execute("SELECT body FROM results WHERE id=?", (rid,)).fetchone()
                if existing and existing[0] != body:
                    raise ValueError("CONFLICTING_REWRITE_RESULTS:" + rid)
                db.execute("INSERT OR IGNORE INTO results(id,body) VALUES(?,?)", (rid, body))
        db.commit()
        with ExitStack() as stack:
            pending = stack.enter_context(gzip.open(out / "pending.rows.jsonl.gz.partial", "wt", encoding="utf-8", compresslevel=3))
            decisions = stack.enter_context(gzip.open(out / "review.decisions.jsonl.gz.partial", "wt", encoding="utf-8", compresslevel=3))
            for target in selected:
                writer = ParquetSink(out / (target + ".parquet.partial"))
                stack.callback(writer.close)
                requests = iter(jsonlines(stage / (target + ".rewrite_requests.jsonl.gz")))
                for item in jsonlines(stage / (target + ".staged.jsonl.gz")):
                    request_map, results, evaluated = {}, {}, {}
                    for rid in item["pending_request_ids"]:
                        request = next(requests)
                        if request["request_id"] != rid:
                            raise ValueError("REWRITE_QUEUE_ORDER_MISMATCH")
                        request_map[rid] = request
                        found = db.execute("SELECT body FROM results WHERE id=?", (rid,)).fetchone()
                        if found:
                            result = json.loads(found[0])
                            evaluated[rid] = evaluate_result(request, result, profile_map[target])
                            decision = evaluated[rid][2]
                            decisions.write(dumps({"id": item["id"], "target": target, **decision}) + "\n")
                            counts[target]["review_decision/" + decision["decision"]] += 1
                            counts[target]["review_accepted_target_calls"] += decision["promoted_calls"]
                            counts[target]["review_restored_supervised_calls"] += decision["restored_supervised_calls"]
                            for reason in decision["rejection_reasons"]:
                                counts[target]["promotion_rejection/" + reason] += 1
                            results[rid] = result
                            db.execute("UPDATE results SET used=1 WHERE id=?", (rid,))
                            counts[target]["accepted_rewrites"] += 1
                    row, status = assemble(item, profile_map[target], request_map, results, evaluated)
                    counts[target]["source_rows"] += 1
                    counts[target]["rows/" + status["status"]] += 1
                    if row is not None:
                        writer.write(row)
                        counts[target]["supervised_messages"] += sum(row["message_loss_mask"])
                        counts[target]["masked_rewrite_messages"] += row["masked_rewrite_count"]
                        counts[target]["exported_promoted_regions"] += row["promoted_region_count"]
                        counts[target]["exported_promoted_target_calls"] += row["promoted_target_calls"]
                    else:
                        pending.write(dumps({"id": item["id"], "target": target, **status}) + "\n")
                if next(requests, None) is not None:
                    raise ValueError("EXTRA_REWRITE_REQUESTS")
        if db.execute("SELECT COUNT(*) FROM results WHERE used=0").fetchone()[0]:
            raise ValueError("UNKNOWN_REWRITE_RESULTS")
        db.commit()
    finally:
        db.close()
    return finish_manifest(out, {"phase": "finalize", "stage": str(stage.resolve()),
        "inputs": {str((stage / "manifest.json").resolve()): file_sha(stage / "manifest.json"),
                   **{str(p.resolve()): file_sha(p) for p in result_paths}},
        "counts": counts, "targets": selected})


def export_prompts(queue: Path, out: Path, limit=0):
    opener = gzip.open if str(out).endswith(".gz") else open
    with opener(out, "xt", encoding="utf-8") as f:
        for i, request in enumerate(jsonlines(queue)):
            if limit and i >= limit:
                break
            f.write(dumps({"request_id": request["request_id"], "input_sha256": request["input_sha256"],
                           "route": request["payload"].get("route", "context_rewrite"),
                           "messages": prompt_messages(request), "response_schema": response_schema(request),
                           "result_field": result_field(request)}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--compiled", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--targets", nargs="+", choices=list(profiles()))
    prep.add_argument("--limit", type=int, default=0)
    final = sub.add_parser("finalize")
    final.add_argument("--stage", type=Path, required=True)
    final.add_argument("--results", dest="result_paths", type=Path, nargs="*", default=[])
    final.add_argument("--targets", nargs="+", choices=list(profiles()))
    final.add_argument("--out", type=Path, required=True)
    prompts = sub.add_parser("prompts")
    prompts.add_argument("--queue", type=Path, required=True)
    prompts.add_argument("--out", type=Path, required=True)
    prompts.add_argument("--limit", type=int, default=0)
    args = vars(parser.parse_args())
    phase = args.pop("phase")
    if args.get("limit", 0) < 0 or (args.get("targets") and len(set(args["targets"])) != len(args["targets"])):
        parser.error("limit must be nonnegative and targets must be unique")
    {"prepare": prepare, "finalize": finalize, "prompts": export_prompts}[phase](**args)


if __name__ == "__main__":
    main()
