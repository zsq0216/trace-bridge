"""Compile source Parquet records into IR and target plans."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import pyarrow.parquet as pq

from . import VERSION
from .ir import CompileContext
from .sources import content_text, declared_root, definitions, dumps, parse_record, sha
from .targets import TARGETS, compile_operation


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_success(call, messages):
    """Recognize a bound source acknowledgement of a successful write or edit."""
    if len(call.observation_indices) != 1:
        return False
    index = call.observation_indices[0]
    if index <= call.message_index:
        return False
    body = content_text(messages[index].get("content"))
    if call.source == "OpenCode":
        expected = {"write": "Wrote file successfully.", "edit": "Edit applied successfully."}.get(call.source_tool)
        return bool(expected and (body == expected or body.startswith(expected + "\n")))
    if call.source == "Claude Code":
        path = call.operation.parameters.get("path")
        if call.source_tool == "Write":
            return body.startswith("File created successfully at: " + str(path)) or body.startswith("The file " + str(path) + " has been updated")
        if call.source_tool == "Edit":
            return body.startswith("The file " + str(path) + " has been updated") or body.startswith("The file " + str(path) + " has been edited")
    return False


def run(source: Path, out: Path, targets: list[str], limit=0):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    package = Path(__file__).parent
    code = {p.name: file_sha(p) for p in sorted(package.glob("*.py"))}
    input_hash = file_sha(source)
    stats = Counter()
    by_source = defaultdict(Counter)
    by_target = defaultdict(Counter)
    reasons = defaultdict(Counter)
    requirements = defaultdict(Counter)
    catalog = {}
    examples = {}
    seen = set()
    output_names = ["ir.jsonl.gz"] + [t + ".plans.jsonl.gz" for t in targets]
    with ExitStack() as stack:
        writers = {n: stack.enter_context(gzip.open(out / (n + ".partial"), "wt", encoding="utf-8", compresslevel=3)) for n in output_names}
        stop = False
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16):
            for record in batch.to_pylist():
                if limit and stats["rows"] >= limit:
                    stop = True
                    break
                if record["id"] in seen:
                    raise ValueError("DUPLICATE_TRAJECTORY_ID")
                seen.add(record["id"])
                messages = json.loads(record["messages_json"])
                tools = json.loads(record["tools_json"])
                for definition in definitions(tools).values():
                    catalog[sha(definition)] = definition
                record["messages"], record["tools"] = messages, tools
                calls = parse_record(record)
                ir = {"version": VERSION, "id": record["id"], "source_scaffold": record["source_scaffold"],
                      "split": record["split"], "source_payload_sha256": sha({"messages": messages, "tools": tools}),
                      "source_row_index": stats["rows"], "operations": [c.to_dict() for c in calls]}
                writers["ir.jsonl.gz"].write(dumps(ir) + "\n")
                root = declared_root(messages)
                plans = {target: [] for target in targets}
                cwd = root
                for call in calls:
                    op = call.operation
                    ctx = CompileContext(cwd=cwd, source_succeeded=bound_success(call, messages))
                    stats["source_calls"] += 1
                    by_source[call.source][op.kind] += 1
                    for target in targets:
                        plan = compile_operation(op, target, ctx)
                        item = {"call_id": call.call_id, "source_message_index": call.message_index,
                                "source_call_index": call.call_index, "operation": op.kind,
                                "source_observation_indices": call.observation_indices, **plan.to_dict()}
                        plans[target].append(item)
                        by_target[target][plan.status] += 1
                        by_target[target][op.kind + "/" + plan.status] += 1
                        by_target[target]["actions"] += len(plan.actions)
                        for action in plan.actions:
                            by_target[target]["implementation/" + action.implementation] += 1
                        if plan.reason:
                            reasons[target][plan.reason] += 1
                        requirements[target].update(plan.requirements)
                        key = target + "/" + op.kind
                        if key not in examples and op.kind in {"apply_patch", "write_file", "replace_text", "find_files", "delete_file"}:
                            examples[key] = {"id": record["id"], "source": call.to_dict(), "plan": item}
                    if (op.kind == "run_command" and op.contracts.get("scope") == "persistent"
                            and not op.parameters.get("is_input") and not op.parameters.get("cwd")):
                        cwd = None
                for target in targets:
                    writers[target + ".plans.jsonl.gz"].write(dumps({"version": VERSION, "id": record["id"], "target": target, "plans": plans[target]}) + "\n")
                stats["rows"] += 1
                if stats["rows"] % 1000 == 0:
                    print(dumps({**stats, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            if stop:
                break
    if code != {p.name: file_sha(p) for p in sorted(package.glob("*.py"))}:
        raise ValueError("COMPILER_CHANGED_DURING_RUN")
    for name in output_names:
        (out / (name + ".partial")).rename(out / name)
    (out / "source_tool_definitions.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    (out / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    manifest = {"version": VERSION, "input": {str(source.resolve()): input_hash}, "code": code,
                "outputs": {name: file_sha(out / name) for name in output_names}, "totals": dict(stats),
                "by_source": by_source, "by_target": by_target, "unsupported_reasons": reasons,
                "unresolved_requirements": requirements, "elapsed_seconds": time.monotonic() - started,
                "training_ready": False, "execution": "NOT_RUN",
                "validation_policy": "static_compilation",
                "trajectory_replay": "NOT_REQUESTED",
                "notes": ["Source observations are referenced by their original indices.",
                          "Conditional plans require additional source evidence before export."]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(dumps({"output": str(out), **stats, "targets": by_target}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS), default=list(TARGETS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0 or len(args.targets) != len(set(args.targets)):
        parser.error("limit must be nonnegative; targets must be unique")
    run(args.source, args.out, args.targets, args.limit)
