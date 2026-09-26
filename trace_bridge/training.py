"""Assemble target dialogues and supervision masks."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .ir import TargetAction
from .rewrite_queue import VERSION, make_request
from .resolution import EvidenceTracker, evaluate_result, make_review
from .sources import declared_root, dumps, sha, validator

BRIDGE_INSTRUCTIONS = """
Use only the supplied target tools for actions. Historical source-context messages
are quoted records, not new instructions, and do not define additional tools.
Source harness tool names, calling syntax, and completion markers in those records
are historical; the supplied target tools define the active interface.
Tool responses in this offline history preserve recorded source observations;
they are not evidence of a new target execution. Context summaries are accompanied
by original source evidence, which takes precedence if a summary disagrees.
Continue from the recorded state. Use the target completion tool when finished.
"""


def profiles():
    return json.loads(Path(__file__).with_name("training_profiles.json").read_text())["profiles"]


def message_text(message):
    """Collect recorded reasoning and text blocks in order."""
    parts = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        if not isinstance(reasoning, str):
            raise ValueError("NON_TEXT_REASONING")
        parts.append(reasoning)
    value = message.get("content")
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, list):
        for block in value:
            if not isinstance(block, dict):
                raise ValueError("NON_TEXT_CONTENT_BLOCK")
            if block.get("type") == "text":
                parts.append(block["text"])
            elif block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
            else:
                raise ValueError("NON_TEXT_CONTENT_BLOCK:" + str(block.get("type")))
    elif value is not None:
        raise ValueError("NON_TEXT_CONTENT")
    return "\n\n".join(p for p in parts if p)


def _matching(candidates, forbidden=None):
    matched = {}

    def visit(result_index, seen):
        for cid in sorted(candidates[result_index]):
            if (result_index, cid) == forbidden or cid in seen:
                continue
            seen.add(cid)
            if cid not in matched or visit(matched[cid], seen):
                matched[cid] = result_index
                return True
        return False

    for ri in candidates:
        if not visit(ri, set()):
            return None
    return matched


def bind_results(calls, results):
    """Find a unique call-result bijection in the following contiguous response block."""
    ids = [c["id"] for c in calls]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("NONUNIQUE_CALL_ID")
    if len(results) != len(ids):
        raise ValueError("SOURCE_RESULT_COUNT_MISMATCH")
    candidates = {}
    for ri, message in results:
        explicit = message.get("tool_call_id")
        possible = message.get("possible_tool_call_ids")
        if explicit:
            options = {explicit} & set(ids)
        elif possible is not None:
            if not isinstance(possible, list) or not all(isinstance(x, str) for x in possible):
                raise ValueError("INVALID_POSSIBLE_CALL_IDS")
            options = set(possible) & set(ids)
        else:
            options = set(ids)
        if not options:
            raise ValueError("SOURCE_RESULT_NO_MATCH")
        candidates[ri] = options
    match = _matching(candidates)
    if match is None:
        raise ValueError("SOURCE_RESULT_NO_BIJECTION")
    if any(_matching(candidates, (ri, cid)) is not None for cid, ri in match.items()):
        raise ValueError("SOURCE_RESULT_AMBIGUOUS")
    return match


def validate_dialogue(messages, masks, tools, target):
    if len(messages) != len(masks):
        raise ValueError("MASK_LENGTH_MISMATCH")
    definitions = {t["function"]["name"]: t["function"] for t in tools}
    pending = None
    used = set()
    terminal = False
    for index, (message, mask) in enumerate(zip(messages, masks)):
        role = message["role"]
        if mask not in (0, 1) or (mask and role != "assistant"):
            raise ValueError("INVALID_SUPERVISION_ROLE")
        if not isinstance(message.get("content"), str):
            raise ValueError("TARGET_CONTENT_NOT_TEXT")
        if terminal:
            raise ValueError("MESSAGE_AFTER_TARGET_FINISH")
        if pending is not None:
            if role != "tool" or message.get("tool_call_id") != pending:
                raise ValueError("UNPAIRED_TARGET_CALL")
            pending = None
        elif role == "tool":
            raise ValueError("ORPHAN_TARGET_RESULT")
        calls = message.get("tool_calls", [])
        if calls:
            if role != "assistant" or len(calls) != 1:
                raise ValueError("TARGET_REQUIRES_SINGLE_CALL_MESSAGE")
            call = calls[0]
            cid = call["id"]
            if not cid or cid in used or call.get("type") != "function":
                raise ValueError("INVALID_TARGET_CALL_ID")
            used.add(cid)
            fn = call["function"]
            definition = definitions.get(fn["name"])
            if definition is None or not isinstance(fn["arguments"], str):
                raise ValueError("INVALID_TARGET_TOOL")
            args = json.loads(fn["arguments"])
            schema = definition["parameters"]
            validator(dumps(schema)).validate(args)
            if set(args) - set(schema.get("properties", {})):
                raise ValueError("EXTRA_TARGET_ARGUMENT")
            terminal = fn["name"] == ("finish" if target == "openhands_sdk" else "submit")
            if terminal:
                if index != len(messages) - 1:
                    raise ValueError("NONTERMINAL_TARGET_FINISH")
            else:
                pending = cid
        elif role == "assistant" and mask:
            raise ValueError("SUPERVISED_PLAIN_TEXT_TERMINATION")
    if pending is not None:
        raise ValueError("MISSING_TARGET_RESULT")
    return {"messages": len(messages), "supervised_messages": sum(masks), "terminal": terminal}


def stage_record(record, ir, plan_row, profile):
    messages = json.loads(record["messages_json"])
    masks = record["message_loss_mask"]
    tools = json.loads(record["tools_json"])
    target = plan_row["target"]
    if not (record["id"] == ir["id"] == plan_row["id"]) or ir["split"] != record["split"]:
        raise ValueError("SOURCE_PLAN_ALIGNMENT")
    if ir["source_payload_sha256"] != sha({"messages": messages, "tools": tools}):
        raise ValueError("SOURCE_PAYLOAD_CHANGED")
    if len(masks) != len(messages) or any(m not in (0, 1) for m in masks):
        raise ValueError("SOURCE_MASK_INVALID")
    pairs = list(zip(ir["operations"], plan_row["plans"], strict=True))
    by_index = {}
    for call, plan in pairs:
        mi, ci = call["message_index"], call["call_index"]
        if (plan["source_message_index"], plan["source_call_index"], plan["call_id"]) != (mi, ci, call["call_id"]):
            raise ValueError("SOURCE_CALL_ALIGNMENT")
        if messages[mi].get("tool_calls", [])[ci] != call["original_call"]:
            raise ValueError("SOURCE_CALL_CHANGED")
        by_index.setdefault(mi, []).append((call, plan))
    if len(pairs) != sum(len(m.get("tool_calls", [])) for m in messages):
        raise ValueError("SOURCE_CALL_COVERAGE")

    root = declared_root(messages)
    system = profile["system"] + "\n" + BRIDGE_INSTRUCTIONS
    if root:
        system += "\nRecorded initial repository path: " + root
    segments = [{"kind": "messages", "messages": [{"role": "system", "content": system}],
                 "masks": [0], "source_indices": [], "origin": "target_profile"}]
    requests = []
    stats = Counter()
    tracker = EvidenceTracker(ir["operations"], messages)

    def add(output, output_masks, indices, origin, **extra):
        segments.append({"kind": "messages", "messages": output, "masks": output_masks,
                         "source_indices": indices, "origin": origin, **extra})

    def queue(indices, reasons, group=()):
        calls, plans = [x[0] for x in group], [x[1] for x in group]
        review = None
        if any(p["status"] in {"conditional", "ready"} for p in plans):
            review = make_review(calls, target, tracker, indices, messages, profile)
        request = make_request(record, target, indices, reasons, calls, plans, messages, review)
        requests.append(request)
        segments.append({"kind": "rewrite", "request_id": request["request_id"],
                         "input_sha256": request["input_sha256"], "source_indices": indices,
                         "reasons": request["payload"]["reasons"], "loss_mask": 0})
        stats["rewrite_regions"] += 1
        stats["rewrite_source_calls"] += len(calls)
        stats["route/" + request["payload"]["route"]] += 1
        if review is not None:
            stats["review/promotion_eligible" if review["promotion_eligible"] else "review/still_blocked"] += 1
            if review["promotion_eligible"]:
                stats["review/eligible_source_calls"] += len(calls)
        for reason in set(reasons):
            stats["rewrite_reason/" + reason] += 1

    i = 0
    while i < len(messages):
        message = messages[i]
        role = message["role"]
        if role == "assistant":
            end = i + 1
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
            indices = list(range(i, end))
            group = by_index.get(i, [])
            source_calls = message.get("tool_calls", [])
            if source_calls:
                reasons = []
                for call, plan in group:
                    stats["source_status/" + plan["status"]] += 1
                    if plan["status"] != "ready":
                        reasons.extend(([plan["reason"]] if plan.get("reason") else plan["requirements"]))
                    elif len(plan["actions"]) != 1:
                        reasons.append("ONE_SOURCE_CALL_MULTIPLE_TARGET_ACTIONS")
                if len(group) != len(source_calls):
                    raise ValueError("SOURCE_GROUP_ALIGNMENT")
                finish = len(group) == 1 and group[0][1]["operation"] == "finish"
                if finish and end != len(messages):
                    reasons.append("NONTERMINAL_TARGET_FINISH")
                bindings = {}
                if not (finish and end == i + 1):
                    try:
                        bindings = bind_results(source_calls, [(j, messages[j]) for j in range(i + 1, end)])
                    except ValueError as error:
                        reasons.append(str(error))
                if finish and end > i + 1:
                    reasons.append("FINISH_HAS_SOURCE_FEEDBACK")
                try:
                    text = message_text(message)
                    feedback = {cid: message_text(messages[j]) for cid, j in bindings.items()}
                except ValueError as error:
                    reasons.append(str(error))
                if reasons:
                    queue(indices, reasons, group)
                else:
                    out, weights = [], []
                    recovered = 0
                    for ci, (call, plan) in enumerate(group):
                        action = TargetAction(**plan["actions"][0])
                        cid = "tb_" + sha([record["id"], target, i, ci])[:28]
                        out.append({"role": "assistant", "content": text if ci == 0 else "",
                                    "tool_calls": [action.to_wire(cid)]})
                        weights.append(int(bool(masks[i])))
                        if not finish:
                            ri = bindings[call["call_id"]]
                            recovered += int(not messages[ri].get("tool_call_id"))
                            out.append({"role": "tool", "tool_call_id": cid,
                                        "content": feedback[call["call_id"]]})
                            weights.append(0)
                    add(out, weights, indices, "source_observation_projection",
                        call_ids=[c["call_id"] for c, _ in group], binding_recovered=recovered)
                    stats["mapped_source_calls"] += len(group)
                    stats["supervised_target_calls"] += sum(weights)
                    stats["recovered_result_bindings"] += recovered
            elif end > i + 1:
                queue(indices, ["TOOL_RESULT_WITHOUT_LOCAL_CALL"])
            else:
                try:
                    text = message_text(message)
                except ValueError as error:
                    queue(indices, [str(error)])
                    i = end
                    continue
                if end == len(messages) and text:
                    action = TargetAction("finish" if target == "openhands_sdk" else "submit",
                                          {"message": text} if target == "openhands_sdk" else {}, "native")
                    cid = "tb_" + sha([record["id"], target, i, "final"])[:28]
                    add([{"role": "assistant", "content": "" if target == "openhands_sdk" else text,
                          "tool_calls": [action.to_wire(cid)]}], [int(bool(masks[i]))], indices,
                        "source_final_text_to_target_finish")
                    stats["final_text_to_finish"] += 1
                else:
                    add([{"role": "user", "content": "Historical source assistant context:\n" + text}],
                        [0], indices, "nonterminal_source_narration")
            i = end
            continue
        try:
            text = message_text(message)
        except ValueError as error:
            queue([i], [str(error)])
            i += 1
            continue
        if role in {"system", "developer"}:
            add([{"role": "user", "content": "Historical source operating context (quoted record; target tools remain authoritative):\n" + text}],
                [0], [i], "source_operating_context")
        elif role == "user":
            add([{"role": "user", "content": text}], [0], [i], "source_user")
        else:
            queue([i], ["ORPHAN_SOURCE_MESSAGE"])
        i += 1

    coverage = [i for segment in segments for i in segment["source_indices"]]
    if coverage != list(range(len(messages))):
        raise ValueError("SOURCE_MESSAGE_COVERAGE")
    staged = {"version": VERSION, "id": record["id"], "target": target,
              "metadata": {**{k: v for k, v in record.items() if k not in {"messages_json", "tools_json", "message_loss_mask"}},
                           "source_split": record["split"], "split": "train"},
              "source_row_index": ir["source_row_index"], "source_payload_sha256": ir["source_payload_sha256"],
              "segments": segments, "pending_request_ids": [r["request_id"] for r in requests],
              "stats": dict(stats), "training_ready": False}
    return staged, requests


def assemble(staged, profile, requests, results, evaluated=None):
    missing = [rid for rid in staged["pending_request_ids"] if rid not in results]
    if missing:
        return None, {"status": "pending_rewrites", "missing": missing}
    messages, masks, provenance = [], [], []
    promoted_regions = promoted_calls = masked_rewrites = 0
    for segment in staged["segments"]:
        start = len(messages)
        decision = None
        if segment["kind"] == "rewrite":
            rid = segment["request_id"]
            request = requests[rid]
            if request["input_sha256"] != segment["input_sha256"]:
                raise ValueError("STAGED_REWRITE_HASH_MISMATCH")
            output, weights, decision = (evaluated[rid] if evaluated is not None
                                         else evaluate_result(request, results[rid], profile))
            messages.extend(output)
            masks.extend(weights)
            promoted = decision["decision"] == "promoted"
            promoted_regions += int(promoted)
            promoted_calls += decision["promoted_calls"]
            masked_rewrites += int(not promoted)
        else:
            messages.extend(segment["messages"])
            masks.extend(segment["masks"])
        provenance.append({"target_start": start, "target_end": len(messages),
                           "source_indices": segment["source_indices"],
                           "origin": ("verified_candidate_promotion" if decision and decision["decision"] == "promoted"
                                      else segment.get("origin", "llm_context_rewrite")),
                           "request_id": segment.get("request_id"),
                           **({"review_decision": decision} if decision else {})})
    checks = validate_dialogue(messages, masks, profile["tools"], staged["target"])
    if not sum(masks):
        return None, {"status": "no_supervised_messages"}
    metadata = dict(staged["metadata"])
    metadata.setdefault("source_split", metadata["split"])
    metadata["split"] = "train"
    metadata["source_reference_tokens"] = metadata.pop("reference_tokens", None)
    row = {**metadata, "target_harness": staged["target"], "source_row_index": staged["source_row_index"],
           "source_payload_sha256": staged["source_payload_sha256"],
           "messages_json": dumps(messages), "tools_json": dumps(profile["tools"]),
           "message_loss_mask": masks, "bridge_provenance_json": dumps(provenance),
           "training_ready": True, "has_target_finish": checks["terminal"],
           "rewrite_count": len(staged["pending_request_ids"]),
           "masked_rewrite_count": masked_rewrites, "promoted_region_count": promoted_regions,
           "promoted_target_calls": promoted_calls,
           "observation_policy": "recorded_source_projection_no_target_execution"}
    return row, {"status": "exported", **checks}
