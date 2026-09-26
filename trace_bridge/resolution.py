"""Resolve candidate actions using recorded source evidence."""
from __future__ import annotations

import re
from dataclasses import asdict

from .ir import CompileContext, Operation, SourceCall, TargetAction
from .patches import apply_update
from .run import bound_success
from .sources import content_text, declared_root, sha
from .targets import TARGETS, absolute, compile_operation, write_command


def source_call(item):
    return SourceCall(**{**item, "operation": Operation(**item["operation"])})


def patch_success(call, messages):
    """Match the complete source patch acknowledgement."""
    if call.source != "Codex-format" or call.operation.kind != "apply_patch" or len(call.observation_indices) != 1:
        return False
    i = call.observation_indices[0]
    if i <= call.message_index or i >= len(messages):
        return False
    changes = call.operation.parameters["files"]
    if any(c.get("move_to") for c in changes):
        return False
    labels = {"update": "Updated file: ", "add": "Added file: ", "delete": "Deleted file: "}
    expected = "Execution output of [apply_patch]:\n" + "\n".join(labels[c["kind"]] + c["path"] for c in changes)
    return content_text(messages[i].get("content")).rstrip("\n") == expected


def fact(kind, indices, **data):
    body = {"kind": kind, "message_indices": sorted(set(indices)), **data}
    return {"evidence_id": sha(body), **body}


class EvidenceTracker:
    """Track known file contents and cwd, invalidating snapshots after opaque mutations."""

    def __init__(self, calls, messages):
        self.calls = [source_call(c) for c in calls]
        self.messages = messages
        self.position = 0
        self.event_position = 0
        first = min((c.message_index for c in self.calls), default=len(messages))
        self.cwd = declared_root(messages[:first])
        self.cwd_fact = fact("declared_initial_cwd", range(first), cwd=self.cwd) if self.cwd else None
        self.files = {}
        self.reads = {}

    def advance(self, index):
        while self.position < len(self.calls) and self.calls[self.position].message_index < index:
            call = self.calls[self.position]
            if any(m["role"] in {"user", "system", "developer"}
                   for m in self.messages[self.event_position:call.message_index]
                   if self.event_position):
                self.files.clear()
                self.reads.clear()
                self.cwd = None
                self.cwd_fact = None
            self._observe(call)
            self.event_position = call.message_index + 1
            self.position += 1
        if self.event_position and any(m["role"] in {"user", "system", "developer"}
                                       for m in self.messages[self.event_position:index]):
            self.files.clear()
            self.reads.clear()
            self.cwd = None
            self.cwd_fact = None
        self.event_position = index

    def _bound_indices(self, call):
        end = call.message_index + 1
        while end < len(self.messages) and self.messages[end]["role"] == "tool":
            end += 1
        if len(call.observation_indices) != 1 or call.observation_indices[0] not in range(call.message_index + 1, end):
            return None
        return [call.message_index, *call.observation_indices]

    def _observe(self, call):
        op = call.operation
        p, c = op.parameters, op.contracts
        indices = self._bound_indices(call)
        if c.get("source_batch_order_unverified"):
            self.files.clear()
            self.reads.clear()
            self.cwd = None
            self.cwd_fact = None
            return
        try:
            path = absolute(p["path"], self.cwd) if "path" in p else None
        except ValueError:
            path = None
        if op.kind == "read_file":
            if path and indices:
                body = content_text(self.messages[indices[-1]].get("content"))
                numbered = ((call.source == "Claude Code" and re.match(r"\s*\d+→", body))
                            or (call.source == "OpenCode" and body.startswith("<path>" + p["path"] + "</path>\n<type>file</type>\n<content>"))
                            or (call.source == "Codex-format" and re.match(r"Execution output of \[read_file\]:\n\s*\d+\|", body)))
                if numbered:
                    self.reads[path] = fact("recorded_read_access", indices, path=path)
            return
        if op.kind in {"find_files", "search_text", "think", "finish"}:
            return
        old = self.files.get(path)
        previous = dict(self.files)
        self.files.clear()
        self.reads.clear()
        if op.kind in {"run_command", "unsupported", "editor_action"}:
            if op.kind != "run_command" or (c.get("scope") != "per_call" and not p.get("cwd")):
                self.cwd = None
                self.cwd_fact = None
            return
        if not indices:
            return
        if op.kind == "write_file" and path and c.get("preserve_content_bytes") and bound_success(call, self.messages):
            self.files[path] = fact("exact_file_snapshot", indices, path=path, content=p["content"])
        elif op.kind == "replace_text" and old and bound_success(call, self.messages):
            before = old["content"]
            if "\r" not in before + p["old"] + p["new"] and before.count(p["old"]) > 0 and (p["mode"] == "all" or before.count(p["old"]) == 1):
                after = before.replace(p["old"], p["new"], -1 if p["mode"] == "all" else 1)
                self.files[path] = fact("exact_file_snapshot", old["message_indices"] + indices, path=path, content=after)
        elif op.kind == "apply_patch" and patch_success(call, self.messages):
            changes = p["files"]
            if len(changes) != 1 or changes[0].get("move_to"):
                return
            change = changes[0]
            try:
                cwd = absolute(p["cwd"], self.cwd) if p.get("cwd") else self.cwd
                path = absolute(change["path"], cwd)
                prior = previous.get(path)
                if change["kind"] == "add":
                    after = change["content"]
                elif change["kind"] == "update" and prior:
                    after = apply_update(prior["content"], change["hunks"])
                    indices += prior["message_indices"]
                else:
                    return
                self.files[path] = fact("exact_file_snapshot", indices, path=path, content=after)
            except ValueError:
                return

    def context(self, call):
        self.advance(call.message_index)
        op = call.operation
        p = op.parameters
        evidence = [self.cwd_fact] if self.cwd_fact else []
        ctx = CompileContext(cwd=self.cwd)
        indices = self._bound_indices(call)
        if indices and bound_success(call, self.messages):
            ctx.source_succeeded = True
            evidence.append(fact("bound_source_success", indices, call_id=call.call_id))
        if indices and patch_success(call, self.messages):
            ctx.source_patch_succeeded = True
            evidence.append(fact("bound_patch_success", indices, call_id=call.call_id))
        paths = []
        try:
            if "path" in p:
                paths.append(absolute(p["path"], ctx.cwd))
            if op.kind == "apply_patch":
                cwd = absolute(p["cwd"], ctx.cwd) if p.get("cwd") else ctx.cwd
                paths.extend(absolute(c["path"], cwd) for c in p["files"])
        except ValueError:
            pass
        for path in dict.fromkeys(paths):
            if path in self.files:
                f = self.files[path]
                ctx.files[path] = f["content"]
                evidence.append(f)
            if path in self.reads:
                ctx.read_paths.add(path)
                evidence.append(self.reads[path])
        data = asdict(ctx)
        data["read_paths"], data["directories"] = sorted(ctx.read_paths), sorted(ctx.directories)
        return data, evidence


def resolved_plan(call, target, context):
    """Recompile a candidate and coalesce single-file patches when the full effect is known."""
    ctx = CompileContext(**{**context, "read_paths": set(context["read_paths"]),
                            "directories": set(context["directories"])})
    op = Operation(**call["operation"])
    plan = compile_operation(op, target, ctx)
    if plan.status == "ready" and len(plan.actions) > 1:
        changes = op.parameters.get("files", [])
        if ctx.source_patch_succeeded and op.kind == "apply_patch" and len(changes) == 1 and changes[0]["kind"] == "update" and not changes[0].get("move_to"):
            cwd = absolute(op.parameters["cwd"], ctx.cwd) if op.parameters.get("cwd") else ctx.cwd
            path = absolute(changes[0]["path"], cwd)
            if isinstance(ctx.files.get(path), str):
                after = apply_update(ctx.files[path], changes[0]["hunks"])
                plan.actions = [TargetAction(TARGETS[target].shell, {"command": write_command(path, after)}, "simple_shell")]
                plan.verified_properties.append("single_file_patch_coalesced_with_one_recorded_response")
    return plan


def make_review(calls, target, tracker, indices, messages, profile):
    from .training import bind_results, message_text, validate_dialogue
    entries = []
    blockers = []
    for item in calls:
        ctx, evidence = tracker.context(source_call(item))
        plan = resolved_plan(item, target, ctx)
        errors = ([plan.reason] if plan.reason else plan.requirements[:])
        if plan.status == "ready" and len(plan.actions) != 1:
            errors.append("ONE_SOURCE_CALL_MULTIPLE_TARGET_ACTIONS")
        entries.append({"call_id": item["call_id"], "context": ctx, "evidence": evidence,
                        "recompiled_plan": plan.to_dict(), "blockers": errors})
        blockers.extend(errors)
    if not calls:
        blockers.append("NO_SOURCE_OPERATION")
    bindings = {}
    try:
        if any(c["operation"]["kind"] == "finish" for c in calls):
            raise ValueError("REVIEW_FINISH_PROMOTION_NOT_SUPPORTED")
        bindings = bind_results([c["original_call"] for c in calls],
                                [(i, messages[i]) for i in indices if messages[i]["role"] == "tool"])
        for i in indices:
            message_text(messages[i])
        if not blockers:
            output, masks = project(calls, entries, bindings, messages, target, "review_check")
            validate_dialogue(output, masks, profile["tools"], target)
    except ValueError as error:
        blockers.append(str(error))
    evidence_indices = sorted({i for e in entries for f in e["evidence"] for i in f["message_indices"]})
    if any(i > max(indices) for i in evidence_indices):
        raise ValueError("REVIEW_FUTURE_EVIDENCE")
    return {"policy": "source_evidence_recompile_v1", "entries": entries,
            "bindings": bindings, "blockers": sorted(set(blockers)),
            "promotion_eligible": not blockers,
            "target_tool_schemas": {t["function"]["name"]: t["function"]["parameters"] for t in profile["tools"]
                                    if any(a["tool"] == t["function"]["name"] for e in entries for a in e["recompiled_plan"]["actions"])},
            "evidence_messages": [{"message_index": i, "message": messages[i]} for i in evidence_indices]}


def project(calls, entries, bindings, messages, target, seed):
    from .training import message_text
    output, masks = [], []
    for n, (call, entry) in enumerate(zip(calls, entries, strict=True)):
        plan = resolved_plan(call, target, entry["context"])
        if plan.status != "ready" or len(plan.actions) != 1:
            raise ValueError("PROMOTION_STILL_CONDITIONAL_OR_MULTIACTION")
        cid = "tb_" + sha([seed, call["call_id"], n])[:28]
        output.append({"role": "assistant", "content": message_text(messages[call["message_index"]]) if n == 0 else "",
                       "tool_calls": [plan.actions[0].to_wire(cid)]})
        masks.append(int(call["supervised"]))
        output.append({"role": "tool", "content": message_text(messages[bindings[call["call_id"]]]), "tool_call_id": cid})
        masks.append(0)
    return output, masks


def evaluate_result(request, result, profile):
    """Validate a review result and return promoted calls or masked context."""
    from .rewrite_queue import render_result, validate_result
    from .training import bind_results, validate_dialogue
    validate_result(request, result)
    payload = request["payload"]
    resolution = result.get("resolution")
    decision = {"request_id": request["request_id"], "input_sha256": request["input_sha256"],
                "result_sha256": sha(result), "route": payload.get("route", "context_rewrite"),
                "decision": "context", "rejection_reasons": [], "promoted_calls": 0,
                "restored_supervised_calls": 0}
    if resolution and resolution["decision"] == "promote":
        review = payload["review"]
        errors = list(review["blockers"])
        calls, entries = payload["source_calls"], review["entries"]
        proposals = resolution["promotions"]
        if len(proposals) != len(calls) or [p["call_id"] for p in proposals] != [c["call_id"] for c in calls]:
            errors.append("PROMOTION_SOURCE_CALL_COVERAGE")
        else:
            for call, entry, proposal in zip(calls, entries, proposals, strict=True):
                plan = resolved_plan(call, payload["target"], entry["context"])
                if plan.status != "ready" or len(plan.actions) != 1:
                    errors.append("PROMOTION_UNRESOLVED_REQUIREMENTS:" + call["call_id"])
                    continue
                canonical = {"tool": plan.actions[0].tool, "arguments": plan.actions[0].arguments}
                if proposal["action"] != canonical:
                    errors.append("PROMOTION_ACTION_NOT_VERIFIED:" + call["call_id"])
                expected = {f["evidence_id"] for f in entry["evidence"]}
                if set(proposal["evidence_ids"]) != expected:
                    errors.append("PROMOTION_EVIDENCE_MISMATCH:" + call["call_id"])
        if not errors:
            messages = {e["message_index"]: e["message"] for e in payload["source_messages"]}
            bindings = bind_results([c["original_call"] for c in calls], [(i, m) for i, m in messages.items() if m["role"] == "tool"])
            output, masks = project(calls, entries, bindings, messages, payload["target"], request["request_id"])
            validate_dialogue(output, masks, profile["tools"], payload["target"])
            decision.update(decision="promoted", promoted_calls=len(calls), restored_supervised_calls=sum(masks),
                            evidence_ids=sorted({f["evidence_id"] for e in entries for f in e["evidence"]}),
                            verification_sha256=sha(review),
                            verified_properties={c["call_id"]: resolved_plan(c, payload["target"], e["context"]).verified_properties
                                                 for c, e in zip(calls, entries, strict=True)})
            return output, masks, decision
        decision.update(decision="promotion_rejected_context", rejection_reasons=sorted(set(errors)))
    return [render_result(request, result)], [0], decision
