"""Reuse identical context rewrites across target harnesses."""
from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .llm_backfill import (envelope, export_results, messages_for, model_json, pack,
                          prepare_db, requires_candidate_review, unpack)
from .resolution import evaluate_result
from .run import file_sha
from .sources import sha

PRODUCER = "llm_reused_identical_context_v1"
CONFIG_KEYS = ("base_url", "model", "temperature", "thinking", "response_format")


def context_body(result):
    if "resolution" in result:
        resolution = result["resolution"]
        if resolution["decision"] != "context" or resolution["promotions"]:
            raise ValueError("REUSE_REQUIRES_CONTEXT_RESULT")
        return resolution["fallback"]
    return result["rewrite"]


class Donor:
    def __init__(self, path, recipient_config):
        self.path = Path(path).resolve()
        self.config = json.loads((self.path / "config.json").read_text())
        if any(self.config[k] != recipient_config[k] for k in CONFIG_KEYS):
            raise ValueError("REUSE_SERVICE_OR_MODEL_CONFIGURATION_CHANGED")
        self.config_sha = file_sha(self.path / "config.json")
        self.driver_sha = file_sha(Path(__file__).with_name("llm_backfill.py"))
        self.db = sqlite3.connect("file:" + str(self.path / "journal.sqlite") + "?mode=ro", uri=True)
        if self.db.execute("SELECT COUNT(*) FROM jobs WHERE status!='done'").fetchone()[0]:
            self.db.close()
            raise ValueError("REUSE_DONOR_INCOMPLETE")
        self.attempt_counts = dict(self.db.execute("SELECT job_id,COUNT(*) FROM attempts GROUP BY job_id"))
        self.single_attempts = {
            rid: (aid, driver, error)
            for rid, aid, driver, error in self.db.execute("SELECT job_id,id,driver_sha256,error FROM attempts")
            if self.attempt_counts.get(rid) == 1
        }

    def close(self):
        self.db.close()

    def get(self, rid):
        row = self.db.execute("SELECT request,result,decision FROM jobs WHERE id=? AND status='done'", (rid,)).fetchone()
        if row is None:
            raise ValueError("REUSE_DONOR_REQUEST_MISSING")
        request, result, decision = map(unpack, row)
        if (result.get("producer") != "llm_segment_context_v2"
                or requires_candidate_review(request) or decision["decision"] != "context"
                or self.attempt_counts.get(rid) != 1):
            return None
        aid, driver, error = self.single_attempts[rid]
        if error or driver != self.driver_sha:
            return None
        response_blob = self.db.execute("SELECT response FROM attempts WHERE id=?", (aid,)).fetchone()[0]
        if not response_blob:
            return None
        response = unpack(response_blob)
        choice = response["choices"][0]
        body = context_body(result)
        if choice["finish_reason"] != "stop" or model_json(choice["message"]["content"]) != body:
            raise ValueError("REUSE_RESPONSE_RESULT_MISMATCH")
        return request, result, body, aid, sha(messages_for(request))

    def index(self):
        index = {}
        candidates = self.db.execute("SELECT id FROM jobs WHERE status='done' ORDER BY seq").fetchall()
        for (rid,) in candidates:
            candidate = self.get(rid)
            if candidate:
                index.setdefault(candidate[4], rid)
        return index

    def adapt(self, request, rid):
        if requires_candidate_review(request):
            raise ValueError("CANDIDATE_REVIEW_CANNOT_REUSE_CONTEXT")
        candidate = self.get(rid)
        if candidate is None:
            raise ValueError("REUSE_DONOR_RESULT_INELIGIBLE")
        old_request, old_result, body, aid, prompt_sha = candidate
        if sha(messages_for(request)) != prompt_sha or messages_for(request) != messages_for(old_request):
            raise ValueError("REUSE_PROMPT_MISMATCH")
        result = envelope(request, body, PRODUCER)
        result["reuse"] = {"donor_run": str(self.path), "donor_request_id": rid,
                           "donor_attempt_id": aid, "donor_result_sha256": sha(old_result),
                           "donor_config_sha256": self.config_sha, "driver_sha256": self.driver_sha,
                           "prompt_sha256": prompt_sha, "new_http_call": False}
        return result

    def verify(self, request, result):
        expected = self.adapt(request, result["reuse"]["donor_request_id"])
        if result != expected:
            raise ValueError("REUSE_PROVENANCE_OR_RESULT_MISMATCH")


def prime(db, config, profile, donor_path):
    donor = Donor(donor_path, config)
    counts = Counter()
    try:
        index = donor.index()
        print(json.dumps({"reusable_distinct_prompts": len(index)}), flush=True)
        pending_ids = db.execute("SELECT id FROM jobs WHERE status='pending' ORDER BY seq").fetchall()
        for (rid,) in pending_ids:
            blob = db.execute("SELECT request FROM jobs WHERE id=?", (rid,)).fetchone()[0]
            request = unpack(blob)
            if requires_candidate_review(request):
                counts["candidate_reviews_left_for_llm"] += 1
                continue
            old_rid = index.get(sha(messages_for(request)))
            if old_rid is None:
                counts["other_pending"] += 1
                continue
            result = donor.adapt(request, old_rid)
            _, masks, decision = evaluate_result(request, result, profile)
            if decision["decision"] != "context" or any(masks):
                raise ValueError("REUSE_CANNOT_RESTORE_SUPERVISION")
            db.execute("UPDATE jobs SET status='done',result=?,decision=?,error=NULL WHERE id=?",
                       (pack(result), pack(decision), rid))
            counts["reused_identical_llm_contexts"] += 1
            if counts["reused_identical_llm_contexts"] % 1000 == 0:
                db.commit()
                print(json.dumps(dict(counts)), flush=True)
        db.commit()
    finally:
        donor.close()
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--target", required=True, choices=["openhands_sdk", "sweagent"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db, config = prepare_db(args.stage, args.target, args.out, args.base_url, args.model, args.disable_thinking)
        try:
            profile = json.loads((args.stage / "profiles.json").read_text())[args.target]
            result = prime(db, config, profile, args.donor)
            export_results(db, args.out)
            (args.out / "reuse_summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            db.close()


if __name__ == "__main__":
    main()
