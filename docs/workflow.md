# Conversion and backfill

## Compile and prepare

    python -m trace_bridge --source data/source.parquet --out runs/compiled --targets openhands_sdk sweagent
    python -m trace_bridge.export_training prepare --source data/source.parquet --compiled runs/compiled --out runs/stage

Output directories for compilation, preparation, and finalization must not already
exist. Input hashes prevent accidentally combining different datasets or plans.
The v3 exporter rejects v2 compilation/staging manifests.

The preparation step writes a source archive, staged dialogues, ready Parquet
subsets, and one rewrite queue per target. It does not call an LLM.

## Optional LLM backfill

Use an API compatible with Chat Completions and JSON-object responses. Supply the
key through TRACE_BRIDGE_API_KEY or --api-key-file. The base URL should include the
API prefix required by the provider; the client appends /chat/completions.

    python -m trace_bridge.llm_backfill --stage runs/stage --target openhands_sdk --out runs/oh-backfill --base-url https://api.example.org/v1 --model YOUR_MODEL --concurrency 32

The client journals requests, responses, and accepted results in SQLite. Repeating
the command with the same configuration resumes pending work. Use --retry-failed
to retry failed requests. The --limit option limits requests scheduled during that
invocation.

The provider-specific thinking=disabled field is omitted by default. Use
--disable-thinking only with a provider that supports it.

Two routes are supported:

- context_rewrite summarizes recorded source interactions as mask-0 context.
- candidate_resolution reviews actions recompiled from source evidence. An LLM
  cannot remove unresolved compiler requirements or introduce arbitrary actions.

Some missing-response submission markers are handled without an LLM. Identical
source-context prompts can be reused across targets with
python -m trace_bridge.reuse_context; target-specific action reviews are separate.

To export prompts for an external client instead:

    python -m trace_bridge.export_training prompts --queue runs/stage/openhands_sdk.rewrite_requests.jsonl.gz --out runs/oh-prompts.jsonl.gz

Each prompt record contains request_id, input_sha256, messages, response_schema,
and result_field. Preserve request_id/input_sha256 in the result envelope and put
the returned object under result_field.

## Finalize

    python -m trace_bridge.export_training finalize --stage runs/stage --results runs/oh-backfill/results.jsonl.gz --targets openhands_sdk --out runs/oh-final

Run backfill/finalization separately for sweagent when needed. Rows with outstanding
requests remain in pending.rows.jsonl.gz. Check the final manifest's row counts
before treating the Parquet output as a complete dataset.

For a dataset with no outstanding requests, --results may be omitted.

## Audit commands

These optional commands inspect stored artifacts without executing trajectories:

    python -m trace_bridge.validate_training runs/stage --out runs/stage-audit.json
    python -m trace_bridge.validate_backfill --stage runs/stage --calls runs/oh-backfill --final runs/oh-final --out runs/oh-audit.json

This review release has not rerun these audits, the test suite, or the full dataset
conversion. The existing v2 benchmark artifacts should be evaluated separately
from newly generated v3 data.
