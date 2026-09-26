# TraceBridge

**Interaction-Aware Trajectory Rewriting across Harnesses for Coding Agent Training**

TraceBridge rewrites existing coding-agent trajectories for supervised fine-tuning
(SFT) under a target harness, considering source actions together with their context
and feedback:

- **Action adaptation:** compile supported operations into target calls using tool
  contracts and recorded evidence.
- **Context preservation:** retain unmapped interactions as summaries with complete
  source fragments and zero direct training loss.

Sources: SWE-agent, mini-swe-agent, OpenHands, Claude Code, OpenCode, and Codex-format.
Targets: OpenHands SDK (`openhands_sdk`) and SWE-agent (`sweagent`).

This review release (v0.3.0) provides the converter, training exporter, LLM client,
and tests. Datasets, experiment logs, and model checkpoints are not included.

## Install

Use Python 3.11+ on Linux.

```bash
python -m pip install -e .
```

Conversion runs offline without a running agent harness. Generated commands assume
Bash and may use GNU sed, ripgrep, and coreutils.

## How it works

1. Parse source messages, tool definitions, and loss masks into an operation IR.
2. Compile target calls, checking operation conditions and feedback bindings.
3. Review eligible candidates with an LLM and program checks; rewrite remaining
   interactions as context.
4. Merge outputs in source order and export SFT Parquet with supervision masks
   and provenance.

Converted actions inherit source masks; tool observations and retained context
have zero loss. Observations come from recorded source feedback. Downstream
trainers must honor `message_loss_mask`.

## Quick start

Create six synthetic records and prepare training exports:

```bash
python examples/make_demo.py --out data/demo.parquet
python -m trace_bridge --source data/demo.parquet --out runs/demo-compiled
python -m trace_bridge.export_training prepare --source data/demo.parquet --compiled runs/demo-compiled --out runs/demo-stage
```

Each target's `ready.parquet` contains records with no pending rewrite requests.
Staged files retain all input rows; complete pending requests before exporting the
full dataset. The demo supplies observations without executing source commands.

See [data format](docs/data-format.md) for input fields and masks, and
[workflow](docs/workflow.md) for LLM processing and final export.

## Version notes

Version 0.3.0 changes command generation from the v2 experimental artifacts. Use
fresh compile/staging directories. The paper's measurements describe the existing
v2 exports, which have not been regenerated.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest tests
```

Tests are included but were not rerun for this release. Optional editor integration
tests require `SWE_AGENT_TOOLS` to point to a SWE-agent tools directory.

## License

Project code uses the [MIT license](LICENSE). Bundled tool definitions retain their
[third-party notices](THIRD_PARTY_NOTICES.md).
