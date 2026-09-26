# Trace Bridge

Trace Bridge translates recorded coding-agent trajectories into training dialogues
for OpenHands SDK and SWE-agent. Source calls are parsed into an operation IR,
compiled against a fixed target tool interface, and exported with message-level
supervision masks.

This is the review release, version 0.3.0. It includes the converter, training
exporter, optional LLM backfill client, and existing tests. Datasets, experiment
logs, service credentials, and model checkpoints are not included.

## Install

Use Python 3.11 or later on Linux.

    python -m pip install -e .

Compilation does not require a running agent harness. Generated shell operations
assume Bash; file operations may also use GNU sed, ripgrep, and coreutils. Tool
schemas are included in the package. See [third-party notices](THIRD_PARTY_NOTICES.md)
for their provenance.

## Pipeline

    Source messages, tool definitions, and loss masks
        -> source parsers
        -> operation IR
        -> target plans
        -> staged dialogues and rewrite requests
        -> optional LLM review or context rewriting
        -> SFT Parquet

Supported source identifiers are SWE-agent, mini-swe-agent, OpenHands, Claude Code,
OpenCode, and Codex-format. Targets are openhands_sdk and sweagent.

The IR represents patches, writes, replacements, reads, searches, glob operations,
deletions, shell commands, editor actions, and completion/thought actions. Unsupported
source interactions retain their original records for context rewriting.

## Small example

The example script creates six synthetic records with supplied observations. It
does not run the commands in those records.

    python examples/make_demo.py --out data/demo.parquet
    python -m trace_bridge --source data/demo.parquet --out runs/demo-compiled
    python -m trace_bridge.export_training prepare --source data/demo.parquet --compiled runs/demo-compiled --out runs/demo-stage

The two target ready.parquet files under runs/demo-stage contain rows with no
outstanding rewrite requests. The staged files retain every input row. On a real
dataset, ready.parquet may contain only a subset until backfill is complete.

See [the data format](docs/data-format.md) for the input schema and loss-mask
handling, and [the workflow](docs/workflow.md) for backfill and final export.

## Operation compilation

    from trace_bridge.ir import CompileContext, Operation
    from trace_bridge.targets import compile_operation

    operation = Operation(
        "write_file",
        {"path": "/repo/example.py", "content": "value = 1\n", "mode": "overwrite"},
    )
    plan = compile_operation(operation, "sweagent", CompileContext())
    calls = plan.wire_calls("source-call-1")

A ready plan can be serialized into target calls. A conditional plan has candidate
actions with unresolved requirements; an unsupported plan has no executable action.
The exporter handles unresolved interactions through the rewrite queue.

## Shell representation

Per-call non-login commands use a subshell with an unescaped body:

    (
    cd -- /repo || exit $?
    cat > example.py <<'EOF'
    value = 1
    EOF
    )

Codex login commands retain bash -lc. A quoted heredoc loads the command into a
subshell-local variable; the launcher passes that variable as one argument. This
preserves embedded quotes, trailing newlines, and the command's original stdin.
The heredoc delimiter is selected to avoid collisions with the command body.

Explicit workdir changes are confined to the current call. File writes use
literal heredocs or printf; replace-all uses GNU sed with staged output; deletions
use test -f followed by rm.

## Scope

The converter preserves recorded source observations and their provenance. It does
not execute trajectories or synthesize target observations. A ready plan is a
compilation result, not a task-success label.

Unknown timeout lifecycle, glob policies, patch preimages, and mutation ordering
remain requirements. All input rows retain a train split, with the original split
stored separately. Downstream trainers must honor message_loss_mask.

This release changes command generation relative to the v2 experimental artifacts.
Use a fresh compile/staging directory for v3. Existing v2 Parquet files have not
been regenerated, and the earlier experimental measurements describe those files.

## Tests

The existing tests are included for reviewers. They were not rerun for this release.

    python -m pip install -e '.[test]'
    python -m pytest tests

Optional SWE-agent editor integration tests use the SWE_AGENT_TOOLS environment
variable, pointing to a SWE-agent tools directory. They are skipped if the editor
is unavailable.

## License

Project code is distributed under the [MIT license](LICENSE). Included upstream
tool definitions retain their [third-party notices](THIRD_PARTY_NOTICES.md).
