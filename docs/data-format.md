# Data format

## Input

Each Parquet row describes one recorded trajectory. Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| id | string | Unique trajectory identifier |
| source_scaffold | string | One of the six supported source identifiers |
| split | string | Original dataset split |
| messages_json | string | JSON-encoded message list |
| tools_json | string | JSON-encoded source tool definitions |
| message_loss_mask | list of integers | One 0/1 value per source message |

Optional metadata includes task_group_key, source_dataset, source_revision,
teacher, acceptance_basis, and reference_tokens.

Messages use role/content fields. Assistant tool calls contain id, type=function,
and function.name/function.arguments. The arguments field may contain an encoded
JSON object or a parsed object. Tool responses refer to the source call through
tool_call_id. The exporter accepts a response binding only when it is unique.

Source tools normally use a function wrapper around name/description/parameters;
OpenCode's flat definitions are also accepted. Definitions are part of the input:
their declared defaults and contracts affect compilation.

Source operating context may identify a repository with an uploaded_files block
or a Cloned path declaration. Relative paths require a known working directory.
The mini-swe-agent parser recognizes its declaration that each action runs in a
new subshell. See examples/make_demo.py for a complete synthetic input.

## IR and plans

The compile command produces:

- ir.jsonl.gz: parsed operations, source calls, and observation indices.
- TARGET.plans.jsonl.gz: target actions, requirements, and source references.
- source_tool_definitions.json: tool definitions indexed by content hash.
- manifest.json: input/code/output hashes and coverage counts.
- examples.json: selected source operations and plans.

Arguments remain objects in IR and plans. TargetAction.to_wire serializes them
to JSON strings at the tool-call boundary.

## Training output

Final Parquet rows contain messages_json, tools_json, message_loss_mask, source
metadata, and bridge_provenance_json. The latter maps target spans to source
message indices and records rewrite or promotion decisions.

Supervision is assigned as follows:

- Accepted target actions inherit the source assistant's loss mask.
- Tool responses and preserved historical context have mask 0.
- Candidate promotions require matching program-derived actions and evidence.
- Final source text may become the target completion tool.

Context summaries include the original source evidence. Preserving an interaction
does not make it a supervised target action. Trainers must map message_loss_mask
to token labels; flattening all assistant messages into supervised text changes
the intended training objective.

All exported rows use split=train and retain the input label as source_split.
The package does not create a validation split or select a model chat template.
