"""Create synthetic source trajectories without executing their commands."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trace_bridge.sources import MINI_SCOPE


def records():
    sources = {
        "SWE-agent": "bash",
        "mini-swe-agent": "bash",
        "OpenHands": "execute_bash",
        "Claude Code": "Bash",
        "OpenCode": "bash",
        "Codex-format": "shell",
    }
    for index, (source, tool) in enumerate(sources.items()):
        description = "Execute a command with bash -lc." if source == "Codex-format" else "Execute a shell command."
        definition = {"name": tool, "description": description, "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
        tools = [definition if source == "OpenCode" else {"type": "function", "function": definition}]
        messages = [
            {"role": "system", "content": MINI_SCOPE if source == "mini-swe-agent" else "Recorded source tool session."},
            {"role": "user", "content": "<uploaded_files>\n/repo\n</uploaded_files>\nPrint hello."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "demo-call", "type": "function",
                "function": {"name": tool, "arguments": json.dumps({"command": "cat <<'EOF'\nhello\nEOF"})}}]},
            {"role": "tool", "tool_call_id": "demo-call", "content": "hello\n"},
            {"role": "assistant", "content": "The recorded command printed hello."},
        ]
        yield {"id": "demo-" + str(index), "source_scaffold": source, "split": "train",
               "source_dataset": "synthetic-demo", "acceptance_basis": "synthetic example",
               "messages_json": json.dumps(messages), "tools_json": json.dumps(tools),
               "message_loss_mask": [0, 0, 1, 0, 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(records())), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
