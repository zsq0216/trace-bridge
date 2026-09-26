"""Operation, evidence, and target-plan data structures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Kind = Literal["apply_patch", "write_file", "replace_text", "find_files", "delete_file",
               "read_file", "run_command", "search_text", "editor_action", "finish",
               "think", "unsupported"]


@dataclass
class Operation:
    kind: Kind
    parameters: dict[str, Any]
    contracts: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceCall:
    call_id: str
    source: str
    source_tool: str
    message_index: int
    call_index: int
    operation: Operation
    original_call: dict[str, Any]
    definition_sha256: str | None = None
    observation_indices: list[int] = field(default_factory=list)
    supervised: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class TargetAction:
    tool: str
    arguments: dict[str, Any]
    implementation: Literal["native", "simple_shell", "source_shell"]

    def to_wire(self, call_id: str):
        import json
        return {"id": call_id, "type": "function", "function": {
            "name": self.tool, "arguments": json.dumps(self.arguments, ensure_ascii=False)}}


@dataclass
class Plan:
    target: str
    actions: list[TargetAction] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    reason: str | None = None
    notes: list[str] = field(default_factory=list)
    preimage_sha256: dict[str, str] = field(default_factory=dict)
    verified_properties: list[str] = field(default_factory=list)

    @property
    def status(self):
        return "unsupported" if self.reason else "conditional" if self.requirements else "ready"

    def to_dict(self):
        return {**asdict(self), "status": self.status, "execution": "sequential_stop_on_error",
                "validation_scope": "static_compilation",
                "observation_policy": "source_references_only", "training_ready": False}

    def wire_calls(self, source_call_id: str):
        if self.status != "ready":
            raise ValueError("PLAN_NOT_READY:" + (self.reason or ",".join(self.requirements)))
        return [a.to_wire(source_call_id if len(self.actions) == 1 else source_call_id + ":" + str(i))
                for i, a in enumerate(self.actions)]


@dataclass(frozen=True)
class GlobPolicy:
    """File selection and result ordering for glob operations."""
    engine: Literal["ripgrep"] = "ripgrep"
    hidden: bool = False
    respect_ignore: bool = True
    follow_symlinks: bool = False
    order: Literal["mtime_desc", "mtime_asc", "path"] = "mtime_desc"
    empty_matches: Literal["success_empty", "error"] = "success_empty"


@dataclass
class CompileContext:
    """Pre-action evidence. files contains complete UTF-8 snapshots; None denotes an absent path."""
    cwd: str | None = None
    files: dict[str, str | None] = field(default_factory=dict)
    source_succeeded: bool = False
    source_patch_succeeded: bool = False
    read_paths: set[str] = field(default_factory=set)
    directories: set[str] = field(default_factory=set)
    glob_policy: GlobPolicy | None = None
    python_feature_version: tuple[int, int] | None = None
