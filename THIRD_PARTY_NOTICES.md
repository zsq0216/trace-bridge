# Third-party notices

The target tool descriptions and schemas in trace_bridge/training_profiles.json
were adapted from local snapshots of:

- SWE-agent v1.1.0: tool definitions under tools/edit_anthropic,
  tools/review_on_submit_m, and config/default.yaml.
- OpenHands SDK: terminal/file_editor tool definitions and finish/think builtins.
  The local snapshot was identified as version 1.40.1 in the interface audit.

The original file fingerprints are retained in the JSON file with
repository-relative paths. These projects use the MIT license; their original
notices are included in LICENSES/swe-agent.txt and LICENSES/openhands-sdk.txt.

The converter does not import or vendor the ADP implementation. ADP-based
experimental baselines and their generated datasets are outside this release.
