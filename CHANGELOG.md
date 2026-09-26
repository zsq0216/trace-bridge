# Changes

## 0.3.0 — review release

- Replace whole-command quoting for per-call shell actions with literal-body
  subshells. Login actions retain bash -lc through a quoted-heredoc loader.
- Preserve the known persistent cwd after a call-local workdir override in both
  initial compilation and evidence tracking.
- Reject stale compilation and staging versions during export.
- Omit the provider-specific thinking parameter by default; add --disable-thinking.
- Support TRACE_BRIDGE_API_KEY in addition to a key file.
- Add standalone packaging, English documentation, and a synthetic example.
- Replace local checkout paths with portable schema provenance and an optional
  SWE_AGENT_TOOLS setting.
- Shorten code comments and docstrings; retain implementation notes for quoting,
  symlink traversal, and in-place file updates.

No full conversion, trajectory replay, or test-suite run accompanies this release.
Previous experimental datasets and their reported measurements are unchanged.
