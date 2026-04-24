# OpenViking Plugin Parity Contract

OpenViking integrations should stay aligned at the behavior level, not by forcing
every host to expose identical commands or share one adapter runtime.

## Required Behavior

- Send tenant headers when a real account and user are configured.
- Do not treat the literal `default` tenant as a real tenant header.
- Search user and agent memory scopes by default for explicit recall.
- Keep explicit `target_uri` recall scoped to that target only.
- Return partial recall results when one default scope fails and another succeeds.
- Surface a short partial-failure note instead of silently dropping failed scopes.
- Filter memory recall to leaf memory nodes when the server provides `level`.
- Sort and limit recall results after combining scopes.
- Support explicit verbatim memory writes separately from extractor-based stores.
- Use the session commit path for extractor-based stores so extraction is durable.
- Preserve host-specific lifecycle, hooks, and tool names where the host requires it.

## Non-Goals

- Do not introduce a shared TypeScript plugin core before the adapter contracts are
  stable.
- Do not force Claude Code, Codex, and OpenClaw to expose identical UI surfaces.
- Do not hide host-specific constraints behind compatibility shims.

## Current Adapter Shape

- OpenClaw owns the richest session lifecycle: context assembly, after-turn capture,
  compaction, resource import, and diagnostics.
- Claude Code owns hook-based auto-recall and auto-capture plus MCP tools for
  explicit memory operations.
- Codex owns MCP tools for explicit memory operations and should stay minimal.

When behavior differs, prefer a small adapter-local helper with tests. Promote a
shared library only after at least two adapters need the same non-trivial logic and
the host APIs have stopped moving.
