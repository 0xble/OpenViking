# Maintenance

## Background

Maintained fork `0xble/OpenViking` of `volcengine/OpenViking`; canonical source
checkout `/Users/brianle/OpenViking-pr-repair`, maintained branch `main`, owned
remote `origin`, source remote `upstream`. The named upstream branch means upstream's
live default branch, resolved on every run before fetching; it is not statically pinned
to `main`. Accepted observation on 2026-09-09:
`origin/main` `a8998823173079ca691e38e8728a3eddde234dbc`; upstream's then-current
default branch `98f7dfe0e39333ee9d11f83c563d59fa3a3b6a71` (1223 upstream-only,
189 fork-only). This is a maintained divergent fork, not a contribution fork.

## Preserve

- Publish only to `origin`; upstream is source authority and is never a push target.
- Preserve the fork-only behavior recorded below across retrieval, memory/session,
  server, CLI, plugin, build, and safety surfaces; current source and its named
  tests decide conflicts, not historical patch order.
- Source reconciliation, publication, installation, and runtime activation are
  separate stages; this contract authorizes none of the latter two.

## Maintenance units

This root is the sole enrollment and scheduling unit. A scoped change loads its
unit and the dependencies in its row. A full maintenance run accounts for every
row, including no-change reviews. Detailed provenance and proof stay with each unit.

| Unit | Purpose / required behavior | Load when | Contract |
| --- | --- | --- | --- |
| OV-001 | Source/time-aware retrieval and resource behavior | Every full run or retrieval/resource changes; source reconciliation before changing shared commits | [Retrieval](maintenance/retrieval.md) |
| OV-002 | Isolated, auditable memory, tenant, redo, and session behavior | Every full run or memory/session changes; OV-005 for reindex integration; source reconciliation for shared commits | [Memory and sessions](maintenance/memory-sessions.md) |
| OV-003 | Consistent agent plugin/runtime contracts | Every full run or plugin changes; source reconciliation for shared commits | [Agent plugins](maintenance/agent-plugins.md) |
| OV-004 | Buildable CLI and fork distribution behavior | Every full run or CLI/build changes; source reconciliation for shared commits | [CLI and distribution](maintenance/cli-distribution.md) |
| OV-005 | Model, retry, safety, parsing, observability, and API compatibility | Every full run or these behaviors change; OV-002 for reindex integration; source reconciliation for shared commits | [Model and API safety](maintenance/model-api-safety.md) |
| Source reconciliation | Cross-unit merge compatibility and deliberate retirement boundaries | Every full run or shared-commit changes; load affected feature units before adaptation, including OV-002 and OV-004 for the ambiguous lock-release record | [Source reconciliation](maintenance/source-reconciliation.md) |

## Update

Each run resolves upstream's live default branch before fetching it, then fetches
`origin` and `upstream` separately, reconciles latest `upstream/$UPSTREAM_DEFAULT`
while preserving only this register, runs the applicable focused tests/build, then
publishes to `origin/main` or reports `Blocked` with stage, refs, and evidence.
Immediately before `Updated` or `Already current`, resolve and fetch upstream's live
default branch again and require `git rev-list --left-right --count
"upstream/$UPSTREAM_DEFAULT...main"` to have upstream-only count `0`; fetch `origin`
and require `origin/main == HEAD`. Any addition, change, retirement, or ambiguous
provenance updates the responsible unit and root routing before publication. Current source reconciliation is
Blocked by the recorded 1223 upstream-only commits; this docs-only enrollment
must not be called a source sync.

## Verify

```text
bin/check
bin/ci
uv run python -m pytest tests/unit/maintenance tests/retrieve tests/server/test_api_maintenance.py
make build-cli
UPSTREAM_DEFAULT="$(git ls-remote --symref upstream HEAD | awk '/^ref:/ {sub("refs/heads/", "", $2); print $2; exit}')"
test -n "$UPSTREAM_DEFAULT"
git fetch --prune upstream "refs/heads/$UPSTREAM_DEFAULT:refs/remotes/upstream/$UPSTREAM_DEFAULT"
git diff --check "upstream/$UPSTREAM_DEFAULT...HEAD"
git rev-list --left-right --count "upstream/$UPSTREAM_DEFAULT...main"
```

For a docs-only change, run `git diff --check` and inspect the root register and
affected units and their exact provenance entries. Do not install, activate, deploy, or otherwise
alter a runtime. Sync success additionally
requires the final fresh-fetch zero count and owned-remote SHA parity above.
