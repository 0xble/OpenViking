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

## Active patch register

The complete active provenance is the 189 exact stable subjects in the required
support file (full SHA). `chore(fork)`/`sync` entries are reconciliation provenance, and revert subjects
record deliberate retirement of only their named change. `3e9ff0cb8849bdfe32315ee269dcbb103a6f34bf`
is source-associated with merged `volcengine/OpenViking#1592`
(`https://github.com/volcengine/OpenViking/pull/1592`, merged 2026-05-08): it ports
the upstream reindex API and retains fork-side consolidator/maintenance-router wiring.
Treat that as an association, not upstream-equivalence or retirement evidence; verify
the released source plus the OV-002/OV-005 surfaces before retiring the retained wiring.
Other fork-range items have no recorded upstream issue/PR after checked 2026-09-09.
Every active group is verified by its listed test surface plus `bin/check` and
`bin/ci`; rollback is a source-level revert of the group commits after checking
dependencies; retire only when a released upstream implementation passes the same
surface tests.

| ID | Active invariant and surfaces | Provenance scope |
| --- | --- | --- |
| OV-001 | Retrieval/filter and resource behavior remains source/time-aware; `openviking/retrieve/`, `openviking/utils/search_filters.py`, `tests/retrieve/`, `tests/unit/test_search_filters.py`. | Subjects beginning `feat(retrieval)`, `fix(retrieval)`, `fix(search)`, `refactor(search)`, `fix(retrieve)`, `fix(fs)`, `fix(resources)`, `feat(resources)`, `feat(metadata)`. |
| OV-002 | Memory, maintenance, tenant, redo, and session behavior remains isolated and auditable; `openviking/maintenance/`, `openviking/session/`, `openviking/storage/`, `tests/unit/maintenance/`, `tests/session/`, `tests/server/test_api_maintenance.py`. | Subjects beginning `feat(memory)`, `fix(memory)`, `fix(session`, `fix(maintenance)`, `fix(storage`, `fix(server`, `fix(lock)`. |
| OV-003 | Agent plugin/runtime contracts remain internally consistent; `examples/*memory-plugin/`, `examples/openclaw-plugin/`, `.claude-plugin/`, their tests and manifests. | Subjects containing `plugin`, `openclaw`, `codex-memory`, `claude-code-memory`, `marketplace`, or `mcp`. |
| OV-004 | CLI, packaging, build and release fork behavior remains buildable without changing distribution policy; `crates/ov_cli/`, `openviking_cli/`, `build_support/`, `bin/`, `Cargo.lock`, `uv.lock`. | Subjects containing `cli`, `build`, `release`, `lockfiles`, `fork version`, or `codesign`. |
| OV-005 | Model, retry, safety, parsing, observability and API compatibility fixes remain present; `openviking/models/`, `openviking/utils/`, `openviking/server/`, `tests/models/`, `tests/server/`, `tests/unit/`. | Remaining non-reconciliation subjects in the complete provenance support file. |

## Required maintenance support

This root is the sole enrollment and scheduling unit. The support file extends
its patch register without changing the baseline, adoption rules, or overall proof.

| Responsibility | Read when | Support file |
| --- | --- | --- |
| Exact commit attribution for OV-001 through OV-005 and reconciliation records | Every maintenance run, including no-change reviews, and whenever a patch changes or retires | [Fork provenance](maintenance/fork-provenance.md) |

## Update

Each run resolves upstream's live default branch before fetching it, then fetches
`origin` and `upstream` separately, reconciles latest `upstream/$UPSTREAM_DEFAULT`
while preserving only this register, runs the applicable focused tests/build, then
publishes to `origin/main` or reports `Blocked` with stage, refs, and evidence.
Immediately before `Updated` or `Already current`, resolve and fetch upstream's live
default branch again and require `git rev-list --left-right --count
"upstream/$UPSTREAM_DEFAULT...main"` to have upstream-only count `0`; fetch `origin`
and require `origin/main == HEAD`. Any addition, change, retirement, or ambiguous
provenance updates this register before publication. Current source reconciliation is
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
complete provenance support file. Do not install, activate, deploy, or otherwise
alter a runtime. Sync success additionally
requires the final fresh-fetch zero count and owned-remote SHA parity above.
