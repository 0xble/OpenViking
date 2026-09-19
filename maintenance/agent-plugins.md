# OV-003: Agent plugins

Part of the [root maintenance contract](../MAINTENANCE.md). Read this unit for
changes to its behavior, source surfaces, or upstream equivalents. Every full
maintenance run evaluates this unit. Before adapting or retiring shared commits,
load [source reconciliation](source-reconciliation.md).

## Required behavior and proof surfaces

Agent plugin/runtime contracts remain internally consistent.

`examples/*memory-plugin/`, `examples/openclaw-plugin/`, `.claude-plugin/`, their tests and manifests.

## Provenance and adoption

The accepted root observation owns the fork and upstream baselines. The entries
below retain the prior register's exact SHA/subject pairs, in their original order.
Routing uses the original scope rule: Subjects containing `plugin`, `openclaw`, `codex-memory`, `claude-code-memory`, `marketplace`, or `mcp`.
Shared reconciliation records and overlapping subject matches remain with source
reconciliation. Subject routing is an inventory aid, not proof of exclusive code
ownership. Inspect the source diff and dependencies before acting.

Other fork-range items have no recorded upstream issue/PR after checked 2026-09-09.
These are retained fork records, not proof of current upstream absence or runtime
adoption. Compare the candidate upstream source with the required behavior before
adopting or retiring changes, and update this unit with material patch decisions.

## Verification, rollback, and retirement

Run this unit's listed test surfaces and the root's `bin/check` and `bin/ci` gates.
For surfaces without a specific focused command above, inspect their existing
repository tests before selecting proof. This inventory does not close coverage
gaps. Roll back by source-level revert of the group commits only after checking
dependencies. Retire only when released upstream passes the same surface tests.
Publication and installation remain subject to the root's separate boundaries.

## Exact retained entries

```text
36356ab293ec5f42c5867336fef7acb4e7c9150f fix(plugins): tighten resource recall and session routing
a4509d8fcf58cadae9492c5f74bdce52c0b31410 fix(openclaw): use readable memory write output
a4a35a705f3fb415a0b03ae15a28a33f9188d2ce feat(plugins): add resource recall tools
b3115673efe7c85e086b2f8f9b168c26d11a49cc fix(openclaw): quiet expected recall fallbacks
1fd559a099687670d2123b1f16b08c2b19490ee5 fix(plugins): tighten recall and budget helpers
3f6092ed4915c276dde876ce94a65dd51afd0393 fix(plugins): align OpenViking memory adapters
7a81f640e2e91a6f19ddb6e3b0bfa13964cf8890 fix(openclaw): harden openviking recall timeouts
5e3818dcc033349eec17f90a4de3d329cc843645 fix(openclaw-plugin): restore createHash import on OpenVikingClient
0e1b9bb14607f27de79f80dd857569bb43204af7 fix(openclaw-plugin): restore spaceCache field on OpenVikingClient
35a8a653c0bede3d6555989e26a2ca9fb72dcb71 fix(openclaw-plugin): align OpenVikingClient call with 10-arg signature
6cf1c67ca24e19f07218d91a9d291c75d58e5303 feat(plugins): add session_recall tool across claude-code, codex, openclaw
08b90bae25a7e0e4354284f4cfacf091509e2e58 chore(openclaw-plugin): bump pluginVersion to 2026.4.21
d007fc4fcf0991345738c5c3b6a3c68e3bed9858 ci(openclaw-plugin): lint install-manifest against runtime imports
1a216e333940dfb65ffef2ca60468dc07a51f449 fix(openclaw-plugin): include recall modules in install-manifest
9003cbec1c26c6aacaa688575c61a3174a003d1e fix(openclaw-plugin): default recall to assemble and bound afterTurn
1c6f6b568f78f17ed82d75471f63c8cf68dd045c refactor(codex-plugin): rename openviking_* tools to memory_* for consistency
b67cbb1ecdc0a89894b7a0c006ecf9f7bf7adfa8 fix(plugins): memory_write review fixes — unwrap request, tighten URI checks
1b316ad50846342ed37b7a800edd58a3f2fd06b6 feat(plugins): add memory_write tool across 4 memory plugins
56b3f20c1a1d2c2cc131556a35ef7c251ac37f38 feat(openclaw-plugin): categorize commit errors in compact reason
35d8cdcb31c25824e2a0a511642d63b36ae784d4 ci: add fork regression gate for openclaw-plugin contract tests
1a92f7fab43744866f4396af160d26fe787ed155 fix(openclaw-plugin): afterTurn uses structured parts with tool fidelity
d0d2b15bb0f9682e72fc225b419c32506cdcc103 feat(openclaw-plugin): self-heal dormant sessions in compactOVSession
85271cd09807b0774bd941c46f09ed1a76d96ac1 fix(openclaw-plugin): send addSessionMessage text as parts array
cccc67b9567f8723d4c53ed7a2640a6b2c75f5c5 fix(claude-code-memory-plugin): drop unresolved ${VAR} env passthrough in .mcp.json
3cbab082cb9e2e6c0166e49fd3e9cecded664d01 feat(marketplace): add .claude-plugin/marketplace.json for fork distribution
17289c929089290516d3fac5c7a30532cbb85cec fix(claude-code-memory-plugin): exec ready runtime when CLAUDE_PLUGIN_ROOT is missing
8ef771d24402a2ec701af51175827c7025ad5e5f fix(openclaw-plugin): route memory to configured tenant
4584cc07e064a77095d533919aa61b84c7a8da30 fix(openclaw-plugin): avoid stale json5 config fallback
8b338d2d3d490c8285882bb9398b3da5a565410d fix(openclaw-plugin): bound adaptive recall caches
a05b605812641dee69f086fcacdb78c31f02f900 fix(openclaw-plugin): adapt memory recall latency
7dc9699c40248dc1e82025d2b4eff41a2654cd99 fix(openclaw-plugin): harden openviking session reliability
251f5669351aa78023a863a82a580413a53cc011 fix(openclaw-plugin): detect active openclaw config file
633bfc49376ffa9fd04b89d6e7fc066c0e88bb70 fix(openclaw-plugin): move recall into assemble by default
698e64cf02ed13fb90552da6d8961af6e3a08e06 fix(codex-memory-plugin): wait for delete consistency
50c339d87e55695beba50e5d5d64ef30f539c70f feat(mcp): proxy example to shared http backend
```
