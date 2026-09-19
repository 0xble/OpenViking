# Source reconciliation

Part of the [root maintenance contract](../MAINTENANCE.md). Read every full
maintenance run and before changing shared reconciliation commits or resolving
ambiguous attribution. This unit owns merge compatibility and deliberate
retirement boundaries, not a second all-feature patch inventory.

## Prerequisites and required behavior

Use the root's exact accepted fork/upstream observations and preserve the behavior
of affected feature units. `chore(fork)`/`sync` entries are reconciliation provenance.
Merge/refork entries have no safe exclusive feature attribution. Revert subjects
record deliberate retirement of only their named change and must not resurrect it.

The subject `fix(memory): release manual write locks promptly` matches both
OV-002's `fix(memory)` prefix and OV-004's literal `release` substring. Keep this
record shared until source inspection establishes dependencies, rather than inventing
precedence. Load [memory and sessions](memory-sessions.md) and
[CLI and distribution](cli-distribution.md) before adapting that commit.
Other shared entries require the affected unit contracts before adaptation:
[retrieval](retrieval.md), [memory and sessions](memory-sessions.md),
[agent plugins](agent-plugins.md), [CLI and distribution](cli-distribution.md),
and [model and API safety](model-api-safety.md). Determine affected units from the
source diff, not the subject alone. A full reconciliation accounts for all five.

## Adoption and verification

Reconcile upstream through the root's update rule. Check cross-unit dependencies
before replaying, reverting, or retiring shared entries. Verify all affected unit
proofs plus `bin/check` and `bin/ci`, then the root's fresh upstream comparison and
owned-remote parity. Ambiguous provenance blocks publication until resolved.
Other fork-range items have no recorded upstream issue/PR after checked 2026-09-09.
This observation is not a current absence claim or runtime proof.

Rollback is a source-level revert after dependent-unit regression proof. Retire a
shared entry only when its continuing compatibility or anti-resurrection decision
is accounted for by released upstream plus the affected unit tests. Keep source
reconciliation, publication, installation, and activation separate.

## Exact shared entries

```text
7516f797267ec69e5919a2de049561202f2cbec9 chore(fork): sync upstream main
13d2e137de14da71880d845106047c1d2ad95fd6 chore(fork): sync upstream main
38d4b99e29a4f3e588659a41ce9fab1f02e3fa20 chore(fork): sync upstream main
6269bc82620a404e9a399513a921188d33d9d14e chore(fork): sync upstream main
b62a7fc8fd56a96a6c891d823174b0ecb49718cb chore(fork): sync upstream main
9471f940ca1e49cc72d859ebe4c6656cf015be3b Merge branch 'feat/queue-dedupe-and-abstract-cache'
27032584ad7a9b788f444cf5f422a895d967f05d fix(memory): release manual write locks promptly
a5fd8f518501821c7e3f61cfa8739ebd96e45d3d chore(fork): sync upstream main
51edd077632762ba8707b453ae3ac967845fb94e chore(fork): sync upstream main 20260422
1bd07019e183f25ee53f6ff9bb314a14b236a77d chore(fork): sync upstream main
35ef4997e9dc7c9303a35b83ee386e6262634208 sync(claude-code-memory-plugin): update to Castor6 v0.1.5
21e74e4b67b574be54e41a37c8a1716e84f67c4b chore(fork): sync upstream main, drop ingestReplyAssist with #1564
48ce9c478aac3218862101b8638fb509b1ca2b3d chore(fork): merge upstream main
e6ac251f34cfc6196b7169305d00887facd5f302 chore(fork): sync upstream main
f6ffcf1d8db70355923d572bf4930451940310c6 revert(openviking): undo local context overflow fixes
5273eb30813da6c2e6d9b253668e2ca55d6e726c refork: hard-cut retrieval time filter cutover
```
