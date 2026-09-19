# OV-001: Retrieval and resources

Part of the [root maintenance contract](../MAINTENANCE.md). Read this unit for
changes to its behavior, source surfaces, or upstream equivalents. Every full
maintenance run evaluates this unit. Before adapting or retiring shared commits,
load [source reconciliation](source-reconciliation.md).

## Required behavior and proof surfaces

Retrieval/filter and resource behavior remains source/time-aware.

`openviking/retrieve/`, `openviking/utils/search_filters.py`, `tests/retrieve/`, `tests/unit/test_search_filters.py`.

## Provenance and adoption

The accepted root observation owns the fork and upstream baselines. The entries
below retain the prior register's exact SHA/subject pairs, in their original order.
Routing uses the original scope rule: Subjects beginning `feat(retrieval)`, `fix(retrieval)`, `fix(search)`, `refactor(search)`, `fix(retrieve)`, `fix(fs)`, `fix(resources)`, `feat(resources)`, `feat(metadata)`.
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
85e567fee33dcb369d2c105ec137a630b7011c37 fix(resources): propagate processing instructions to semantic summaries
8faeeb1647fe90dc3b8a5f148bfb9614a0753ca4 fix(retrieval): improve recall resource coverage
2c6f20c77e0ba671d0d355fb845ad1a70ffd5f8e feat(metadata): add resource and session metadata APIs
0716708c1d2eda9221dd1716a256f50ffd15c9de feat(resources): add metadata patch API
808b982a6ccdcd8d32d3ad92f804830e56192a25 feat(resources): expose provenance metadata
d4943dd0be23d9fe9188a32bdcfa51627aeb759b feat(retrieval): add file and session time filters
1c9729c173a3af2fbd1eb71f8339fde6ac5ee2c0 fix(retrieve): skip empty docs before rerank
b6ac5c92b34ddaaabdd8ecc6cacfc47ccfdb1789 fix(fs): grep walks all children, not first 1000
44fa78bfc32a94d7c85f0d04c7da11175978813a refactor(search): consolidate time filter handling
f2aa37c4acd4090c97d2b02b1babb73fd47da686 fix(retrieval): drop stray tracker arg
faac64f593a51197c16ebeeffd74befa05283950 fix(retrieval): reset leaf created_at on reindex
9977e530f29a4a5571c3be007a3a727b82b50657 fix(retrieval): harden rerank and memory reindex
6a08f0f71e64e9a75ce091afd362ec258f66df71 fix(search): restore source and time filtering
61117d39cf6b4183ec295e582bd94ffb08a6573b feat(retrieval): restore source-aware filters and backfill
```
