# OV-002: Memory and session isolation

Part of the [root maintenance contract](../MAINTENANCE.md). Read this unit for
changes to its behavior, source surfaces, or upstream equivalents. Every full
maintenance run evaluates this unit. Before adapting or retiring shared commits,
load [source reconciliation](source-reconciliation.md).

## Required behavior and proof surfaces

Memory, maintenance, tenant, redo, and session behavior remains isolated and auditable.

`openviking/maintenance/`, `openviking/session/`, `openviking/storage/`, `tests/unit/maintenance/`, `tests/session/`, `tests/server/test_api_maintenance.py`.

## Provenance and adoption

The accepted root observation owns the fork and upstream baselines. The entries
below retain the prior register's exact SHA/subject pairs, in their original order.
Routing uses the original scope rule: Subjects beginning `feat(memory)`, `fix(memory)`, `fix(session`, `fix(maintenance)`, `fix(storage`, `fix(server`, `fix(lock)`.
Shared reconciliation records and overlapping subject matches remain with source
reconciliation. Subject routing is an inventory aid, not proof of exclusive code
ownership. Inspect the source diff and dependencies before acting.

Commit `3e9ff0cb8849bdfe32315ee269dcbb103a6f34bf` is source-associated with
merged [volcengine/OpenViking#1592](https://github.com/volcengine/OpenViking/pull/1592)
(merged 2026-05-08). It ports the upstream reindex API and retains fork-side
consolidator/maintenance-router wiring. Association does not establish released
equivalence. Retirement requires both OV-002 and OV-005 source and proof surfaces.

Load [model and API safety](model-api-safety.md) before changing or retiring that reindex integration.

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
a8998823173079ca691e38e8728a3eddde234dbc fix(server): write maintenance audits as system
76822460bc34628fe26a9060a54a09f0ed605f2d fix(server): keep dry-run maintenance failures transient
0c7e527cfdf8d7c055d8c2511c1e8e1a28f8ed61 fix(server): respect shared agent maintenance scopes
335a35e606fa5aef4136200680dbe698699b1f82 fix(server): preserve concurrent maintenance dirties
167fdfd9a6b1cf2e297779827ec50ba3adb05fa7 fix(server): respect shared user maintenance scopes
452d64e1920f093c9f16893c8f7d3b21f6ed1afc fix(server): filter maintenance scopes before capping
d674319a41663f3c1e4fc8d49baf04ed1f6dd90b fix(server): isolate maintenance scope bookkeeping
b6d351ec245a87ab65b439bace66cd4984d57f13 fix(server): keep failed maintenance scopes dirty
d2254d848014ede3f69e0c052807ed376692a94c fix(server): validate maintenance scope ownership
f2c40c15f5eccfae967a28e89e8d60f57d2d1967 fix(server): enforce maintenance scope tenancy
9fff80907214622e382ab627da97303d610d9116 fix(server): tighten memory maintenance run semantics
24576aa0acb98e0ad0aedb8d91b1c40322fb98fc fix(server): expose memory maintenance endpoints
c60ce18c2e91b73b08a1235afe2f3dcce2f6eff5 fix(memory): accept serialized search tool results
9bf16827f9fc1516f4c98a8d727fecdf54494a1f fix(lock): preserve batch timeout defaults
623e670249674874d1acd59ecbd3fbc3a1c6b749 fix(session): bound archive memory lock attempts
19ff5a6a36148350c08d1d6e21b625504c3a21b3 fix(session,memory): respect redo_recovery_enabled flag + pass viking_fs to memory tools + use clean registry in tests
28032139c3fb1adc48bd957e22a84b344e572d91 fix(storage,tests): restore embedding URI canonicalization + define _mark_failed; skip stale upstream tests
3e89e7ae1554977e0c8c8e83a28e54d021367083 fix(server,tests): fs_service import + SDK test tenant alignment
78503e347905d11ab793c1d192fb4ffc3edddd41 fix(server,tests): reset MCP session manager between uvicorn fixtures + 400 for content-type body parse errors
a70c948fcbc1f7816ddc41631c70ffb6e7924961 fix(server,memory,tests): merge-resolution regressions across validation, fs metadata, ovpack vectorization
67e630e36467d25084f8f7294ff8823c19fc4fb1 fix(memory): real-date fallback for empty ranges; dedupe field lines on merge
1705c7fa35b6951cc03339ff61eb3421401ea05a fix(server): treat extra_forbidden validation errors as 400 not 422
8e25d5a942e01b6a7443aecef6509d701101213a fix(server,memory): more merge regressions
9efd821c5dbeb5519bab308d725af693981b9a8c fix(server,memory): resolve merge regressions
efd0ccef4415f77eaaeca7cba94341b3680ccfa0 fix(session): keep short real prompts extractable
d6adf688fc13aa199b5c1cf53b738c27788499c0 fix(session): extract structured heartbeat text
56fc5a1304417aa428217b039592eb70ae842608 fix(memory): tolerate missing extraction provenance
a89f04ef59c85400d140899d09ecea7c93f6c3c4 feat(memory): readable tool result text for memory_write/store/forget
c59c9e3296382912a28c3951a8f9c0233ba71eab fix(session): prefer requested id when loading moved sessions
e1a112ff61d4a5bca641a10f6355cd3a99447e69 fix(memory): complete deduped write waits
6aaf4c29879d4ddc6e58976c9da35995c575b1eb fix(memory): return manual writes after enqueue
3d16f2d8cd62fb794d29244b8467bb8e760bd983 fix(memory): harden manual write contention
4939e646a076a86fe0e8908701b25a362d987b70 fix(memory): repair event directory summaries
da03d7b5dc6d0eed6d9f5e4d9c5d36b2697dce00 fix(memory): harden maintenance writes
9e8c80f0c4a0c90a03454ceb368ab1fe3ab1a957 fix(memory): reject placeholder semantic artifacts
85e99575bc3104b23403b44deb7c2a2520d727a4 fix(memory): extend wait write timeouts
e4a8374340307f53a1b22a1ccfd40e7b333162d6 fix(memory): harden extraction and maintenance
0cf8cdb21d6e6a2373dbfcfa1b174a410a0532a0 fix(storage): persist semantic summary cache by filename
9b0ec89565d6f79f402d1bed8bdc898cb3ec48a3 fix(storage): preserve stat errors in content write
947274e814ddb4d842ec1b6b0b34e7d7ac645fc2 fix(session): scope memory dedup search by owner
b29642bbbcbaa8eeb481982d911cb9a40555ef71 fix(memory): preserve content write create mode
7b3861a136b7bfd6e9c3da312d0880ddf6b7f0d5 fix(maintenance): gate consolidator phase 5 vlm regen on actual mutations
695b06d655eb056a0d86c5afae999a6e8c584520 fix(session): prevent duplicate memories during redo recovery
ae993b4ba1f9031def2c5c819fd0d7e2ad1e3850 feat(memory): add POST /api/v1/memories for direct memory creation
c1efc53f2c47b491cafb8ff013fc3a6c6e6a9b64 feat(memory): per-canary top_n sensitivity knob
68762d2b20928c7beb3ca4bdcf98940e90c76b47 fix(memory): /consolidate/runs returns audit records (was empty)
14dff19cc4a3e34ea515d47869912064523b497f feat(memory): scheduler + HTTP endpoints + canary phase (Phases B-D)
2a3b7befabfa55d22104082a5e703f1ec5be2c07 feat(memory): add scheduled consolidation foundation (Phase A)
afe3fc64f3babb392c54a8b53c9ea578c2932633 fix(session): clear live transcript on archive
41f1af3477d1897f108a4674305312c6c5783fe3 fix(session): bound live compaction fallback
64c93a5ad606f5172fa297093cc4f11697847d72 fix(session): stabilize detached commit completion
47872230363216f5a75bf64947ef3c6e3132329f fix(session): detach memory extraction from commit completion
```
