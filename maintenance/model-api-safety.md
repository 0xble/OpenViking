# OV-005: Model and API safety

Part of the [root maintenance contract](../MAINTENANCE.md). Read this unit for
changes to its behavior, source surfaces, or upstream equivalents. Every full
maintenance run evaluates this unit. Before adapting or retiring shared commits,
load [source reconciliation](source-reconciliation.md).

## Required behavior and proof surfaces

Model, retry, safety, parsing, observability and API compatibility fixes remain present.

`openviking/models/`, `openviking/utils/`, `openviking/server/`, `tests/models/`, `tests/server/`, `tests/unit/`.

## Provenance and adoption

The accepted root observation owns the fork and upstream baselines. The entries
below retain the prior register's exact SHA/subject pairs, in their original order.
Routing uses the original scope rule: Remaining non-reconciliation subjects after OV-001 through OV-004 selection.
Shared reconciliation records and overlapping subject matches remain with source
reconciliation. Subject routing is an inventory aid, not proof of exclusive code
ownership. Inspect the source diff and dependencies before acting.

Commit `3e9ff0cb8849bdfe32315ee269dcbb103a6f34bf` is source-associated with
merged [volcengine/OpenViking#1592](https://github.com/volcengine/OpenViking/pull/1592)
(merged 2026-05-08). It ports the upstream reindex API and retains fork-side
consolidator/maintenance-router wiring. Association does not establish released
equivalence. Retirement requires both OV-002 and OV-005 source and proof surfaces.

Load [memory and sessions](memory-sessions.md) before changing or retiring that reindex integration.

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
94ebeacf48e8785f76008f6ba45508e60784b777 fix(vectordb): accept local aggregate count key
76693fbb012818f34a056bf34cf7392c62fe6ae4 test(server): reset MCP session manager at running_server setup too
799a9522437ed0917efd5878e3399bb19f172606 fix(memory/utils/uri): tolerate None values for non-template-referenced memory fields
0b00948f8cb75ca0dcc057c211456255eea75c32 fix(tests): bump APIKeyManager wait to 30s in SDK server fixture
eb3a17a62a62e16fa10dfd8fc4bacb45d64887c8 test(memory): skip legacy _apply_edit tests after upstream renamed to _apply_upsert
12b08b4e46794086b9556621b67d8cc3413fd321 test(server): reset MCP session manager before lifespan isolation test
3d0b4e269dd50efe6413bebe8e9acf6b4c5d9080 fix(vlm,embedder,time): restore fork's slow-call threshold, time formatting, streaming, and port upstream max_retries/extra_body
0f6900eea32037049a3deeb4245c833157214af5 fix(memory/utils): drop redundant ResolvedOperations import in uri.py
a87ca21068cbd55e423622c369e1037a0e45fd35 fix(openviking): restore wm queue waiting
f0529700c3b7ce30a0b42a260ddaadc28a3fd81f fix(openviking): repair session sync fallout
496a5cecfa8c238e4b2092a76cd3492af7f5647f fix(openviking): resolve context engine sync regressions
bd4b40667e4a0527c6c92ed6b6a3a154d90c7980 fix(tests): make fork_version legacy-suffix test resilient to version bumps
23602a352a11311c1b3f6ac3523fa5a1e15151aa feat(server): add tolerate_bare_session_id flag for GET /sessions/{id}
aa60805d1f5523922cf1e2463f9dc2ca66f64051 fix(openviking): avoid repeated memory consolidation plans
5ebf7168387356a233eba24ca94c5b668351e704 fix(openviking): harden auth and remote input handling
e4e90b1beec2849a74ce3ad5436d2736610fd2ed fix(openviking): restore local ci after upstream sync
a50564635aeec808d8aa7e41999c651b116e20b4 docs(storage): mark imported session sidecars legacy
faa4203a9afc5346e83f5cda9bacd89441bea446 fix(telemetry): tolerate summary telemetry shims
e3bbc315afa1f70b2db1438652d0ca1dbd3ec442 fix(queue): close handler status accounting
d7580ae94faf36e070ae0500bb7a290c29323f62 fix(reindex): handle memory file targets
d9796b523980234105717bae32cd6fea1c602121 fix(observer): support transaction status alias
51837ee4519031f1f2d22dd9d50928be4ebb18c3 fix(telemetry): preserve async operation attribution
d17218dd32ca23d599dcee74136f9bba8c820eba fix(vlm): make usage ledger lock portable
9fe7d094642cf531a9d41afed682ff32cdcd53bc fix(vlm): serialize usage ledger writes
8d1608334a1103efa4b9da58ec5dbfaf9c3cfbd1 fix(observer): guard optional model usage data
c9658a8f9a1efc1365fd89b980aca42995712633 fix(vlm): use minimal flash-lite reasoning
707a2afa131ed5a3cb25f596a53a34334d9f4c2d feat(vlm): add usage ledger and safer extraction
cffa0779083d1a91cdbe20d54897b5f115636e2e fix(content): bypass queues for direct writes
a12452c183059bd6bedefddb48c4cca818058a22 fix(ci): exclude integration-marked tests from pre-push gate
e076b3675fe77fe86e7e363877f78c8eca5c77c7 fix(semantic): skip rewrites on overview cache hit; tighten cache wiring
82d80920fab9ecd24ed6ad11fa709ac634c7f163 perf(semantic): hash-skip the VLM call when overview inputs are unchanged
8d4d93c718b878d41d7f6945f1eefd87c87ffac2 perf(semantic): generalize parent-semantic dedupe to resource and session
051c2a7b83995e982e63df85e71b972c77f216e2 fix(ci): require full tests before push
c3dc527e8aa9346c21efef4843584f6f79d51675 fix(tests): restore openviking test suite
e5edc83e7d3ba82403042c2c587c528c507e855d fix(semantic): skip generated session summaries
cfbae7a4beede9d815ec2a8d1975de3bae6cafd7 fix(ci): clean native artifacts before packaging
b7e10a73fb39c092027aed2b2ae7c42ddcc5bb4c test(memory): format semantic stall test
472d8db75e4bed53d46ba6ea31d72bdc937ffda8 fix(fork): normalize local ci version handling
71d5c82659b9c8e362b4bd6564b42d9df5594f35 ci(local): replace github workflows with local check gate
76f721b849d3ea002c75f251a61a24828b8f36f3 fix(fork): harden openviking runtime provenance
bd56d8e218f156be590d5733a91255e7bd06d221 docs(retrieval): keep upstream docs unchanged
fb985ec354da399d54fe60b962bf5697fa39bf0f test(server): align regression tests with tenant auth
d3eaf455219b8f1edd8864dc898277cdeb2da799 test(search): speed up grep overflow regression
009f92c742532c7b7234a1d646d57bd873bd4cb8 refactor(safety): apply simplify findings on runaway-billing batch
524931ce3c90adfbb72dfc90228e62b699ded128 feat(safety): add OPENVIKING_DISABLE_VLM kill-switch env var
0c9920a1493ee51a08e520932b6efaa19eb9a3d3 refactor(vlm): use shared retry_async in volcengine backend
48563839022d8078f0352894b7140d7df251c58e fix(retry): exclude rate-limit errors from transient retry path
3e9ff0cb8849bdfe32315ee269dcbb103a6f34bf feat(reindex): land upstream PR #1592 admin reindex api with consolidator integration
17c1e1277b80348af2418605c2bd0e50606da86e fix(claude): sync recall module into runtime
2d5551c26d25d65336b3436d358bca13b3fc7764 feat(session): make extract work against archived messages
c31e0015c987447a32c65784dc2a58e55b268006 feat(semantic): add output_language_override to pin summary/overview language
8452d7a2cc0fffcc76262ec8f8ee05e44505e6b4 fix(rerank): accept data response envelope
93045d87d07821f1da7974bc650c6f5472972c43 fix(fork): repair ov mkdir sync conflict
eada985ea19f5869f698bc570746214a5c34d145 fix(ragfs): truncate local writes on create
91c913e4b817f5afa1396caaaf11bfa3165b6c6a ci(docker): use rust toolchain required by dependencies
92ffc408d86343bc6f23357906034689acaf5af7 ci(docker): include native extension third party sources
08cab532406779de6d008027a213681a319b2fff ci(docker): skip docker hub publish without credentials
2ac362a8905f6ab290ba32edecc4e6b86d0d15d5 fix(examples): use active session id for archive expand
51d4718e6583dac42feaa1299bf9a5957f98994a feat(examples): wire retrieval usage feedback into memory runtimes
451c03c0bc5262afa74131dbc4f72379e272946d feat(ov): cut over retrieval filters
101cbdd507af8114cc0402723343daa704d9f29e test(search): align telemetry assertions with pruning
```
