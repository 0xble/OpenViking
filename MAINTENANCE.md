# Maintenance

## Background

Maintained fork `0xble/OpenViking` of `volcengine/OpenViking`; canonical source
checkout `/Users/brianle/OpenViking-pr-repair`, maintained branch `main`, owned
remote `origin`, source remote `upstream`. Accepted observation on 2026-09-09:
`origin/main` `a8998823173079ca691e38e8728a3eddde234dbc`; freshly fetched
`upstream/main` `98f7dfe0e39333ee9d11f83c563d59fa3a3b6a71` (1223 upstream-only,
189 fork-only). This is a maintained divergent fork, not a contribution fork.

## Preserve

- Publish only to `origin`; upstream is source authority and is never a push target.
- Preserve the fork-only behavior recorded below across retrieval, memory/session,
  server, CLI, plugin, build, and safety surfaces; current source and its named
  tests decide conflicts, not historical patch order.
- Source reconciliation, publication, installation, and runtime activation are
  separate stages; this contract authorizes none of the latter two.

## Active patch register

The complete active provenance is the 189 exact stable subjects below (full SHA);
`chore(fork)`/`sync` entries are reconciliation provenance, and revert subjects
record deliberate retirement of only their named change. No upstream issue or
upstream PR is recorded for this private fork range after checked 2026-09-09.
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
| OV-005 | Model, retry, safety, parsing, observability and API compatibility fixes remain present; `openviking/models/`, `openviking/utils/`, `openviking/server/`, `tests/models/`, `tests/server/`, `tests/unit/`. | Remaining non-reconciliation subjects in the complete index. |

### Complete provenance index

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
94ebeacf48e8785f76008f6ba45508e60784b777 fix(vectordb): accept local aggregate count key
7516f797267ec69e5919a2de049561202f2cbec9 chore(fork): sync upstream main
42910a080d15037fa521bc69e869592c33109a25 fix(client): initialize agent dirs before adding skills
76693fbb012818f34a056bf34cf7392c62fe6ae4 test(server): reset MCP session manager at running_server setup too
799a9522437ed0917efd5878e3399bb19f172606 fix(memory/utils/uri): tolerate None values for non-template-referenced memory fields
0b00948f8cb75ca0dcc057c211456255eea75c32 fix(tests): bump APIKeyManager wait to 30s in SDK server fixture
19ff5a6a36148350c08d1d6e21b625504c3a21b3 fix(session,memory): respect redo_recovery_enabled flag + pass viking_fs to memory tools + use clean registry in tests
28032139c3fb1adc48bd957e22a84b344e572d91 fix(storage,tests): restore embedding URI canonicalization + define _mark_failed; skip stale upstream tests
eb3a17a62a62e16fa10dfd8fc4bacb45d64887c8 test(memory): skip legacy _apply_edit tests after upstream renamed to _apply_upsert
12b08b4e46794086b9556621b67d8cc3413fd321 test(server): reset MCP session manager before lifespan isolation test
3e89e7ae1554977e0c8c8e83a28e54d021367083 fix(server,tests): fs_service import + SDK test tenant alignment
78503e347905d11ab793c1d192fb4ffc3edddd41 fix(server,tests): reset MCP session manager between uvicorn fixtures + 400 for content-type body parse errors
a70c948fcbc1f7816ddc41631c70ffb6e7924961 fix(server,memory,tests): merge-resolution regressions across validation, fs metadata, ovpack vectorization
67e630e36467d25084f8f7294ff8823c19fc4fb1 fix(memory): real-date fallback for empty ranges; dedupe field lines on merge
1705c7fa35b6951cc03339ff61eb3421401ea05a fix(server): treat extra_forbidden validation errors as 400 not 422
8e25d5a942e01b6a7443aecef6509d701101213a fix(server,memory): more merge regressions
9efd821c5dbeb5519bab308d725af693981b9a8c fix(server,memory): resolve merge regressions
3d0b4e269dd50efe6413bebe8e9acf6b4c5d9080 fix(vlm,embedder,time): restore fork's slow-call threshold, time formatting, streaming, and port upstream max_retries/extra_body
0f6900eea32037049a3deeb4245c833157214af5 fix(memory/utils): drop redundant ResolvedOperations import in uri.py
cd9ffd1ada5d280856ee10bcddb67caa7456d791 fix(crates/ov_cli): restore fork's ov_cli sources after upstream sync
13d2e137de14da71880d845106047c1d2ad95fd6 chore(fork): sync upstream main
85e567fee33dcb369d2c105ec137a630b7011c37 fix(resources): propagate processing instructions to semantic summaries
a87ca21068cbd55e423622c369e1037a0e45fd35 fix(openviking): restore wm queue waiting
f0529700c3b7ce30a0b42a260ddaadc28a3fd81f fix(openviking): repair session sync fallout
496a5cecfa8c238e4b2092a76cd3492af7f5647f fix(openviking): resolve context engine sync regressions
38d4b99e29a4f3e588659a41ce9fab1f02e3fa20 chore(fork): sync upstream main
bd4b40667e4a0527c6c92ed6b6a3a154d90c7980 fix(tests): make fork_version legacy-suffix test resilient to version bumps
23602a352a11311c1b3f6ac3523fa5a1e15151aa feat(server): add tolerate_bare_session_id flag for GET /sessions/{id}
36356ab293ec5f42c5867336fef7acb4e7c9150f fix(plugins): tighten resource recall and session routing
6269bc82620a404e9a399513a921188d33d9d14e chore(fork): sync upstream main
aa60805d1f5523922cf1e2463f9dc2ca66f64051 fix(openviking): avoid repeated memory consolidation plans
5ebf7168387356a233eba24ca94c5b668351e704 fix(openviking): harden auth and remote input handling
e4e90b1beec2849a74ce3ad5436d2736610fd2ed fix(openviking): restore local ci after upstream sync
b62a7fc8fd56a96a6c891d823174b0ecb49718cb chore(fork): sync upstream main
a4509d8fcf58cadae9492c5f74bdce52c0b31410 fix(openclaw): use readable memory write output
a50564635aeec808d8aa7e41999c651b116e20b4 docs(storage): mark imported session sidecars legacy
faa4203a9afc5346e83f5cda9bacd89441bea446 fix(telemetry): tolerate summary telemetry shims
e3bbc315afa1f70b2db1438652d0ca1dbd3ec442 fix(queue): close handler status accounting
d7580ae94faf36e070ae0500bb7a290c29323f62 fix(reindex): handle memory file targets
d9796b523980234105717bae32cd6fea1c602121 fix(observer): support transaction status alias
51837ee4519031f1f2d22dd9d50928be4ebb18c3 fix(telemetry): preserve async operation attribution
d17218dd32ca23d599dcee74136f9bba8c820eba fix(vlm): make usage ledger lock portable
efd0ccef4415f77eaaeca7cba94341b3680ccfa0 fix(session): keep short real prompts extractable
d6adf688fc13aa199b5c1cf53b738c27788499c0 fix(session): extract structured heartbeat text
9fe7d094642cf531a9d41afed682ff32cdcd53bc fix(vlm): serialize usage ledger writes
8d1608334a1103efa4b9da58ec5dbfaf9c3cfbd1 fix(observer): guard optional model usage data
c9658a8f9a1efc1365fd89b980aca42995712633 fix(vlm): use minimal flash-lite reasoning
707a2afa131ed5a3cb25f596a53a34334d9f4c2d feat(vlm): add usage ledger and safer extraction
56fc5a1304417aa428217b039592eb70ae842608 fix(memory): tolerate missing extraction provenance
cffa0779083d1a91cdbe20d54897b5f115636e2e fix(content): bypass queues for direct writes
a12452c183059bd6bedefddb48c4cca818058a22 fix(ci): exclude integration-marked tests from pre-push gate
e076b3675fe77fe86e7e363877f78c8eca5c77c7 fix(semantic): skip rewrites on overview cache hit; tighten cache wiring
9471f940ca1e49cc72d859ebe4c6656cf015be3b Merge branch 'feat/queue-dedupe-and-abstract-cache'
82d80920fab9ecd24ed6ad11fa709ac634c7f163 perf(semantic): hash-skip the VLM call when overview inputs are unchanged
8d4d93c718b878d41d7f6945f1eefd87c87ffac2 perf(semantic): generalize parent-semantic dedupe to resource and session
a89f04ef59c85400d140899d09ecea7c93f6c3c4 feat(memory): readable tool result text for memory_write/store/forget
c59c9e3296382912a28c3951a8f9c0233ba71eab fix(session): prefer requested id when loading moved sessions
051c2a7b83995e982e63df85e71b972c77f216e2 fix(ci): require full tests before push
c3dc527e8aa9346c21efef4843584f6f79d51675 fix(tests): restore openviking test suite
e5edc83e7d3ba82403042c2c587c528c507e855d fix(semantic): skip generated session summaries
e1a112ff61d4a5bca641a10f6355cd3a99447e69 fix(memory): complete deduped write waits
27032584ad7a9b788f444cf5f422a895d967f05d fix(memory): release manual write locks promptly
cda3dff5f32cb1bb567751583ca2fb4a26489760 fix(ci): rebuild native artifacts during local checks
cfbae7a4beede9d815ec2a8d1975de3bae6cafd7 fix(ci): clean native artifacts before packaging
3b223aee86d27f6ff3c7e64620fa7dce5d575bcf fix(cli): honor wait timeout body
b7e10a73fb39c092027aed2b2ae7c42ddcc5bb4c test(memory): format semantic stall test
472d8db75e4bed53d46ba6ea31d72bdc937ffda8 fix(fork): normalize local ci version handling
6aaf4c29879d4ddc6e58976c9da35995c575b1eb fix(memory): return manual writes after enqueue
3d16f2d8cd62fb794d29244b8467bb8e760bd983 fix(memory): harden manual write contention
71d5c82659b9c8e362b4bd6564b42d9df5594f35 ci(local): replace github workflows with local check gate
76f721b849d3ea002c75f251a61a24828b8f36f3 fix(fork): harden openviking runtime provenance
4939e646a076a86fe0e8908701b25a362d987b70 fix(memory): repair event directory summaries
da03d7b5dc6d0eed6d9f5e4d9c5d36b2697dce00 fix(memory): harden maintenance writes
9e8c80f0c4a0c90a03454ceb368ab1fe3ab1a957 fix(memory): reject placeholder semantic artifacts
85e99575bc3104b23403b44deb7c2a2520d727a4 fix(memory): extend wait write timeouts
a4a35a705f3fb415a0b03ae15a28a33f9188d2ce feat(plugins): add resource recall tools
8faeeb1647fe90dc3b8a5f148bfb9614a0753ca4 fix(retrieval): improve recall resource coverage
e4a8374340307f53a1b22a1ccfd40e7b333162d6 fix(memory): harden extraction and maintenance
2c6f20c77e0ba671d0d355fb845ad1a70ffd5f8e feat(metadata): add resource and session metadata APIs
0716708c1d2eda9221dd1716a256f50ffd15c9de feat(resources): add metadata patch API
808b982a6ccdcd8d32d3ad92f804830e56192a25 feat(resources): expose provenance metadata
0cf8cdb21d6e6a2373dbfcfa1b174a410a0532a0 fix(storage): persist semantic summary cache by filename
bd56d8e218f156be590d5733a91255e7bd06d221 docs(retrieval): keep upstream docs unchanged
d4943dd0be23d9fe9188a32bdcfa51627aeb759b feat(retrieval): add file and session time filters
fb985ec354da399d54fe60b962bf5697fa39bf0f test(server): align regression tests with tenant auth
4d5366b18b37e2676ffaf7f313a0d9fc92621cfd fix(cli): align retrieval time flags with upstream
a5fd8f518501821c7e3f61cfa8739ebd96e45d3d chore(fork): sync upstream main
9b0ec89565d6f79f402d1bed8bdc898cb3ec48a3 fix(storage): preserve stat errors in content write
947274e814ddb4d842ec1b6b0b34e7d7ac645fc2 fix(session): scope memory dedup search by owner
b29642bbbcbaa8eeb481982d911cb9a40555ef71 fix(memory): preserve content write create mode
d3eaf455219b8f1edd8864dc898277cdeb2da799 test(search): speed up grep overflow regression
009f92c742532c7b7234a1d646d57bd873bd4cb8 refactor(safety): apply simplify findings on runaway-billing batch
1c9729c173a3af2fbd1eb71f8339fde6ac5ee2c0 fix(retrieve): skip empty docs before rerank
2763ba45b2b181010c27d7869cb3f929ef55af00 chore(release): bump fork to 0xble.1.2.0
524931ce3c90adfbb72dfc90228e62b699ded128 feat(safety): add OPENVIKING_DISABLE_VLM kill-switch env var
0c9920a1493ee51a08e520932b6efaa19eb9a3d3 refactor(vlm): use shared retry_async in volcengine backend
48563839022d8078f0352894b7140d7df251c58e fix(retry): exclude rate-limit errors from transient retry path
2c138fb36f96aed10436034ac44ceea5598b6e87 test(rebuild): correct agent_namespace semantic_calls expectation and drop bogus xfails
3e9ff0cb8849bdfe32315ee269dcbb103a6f34bf feat(reindex): land upstream PR #1592 admin reindex api with consolidator integration
7b3861a136b7bfd6e9c3da312d0880ddf6b7f0d5 fix(maintenance): gate consolidator phase 5 vlm regen on actual mutations
b3115673efe7c85e086b2f8f9b168c26d11a49cc fix(openclaw): quiet expected recall fallbacks
17c1e1277b80348af2418605c2bd0e50606da86e fix(claude): sync recall module into runtime
1fd559a099687670d2123b1f16b08c2b19490ee5 fix(plugins): tighten recall and budget helpers
3f6092ed4915c276dde876ce94a65dd51afd0393 fix(plugins): align OpenViking memory adapters
7a81f640e2e91a6f19ddb6e3b0bfa13964cf8890 fix(openclaw): harden openviking recall timeouts
5e3818dcc033349eec17f90a4de3d329cc843645 fix(openclaw-plugin): restore createHash import on OpenVikingClient
0e1b9bb14607f27de79f80dd857569bb43204af7 fix(openclaw-plugin): restore spaceCache field on OpenVikingClient
35a8a653c0bede3d6555989e26a2ca9fb72dcb71 fix(openclaw-plugin): align OpenVikingClient call with 10-arg signature
6cf1c67ca24e19f07218d91a9d291c75d58e5303 feat(plugins): add session_recall tool across claude-code, codex, openclaw
3b449711574f8f64c57a338d9534820243a322bd fix(fork): repair unresolved upstream-sync conflict in ov_cli main.rs
b6ac5c92b34ddaaabdd8ecc6cacfc47ccfdb1789 fix(fs): grep walks all children, not first 1000
51edd077632762ba8707b453ae3ac967845fb94e chore(fork): sync upstream main 20260422
08b90bae25a7e0e4354284f4cfacf091509e2e58 chore(openclaw-plugin): bump pluginVersion to 2026.4.21
d007fc4fcf0991345738c5c3b6a3c68e3bed9858 ci(openclaw-plugin): lint install-manifest against runtime imports
1a216e333940dfb65ffef2ca60468dc07a51f449 fix(openclaw-plugin): include recall modules in install-manifest
9003cbec1c26c6aacaa688575c61a3174a003d1e fix(openclaw-plugin): default recall to assemble and bound afterTurn
1c6f6b568f78f17ed82d75471f63c8cf68dd045c refactor(codex-plugin): rename openviking_* tools to memory_* for consistency
b67cbb1ecdc0a89894b7a0c006ecf9f7bf7adfa8 fix(plugins): memory_write review fixes — unwrap request, tighten URI checks
1b316ad50846342ed37b7a800edd58a3f2fd06b6 feat(plugins): add memory_write tool across 4 memory plugins
6fbfb1ed68f16f35907ebc0ad149c2f97ae86b77 fix(ov_cli): wire Extract arm into main.rs handle_session dispatch
695b06d655eb056a0d86c5afae999a6e8c584520 fix(session): prevent duplicate memories during redo recovery
2d5551c26d25d65336b3436d358bca13b3fc7764 feat(session): make extract work against archived messages
ae993b4ba1f9031def2c5c819fd0d7e2ad1e3850 feat(memory): add POST /api/v1/memories for direct memory creation
c31e0015c987447a32c65784dc2a58e55b268006 feat(semantic): add output_language_override to pin summary/overview language
56b3f20c1a1d2c2cc131556a35ef7c251ac37f38 feat(openclaw-plugin): categorize commit errors in compact reason
35d8cdcb31c25824e2a0a511642d63b36ae784d4 ci: add fork regression gate for openclaw-plugin contract tests
1a92f7fab43744866f4396af160d26fe787ed155 fix(openclaw-plugin): afterTurn uses structured parts with tool fidelity
d0d2b15bb0f9682e72fc225b419c32506cdcc103 feat(openclaw-plugin): self-heal dormant sessions in compactOVSession
85271cd09807b0774bd941c46f09ed1a76d96ac1 fix(openclaw-plugin): send addSessionMessage text as parts array
1bd07019e183f25ee53f6ff9bb314a14b236a77d chore(fork): sync upstream main
c1efc53f2c47b491cafb8ff013fc3a6c6e6a9b64 feat(memory): per-canary top_n sensitivity knob
68762d2b20928c7beb3ca4bdcf98940e90c76b47 fix(memory): /consolidate/runs returns audit records (was empty)
14dff19cc4a3e34ea515d47869912064523b497f feat(memory): scheduler + HTTP endpoints + canary phase (Phases B-D)
2a3b7befabfa55d22104082a5e703f1ec5be2c07 feat(memory): add scheduled consolidation foundation (Phase A)
cccc67b9567f8723d4c53ed7a2640a6b2c75f5c5 fix(claude-code-memory-plugin): drop unresolved ${VAR} env passthrough in .mcp.json
3cbab082cb9e2e6c0166e49fd3e9cecded664d01 feat(marketplace): add .claude-plugin/marketplace.json for fork distribution
17289c929089290516d3fac5c7a30532cbb85cec fix(claude-code-memory-plugin): exec ready runtime when CLAUDE_PLUGIN_ROOT is missing
35ef4997e9dc7c9303a35b83ee386e6262634208 sync(claude-code-memory-plugin): update to Castor6 v0.1.5
21e74e4b67b574be54e41a37c8a1716e84f67c4b chore(fork): sync upstream main, drop ingestReplyAssist with #1564
48ce9c478aac3218862101b8638fb509b1ca2b3d chore(fork): merge upstream main
8452d7a2cc0fffcc76262ec8f8ee05e44505e6b4 fix(rerank): accept data response envelope
93045d87d07821f1da7974bc650c6f5472972c43 fix(fork): repair ov mkdir sync conflict
e6ac251f34cfc6196b7169305d00887facd5f302 chore(fork): sync upstream main
f6ffcf1d8db70355923d572bf4930451940310c6 revert(openviking): undo local context overflow fixes
eada985ea19f5869f698bc570746214a5c34d145 fix(ragfs): truncate local writes on create
afe3fc64f3babb392c54a8b53c9ea578c2932633 fix(session): clear live transcript on archive
41f1af3477d1897f108a4674305312c6c5783fe3 fix(session): bound live compaction fallback
8ef771d24402a2ec701af51175827c7025ad5e5f fix(openclaw-plugin): route memory to configured tenant
64c93a5ad606f5172fa297093cc4f11697847d72 fix(session): stabilize detached commit completion
47872230363216f5a75bf64947ef3c6e3132329f fix(session): detach memory extraction from commit completion
4584cc07e064a77095d533919aa61b84c7a8da30 fix(openclaw-plugin): avoid stale json5 config fallback
44fa78bfc32a94d7c85f0d04c7da11175978813a refactor(search): consolidate time filter handling
8b338d2d3d490c8285882bb9398b3da5a565410d fix(openclaw-plugin): bound adaptive recall caches
a05b605812641dee69f086fcacdb78c31f02f900 fix(openclaw-plugin): adapt memory recall latency
91c913e4b817f5afa1396caaaf11bfa3165b6c6a ci(docker): use rust toolchain required by dependencies
92ffc408d86343bc6f23357906034689acaf5af7 ci(docker): include native extension third party sources
08cab532406779de6d008027a213681a319b2fff ci(docker): skip docker hub publish without credentials
7dc9699c40248dc1e82025d2b4eff41a2654cd99 fix(openclaw-plugin): harden openviking session reliability
251f5669351aa78023a863a82a580413a53cc011 fix(openclaw-plugin): detect active openclaw config file
633bfc49376ffa9fd04b89d6e7fc066c0e88bb70 fix(openclaw-plugin): move recall into assemble by default
05d5db98097e22fdd1338462bb9c07c05fe8326e fix(build): codesign ov during upgrade
f2aa37c4acd4090c97d2b02b1babb73fd47da686 fix(retrieval): drop stray tracker arg
faac64f593a51197c16ebeeffd74befa05283950 fix(retrieval): reset leaf created_at on reindex
5273eb30813da6c2e6d9b253668e2ca55d6e726c refork: hard-cut retrieval time filter cutover
2ac362a8905f6ab290ba32edecc4e6b86d0d15d5 fix(examples): use active session id for archive expand
9977e530f29a4a5571c3be007a3a727b82b50657 fix(retrieval): harden rerank and memory reindex
048735ea162256e0b859f20359ea63cafe8c9698 chore(lockfiles): sync Cargo.lock and uv.lock
698e64cf02ed13fb90552da6d8961af6e3a08e06 fix(codex-memory-plugin): wait for delete consistency
51d4718e6583dac42feaa1299bf9a5957f98994a feat(examples): wire retrieval usage feedback into memory runtimes
50c339d87e55695beba50e5d5d64ef30f539c70f feat(mcp): proxy example to shared http backend
451c03c0bc5262afa74131dbc4f72379e272946d feat(ov): cut over retrieval filters
101cbdd507af8114cc0402723343daa704d9f29e test(search): align telemetry assertions with pruning
6a08f0f71e64e9a75ce091afd362ec258f66df71 fix(search): restore source and time filtering
61117d39cf6b4183ec295e582bd94ffb08a6573b feat(retrieval): restore source-aware filters and backfill
```

## Update

Each run fetches `origin` and `upstream` separately, reconciles latest
`upstream/main` while preserving only this register, runs the applicable focused
tests/build, then publishes to `origin/main` or reports `Blocked` with stage,
refs, and evidence. Immediately before `Updated` or `Already current`, fetch
`upstream` again and require `git rev-list --left-right --count
upstream/main...main` to have upstream-only count `0`; fetch `origin` and require
`origin/main == HEAD`. Any addition, change, retirement, or ambiguous provenance
updates this register before publication. Current source reconciliation is
Blocked by the recorded 1223 upstream-only commits; this docs-only enrollment
must not be called a source sync.

## Verify

```text
bin/check
bin/ci
python -m pytest tests/unit/maintenance tests/retrieve tests/server/test_api_maintenance.py
make build-cli
git diff --check upstream/main...HEAD
git rev-list --left-right --count upstream/main...main
```

For a docs-only change, run `git diff --check` and inspect this index; do not
install, activate, deploy, or otherwise alter a runtime. Sync success additionally
requires the final fresh-fetch zero count and owned-remote SHA parity above.
