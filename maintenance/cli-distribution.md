# OV-004: CLI and distribution

Part of the [root maintenance contract](../MAINTENANCE.md). Read this unit for
changes to its behavior, source surfaces, or upstream equivalents. Every full
maintenance run evaluates this unit. Before adapting or retiring shared commits,
load [source reconciliation](source-reconciliation.md).

## Required behavior and proof surfaces

CLI, packaging, build and release fork behavior remains buildable without changing distribution policy.

`crates/ov_cli/`, `openviking_cli/`, `build_support/`, `bin/`, `Cargo.lock`, `uv.lock`.

## Provenance and adoption

The accepted root observation owns the fork and upstream baselines. The entries
below retain the prior register's exact SHA/subject pairs, in their original order.
Routing uses the original scope rule: Subjects containing `cli`, `build`, `release`, `lockfiles`, `fork version`, or `codesign`.
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
42910a080d15037fa521bc69e869592c33109a25 fix(client): initialize agent dirs before adding skills
cd9ffd1ada5d280856ee10bcddb67caa7456d791 fix(crates/ov_cli): restore fork's ov_cli sources after upstream sync
cda3dff5f32cb1bb567751583ca2fb4a26489760 fix(ci): rebuild native artifacts during local checks
3b223aee86d27f6ff3c7e64620fa7dce5d575bcf fix(cli): honor wait timeout body
4d5366b18b37e2676ffaf7f313a0d9fc92621cfd fix(cli): align retrieval time flags with upstream
2763ba45b2b181010c27d7869cb3f929ef55af00 chore(release): bump fork to 0xble.1.2.0
2c138fb36f96aed10436034ac44ceea5598b6e87 test(rebuild): correct agent_namespace semantic_calls expectation and drop bogus xfails
3b449711574f8f64c57a338d9534820243a322bd fix(fork): repair unresolved upstream-sync conflict in ov_cli main.rs
6fbfb1ed68f16f35907ebc0ad149c2f97ae86b77 fix(ov_cli): wire Extract arm into main.rs handle_session dispatch
05d5db98097e22fdd1338462bb9c07c05fe8326e fix(build): codesign ov during upgrade
048735ea162256e0b859f20359ea63cafe8c9698 chore(lockfiles): sync Cargo.lock and uv.lock
```
