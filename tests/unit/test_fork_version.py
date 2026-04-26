from build_support.versioning import resolve_openviking_version
from openviking._fork_version import FORK_VERSION_SUFFIX, apply_fork_version_suffix


def test_apply_fork_version_suffix_is_idempotent():
    version = f"0.3.13{FORK_VERSION_SUFFIX}"

    assert apply_fork_version_suffix(version) == version


def test_resolve_openviking_version_applies_suffix_to_env_version():
    version = resolve_openviking_version(env={"OPENVIKING_VERSION": "0.3.13"})

    assert version == f"0.3.13{FORK_VERSION_SUFFIX}"


def test_apply_fork_version_suffix_normalizes_legacy_hyphen_suffix():
    version = apply_fork_version_suffix("0.3.13-0xble.1.2.2")

    assert version == f"0.3.13{FORK_VERSION_SUFFIX}"
