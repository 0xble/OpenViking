"""Fork-local version metadata."""

FORK_VERSION_SUFFIX = "+0xble.1.3.0"
LEGACY_FORK_VERSION_SUFFIX = "-0xble.1.3.0"


def apply_fork_version_suffix(version: str) -> str:
    """Return version with the fork suffix applied exactly once."""
    if FORK_VERSION_SUFFIX in version:
        return version
    if LEGACY_FORK_VERSION_SUFFIX in version:
        return version.replace(LEGACY_FORK_VERSION_SUFFIX, FORK_VERSION_SUFFIX)
    return f"{version}{FORK_VERSION_SUFFIX}"
