"""Fork-local version metadata."""

FORK_VERSION_SUFFIX = "-0xble.1.2.2"


def apply_fork_version_suffix(version: str) -> str:
    """Return version with the fork suffix applied exactly once."""
    if FORK_VERSION_SUFFIX in version:
        return version
    return f"{version}{FORK_VERSION_SUFFIX}"
