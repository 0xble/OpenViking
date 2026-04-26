from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path
from typing import Mapping

SCM_TAG_REGEX = r"^(?:v)?(?:[a-zA-Z0-9_]+@)?(?P<version>[0-9]+(?:\.[0-9]+)*)$"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_scm_version(project_root: Path) -> str:
    try:
        from setuptools_scm import get_version
    except ImportError:
        return _get_generated_or_git_version(project_root)

    return get_version(
        root=str(project_root),
        relative_to=__file__,
        local_scheme="no-local-version",
        tag_regex=SCM_TAG_REGEX,
    )


def _get_generated_or_git_version(project_root: Path) -> str:
    generated_version_path = project_root / "openviking" / "_version.py"
    generated_version = _load_version_from_file(generated_version_path, "version")
    if generated_version:
        return generated_version

    result = subprocess.run(
        ["git", "describe", "--tags", "--always"],
        cwd=str(project_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    described = result.stdout.strip()
    match = re.match(r"^v?([0-9]+\.[0-9]+\.[0-9]+)", described)
    if match:
        return f"{match.group(1)}.dev0"
    return "0.0.0+unknown"


def _load_version_from_file(path: Path, attr_name: str) -> str:
    if not path.exists():
        return ""
    spec = importlib.util.spec_from_file_location(f"_openviking_{attr_name}", path)
    if spec is None or spec.loader is None:
        return ""
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, attr_name, "")
    return value if isinstance(value, str) else ""


def _apply_fork_version_suffix(version: str, project_root: Path) -> str:
    fork_version_path = project_root / "openviking" / "_fork_version.py"
    spec = importlib.util.spec_from_file_location("_openviking_fork_version", fork_version_path)
    if spec is None or spec.loader is None:
        return version
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_fork_version_suffix(version)


def resolve_openviking_version(
    env: Mapping[str, str] | None = None, project_root: Path | None = None
) -> str:
    """Resolve the version shared by the Python package and bundled ov binary."""
    env = env or os.environ
    project_root = project_root or PROJECT_ROOT

    for key in ("OPENVIKING_VERSION", "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING"):
        value = env.get(key, "").strip()
        if value:
            return _apply_fork_version_suffix(value, project_root)

    return _apply_fork_version_suffix(_get_scm_version(project_root), project_root)
