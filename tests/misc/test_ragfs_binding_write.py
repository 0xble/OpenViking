"""Regression tests for the RAGFS Python binding."""

import pytest


def test_ragfs_binding_write_replaces_shorter_localfs_content(tmp_path):
    from openviking.pyagfs import get_binding_client

    try:
        RAGFSBindingClient, _ = get_binding_client()
    except ImportError as exc:
        pytest.skip(f"ragfs_python native library is unavailable: {exc}")

    if RAGFSBindingClient is None:
        pytest.skip("ragfs_python native library is unavailable")

    client = RAGFSBindingClient()
    client.mount("localfs", "/local", {"local_dir": str(tmp_path)})

    client.write("/local/test.md", b"abcdef")
    client.write("/local/test.md", b"xy")

    assert (tmp_path / "test.md").read_bytes() == b"xy"
