from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_packaging_only_includes_abi3_engine_extensions():
    setup_py = _read_text("setup.py")
    pyproject = _read_text("pyproject.toml")

    assert "storage/vectordb/engine/*.abi3.so" in setup_py
    assert "storage/vectordb/engine/*.abi3.so" in pyproject
    assert "storage/vectordb/engine/*.dll" not in setup_py
    assert "storage/vectordb/engine/*.dll" not in pyproject
    assert "storage/vectordb/engine/*.so" not in setup_py
    assert "storage/vectordb/engine/*.so" not in pyproject


def test_windows_engine_loader_registers_dll_search_paths():
    engine_init = _read_text("openviking/storage/vectordb/engine/__init__.py")

    assert "add_dll_directory" in engine_init
    assert "module_path" in engine_init
    assert 'package_root / "lib"' in engine_init
    assert 'package_root / "bin"' in engine_init


def test_setup_no_longer_bundles_mingw_runtime_dlls_for_engine():
    setup_py = _read_text("setup.py")

    assert "WINDOWS_ENGINE_RUNTIME_DLL_PATTERNS" not in setup_py
    assert "_stage_windows_engine_runtime_dlls" not in setup_py


def test_local_ci_pins_upstream_workflow_contract_and_drops_pybind11():
    check_script = _read_text("bin/check")
    upstream_workflows = _read_text("ci/upstream-workflows.sha256")
    uv_lock = _read_text("uv.lock")

    assert ".github/workflows/_build.yml" in upstream_workflows
    assert ".github/workflows/_test_lite.yml" in upstream_workflows
    assert "check_upstream_parity" in check_script
    assert "refs/heads/main:refs/remotes/upstream/main" in check_script
    assert "pybind11" not in check_script
    assert 'name = "pybind11"' not in uv_lock


def test_local_build_gate_no_longer_defines_extra_wheel_verify_jobs():
    check_script = _read_text("bin/check")

    assert "verify-linux-abi3-wheel:" not in check_script
    assert "verify-macos-14-wheel-on-macos-15:" not in check_script


def test_local_build_gate_smoke_tests_wheel_engine_import():
    check_script = _read_text("bin/check")

    assert "-m pip install --force-reinstall dist/*.whl" in check_script
    assert "import openviking.storage.vectordb.engine as engine" in check_script
    assert "engine.ENGINE_VARIANT" in check_script


def test_local_build_gate_smoke_tests_ragfs_binding_import():
    check_script = _read_text("bin/check")

    assert "from openviking.pyagfs import get_binding_client" in check_script


def test_windows_abi3_backend_uses_stable_python_linkage():
    setup_py = _read_text("setup.py")
    src_cmake = _read_text("src/CMakeLists.txt")

    assert "OV_PYTHON_SABI_LIBRARY" in setup_py
    assert "python3.dll" in setup_py
    assert "OV_PYTHON_SABI_LIBRARY" in src_cmake
    assert "Python3::Python" not in src_cmake


def test_local_build_gate_no_longer_defines_windows_python312_verify_job():
    check_script = _read_text("bin/check")

    assert "verify-windows-abi3-wheel-on-python312:" not in check_script
    assert "Smoke test Windows abi3 wheel on Python 3.12" not in check_script


def test_abi3_backend_releases_gil_and_rejects_invalid_storage_op_type():
    backend_source = _read_text("src/abi3_engine_backend.cpp")

    assert "PyEval_SaveThread" in backend_source
    assert "PyEval_RestoreThread" in backend_source
    assert "Invalid storage op type" in backend_source


def test_repo_no_longer_contains_pybind11_engine_bindings():
    assert not (REPO_ROOT / "src" / "pybind11_interface.cpp").exists()
    assert not (REPO_ROOT / "src" / "cpu_feature_probe.cpp").exists()
    assert not (REPO_ROOT / "src" / "py_accessors.h").exists()


def test_python_engine_exports_only_live_abi3_api():
    import openviking.storage.vectordb.engine as engine

    assert not hasattr(engine, "FetchDataResult")
