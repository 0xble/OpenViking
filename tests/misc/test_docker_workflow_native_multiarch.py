from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_local_ci_pins_upstream_docker_workflow_contract():
    upstream_workflows = _read_text("ci/upstream-workflows.sha256")
    check_script = _read_text("bin/check")

    assert ".github/workflows/build-docker-image.yml" in upstream_workflows
    assert ".github/workflows/release.yml" in upstream_workflows
    assert "check_upstream_parity" in check_script


def test_local_docker_gate_builds_without_registry_pushes():
    check_script = _read_text("bin/check")

    assert "docker build --build-arg" in check_script
    assert "docker push" not in check_script
    assert "docker buildx imagetools create" not in check_script


def test_local_docker_gate_resolves_real_openviking_version():
    check_script = _read_text("bin/check")

    assert "from build_support.versioning import resolve_openviking_version" in check_script
    assert "OPENVIKING_VERSION=$version" in check_script


def test_local_docker_gate_does_not_force_zero_version_on_main_builds():
    check_script = _read_text("bin/check")
    zero_build_arg = (
        "OPENVIKING_VERSION=${{ (github.event_name == 'workflow_dispatch' && "
        "github.event.inputs.version) || (github.ref_type == 'tag' && "
        "github.ref_name) || '0.0.0' }}"
    )

    assert "0.0.0" in check_script
    assert "Resolved invalid OpenViking version" in check_script
    assert zero_build_arg not in check_script
    assert "fallback to 0.0.0" not in check_script


def test_local_docker_gate_uses_stable_local_image_name():
    check_script = _read_text("bin/check")

    assert "openviking:local-ci" in check_script


def test_local_release_gate_never_publishes_registry_manifests():
    check_script = _read_text("bin/check")

    assert "not part of CI" in check_script
    assert "approval-gated release task" in check_script
