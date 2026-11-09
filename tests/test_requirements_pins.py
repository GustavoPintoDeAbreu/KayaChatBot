"""The requirements.txt pins the nightly lab-audit asked for (2026-10-26, 2026-11-01).

Audited packages with CVEs fixed only in newer releases must be pinned at the
fixed version or carry a floor at it, so a rebuild cannot resolve below it.
gradio is gone: every release caps pillow below 13, and only the dev UI used it. The fine-tuning stack (unsloth,
peft, ...) must be gone, and the serving/teacher pins must stay.
"""

from pathlib import Path
from typing import Optional

import importlib.metadata
import re

import pytest

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _requirements() -> dict:
    requirements = {}
    for raw in (ROOT / "requirements.txt").read_text().splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        requirements[canonicalize_name(requirement.name)] = requirement
    return requirements


def _lowest_allowed(requirement: Requirement) -> Optional[Version]:
    versions = [
        Version(spec.version)
        for spec in requirement.specifier
        if spec.operator in ("==", ">=", "~=", "===")
    ]
    return max(versions) if versions else None


FLOORS = {
    "aiohttp": "3.14.5",
    "anyio": "4.16.0",
    "brotli": "1.2.1",
    "certifi": "2026.10.23",
    "filelock": "4.2.1",
    "fsspec": "2026.6.0",
    "idna": "3.20",
    "oauthlib": "4.0.1",
    "pillow": "13.0.0",
    "pypdf": "6.18.0",
    "python-multipart": "0.0.33",
    "sentence-transformers": "5.6.0",
    "starlette": "1.10.0",
    "transformers": "5.10.0",
    "urllib3": "2.8.0",
}

EXACT_PINS = {
    "ctranslate2": "4.8.5",
    "fastapi": "0.162.2",
    "pandas": "3.1.1",
    "sentencepiece": "0.2.3",
    "transformers": "5.22.2",
}


@pytest.mark.parametrize("name, floor", sorted(FLOORS.items()))
def test_audited_packages_cannot_resolve_below_the_fix(name, floor):
    requirement = _requirements().get(name)
    assert requirement is not None, f"{name} is not pinned in requirements.txt"
    lowest = _lowest_allowed(requirement)
    assert lowest is not None, f"{name} has no lower bound"
    assert lowest >= Version(floor), (
        f"{name} may resolve to {lowest}, below the fixed version {floor}"
    )


def test_gradio_and_its_dev_ui_are_gone():
    assert "gradio" not in _requirements(), "every gradio release caps pillow below 13.0"
    assert not (ROOT / "src" / "chat" / "web_app.py").exists()
    compose = ROOT / "docker-compose.yml"
    if compose.exists():                          # not copied into the image CI tests in
        assert "kaya-dev:" not in compose.read_text()


def test_finetuning_stack_is_gone():
    requirements = _requirements()
    for name in ("unsloth", "unsloth-zoo", "xformers", "torchao", "trl", "peft", "datasets"):
        assert canonicalize_name(name) not in requirements, (
            f"{name} is fine-tuning-only and must not be installed"
        )


def test_serving_and_teacher_dependencies_stay():
    requirements = _requirements()
    for name in ("transformers", "accelerate", "bitsandbytes"):
        assert canonicalize_name(name) in requirements, (
            f"{name} is needed: tokenizer in the app, 4-bit load in the local teacher"
        )


def test_dockerfile_does_not_patch_peft():
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.exists():
        pytest.skip("the Dockerfile is not copied into the image CI tests in")
    assert "peft" not in dockerfile.read_text().lower(), (
        "the PEFT float8 patch belonged to the fine-tuning stack"
    )


def test_finetuning_entry_points_are_gone():
    for path in (
        "src/finetuning",
        "src/chat/chat.py",
        "src/chat/inference.py",
        "src/data/merge_datasets.py",
        "src/data/format_direct_training.py",
        "src/data/generate_local_synthetic.py",
        "src/data/generate_needle_training.py",
        "src/data/readers.py",
        "scripts/export_gguf.py",
        "scripts/train_bakeoff.py",
        "scripts/heretic_sweep.py",
    ):
        assert not (ROOT / path).exists(), f"{path} is fine-tuning-only"


def test_nothing_imports_datasets():
    pattern = re.compile(r"^\s*(import|from)\s+datasets\b", re.MULTILINE)
    offenders = []
    for folder in ("src", "scripts", "tests"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"datasets was dropped from requirements.txt but is imported by {offenders}"


def test_requirements_do_not_move_torch():
    match = re.search(r"Pinned against: torch==([0-9][0-9.]*)", (ROOT / "requirements.txt").read_text())
    assert match, "requirements.txt header no longer names the torch pin"
    try:
        installed = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("torch is not installed here")
    assert Version(installed).public == Version(match.group(1)).public, (
        f"torch {installed} is installed, but requirements.txt is pinned against {match.group(1)}; "
        "a bumped package pulled a different torch"
    )


@pytest.mark.parametrize("name, pinned", sorted(EXACT_PINS.items()))
def test_exact_pins_at_the_audit_version(name, pinned):
    requirement = _requirements().get(name)
    assert requirement is not None, f"{name} is not pinned in requirements.txt"
    assert _lowest_allowed(requirement) == Version(pinned), (
        f"{name} must be pinned exactly at {pinned}, the audit's fixed version"
    )


def test_dockerfile_torch_matches_the_requirements_header():
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.exists():
        pytest.skip("the Dockerfile is not copied into the image CI tests in")
    match = re.search(r"torch==([0-9][0-9.]*)", dockerfile.read_text())
    assert match, "the Dockerfile no longer names a torch pin"
    header = re.search(r"Pinned against: torch==([0-9][0-9.]*)",
                       (ROOT / "requirements.txt").read_text())
    assert header and Version(match.group(1)) == Version(header.group(1)), (
        "the Dockerfile torch pin and the requirements.txt header must bump together"
    )


def test_torch_installs_from_a_cuda_index():
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.exists():
        pytest.skip("the Dockerfile is not copied into the image CI tests in")
    assert re.search(r"--index-url https://download\.pytorch\.org/whl/cu\d+", dockerfile.read_text()), (
        "torch must come from the CUDA-specific PyTorch index, not PyPI"
    )
