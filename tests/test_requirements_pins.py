"""The requirements.txt pins the 2026-10-26 audit asked for.

Four packages had audited CVEs with fixes only in newer releases; the pin must
name the fixed version (or a floor at it) so a rebuild cannot resolve below it.
gradio is gone: every release caps pillow below 13, and only the dev UI used it. The fine-tuning stack (unsloth,
peft, ...) must be gone, and the serving/teacher pins must stay.
"""

from pathlib import Path
from typing import Optional

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
    "anyio": "4.14.2",
    "certifi": "2026.10.19",
    "pillow": "13.0.0",
    "sentence-transformers": "5.6.0",
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
    for name in ("unsloth", "unsloth-zoo", "xformers", "torchao", "trl", "peft"):
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
