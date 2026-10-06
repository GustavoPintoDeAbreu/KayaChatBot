"""The manual shutdown: what /homelaboff starts on the PC, run against fake tools."""
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent.parent
SCRIPT = REPO / "deploy" / "power" / "kaya-shutdown.sh"
BASH = shutil.which("bash")
FLOCK = shutil.which("flock")

pytestmark = pytest.mark.skipif(not BASH or not FLOCK, reason="needs bash and flock")

FAKES = ("sudo", "gdbus", "pgrep", "nvidia-smi", "curl", "systemctl", "rtcwake", "notify-send")


def _write_fakes(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in FAKES:
        (bin_dir / name).write_text(
            '#!/usr/bin/env bash\nprintf \'%s\\t%s\\n\' ' + name + ' "$*" >> "${FAKE_LOG:?}"\n')
        (bin_dir / name).chmod(0o755)
    # sudo drops -u USER and VAR=val args, then execs the rest (as_user's job).
    (bin_dir / "sudo").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\t%s\\n\' sudo "$*" >> "${FAKE_LOG:?}"\n'
        'args=()\nwhile [ $# -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    -u) shift 2 ;;\n'
        '    *=*) shift ;;\n'
        '    *) args+=("$1"); shift ;;\n'
        '  esac\n'
        'done\nexec "${args[@]}"\n')
    (bin_dir / "sudo").chmod(0o755)
    (bin_dir / "gdbus").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\t%s\\n\' gdbus "$*" >> "${FAKE_LOG:?}"\n'
        'printf "(uint64 ${FAKE_IDLE_MS:-3600000},)\\n"\n')
    (bin_dir / "gdbus").chmod(0o755)
    (bin_dir / "pgrep").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\t%s\\n\' pgrep "$*" >> "${FAKE_LOG:?}"\n'
        'if [ -n "${FAKE_PGREP:-}" ] && [[ "$*" == *"$FAKE_PGREP"* ]]; then exit 0; fi\n'
        'exit 1\n')
    (bin_dir / "pgrep").chmod(0o755)
    (bin_dir / "docker").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\t%s\\n\' docker "$*" >> "${FAKE_LOG:?}"\n'
        'if [[ "$*" == *"label=idea-pipeline.idea"* ]]; then\n'
        '  printf "%s\\n" "${FAKE_IDEA_CONTAINERS:-}"\n'
        'else\n'
        '  printf "%s\\n" "${FAKE_DOCKER_PS:-}"\n'
        'fi\n')
    (bin_dir / "docker").chmod(0o755)
    (bin_dir / "nvidia-smi").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\t%s\\n\' nvidia-smi "$*" >> "${FAKE_LOG:?}"\n'
        'printf "%s\\n" "${FAKE_GPU_UTIL:-0}"\n')
    (bin_dir / "nvidia-smi").chmod(0o755)
    (bin_dir / "curl").write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\t%s\\n\' curl "$*" >> "${FAKE_LOG:?}"\n'
        'if [[ "$*" == *"relay/status"* ]]; then\n'
        '  status=${FAKE_RELAY_STATUS:-\'{"pending_replies": 0}\'\n'
        '  printf "%s" "$status"\n'
        'fi\n')
    (bin_dir / "curl").chmod(0o755)


@pytest.fixture
def rig(tmp_path):
    _write_fakes(tmp_path)
    env_file = tmp_path / "kaya-power.env"
    env_file.write_text("KAYA_RELAY_TOKEN=relay\n")
    boot = tmp_path / "boot"
    boot.mkdir()
    (boot / "vmlinuz-7.0.0-35").write_text("")
    modules = tmp_path / "modules" / "7.0.0-35"
    modules.mkdir(parents=True)
    (modules / "nvidia.ko").write_text("")
    env = {
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "KAYA_POWER_ENV": str(env_file),
        "POWER_USER": getpass.getuser(),
        "POWER_STAY_ON_FILE": str(tmp_path / "stay-on"),
        "DOCKER_BIN": str(tmp_path / "bin" / "docker"),
        "POWER_BOOT_DIR": str(boot),
        "POWER_MODULES_DIR": str(tmp_path / "modules"),
        "POWER_STATE_DIR": str(tmp_path / "state"),
        "POWER_REPO": str(REPO),
        "POWER_PYTHON": sys.executable,
        "POWER_GPU_RECHECK_SECONDS": "0",
        "POWER_WARN_SECONDS": "0",
        "POWER_MANUAL_CHECK_SECONDS": "1",
        "POWER_GATEWAY_URL": "http://gw.test:8088",
        "POWER_PC_APP_URL": "http://app.test:7860",
        "FAKE_LOG": str(tmp_path / "fakes.log"),
    }
    return env, tmp_path


def _run_script(env, *args, timeout=60):
    return subprocess.run([BASH, str(SCRIPT), *args], env=env, capture_output=True,
                          text=True, timeout=timeout)


def _fakes(tmp_path):
    log = tmp_path / "fakes.log"
    if not log.exists():
        return []
    return [line.split("\t", 1) for line in log.read_text().splitlines() if line]


def _updates(tmp_path):
    """The /pc/power/update bodies, in log order.

    The curl fake logs ``curl\t<args space-joined>``; the body is the value after
    ``--data`` and the URL is the last token, so split on those two anchors.
    """
    updates = []
    for name, rest in _fakes(tmp_path):
        if name != "curl" or " --data " not in rest:
            continue
        start = rest.find(" --data ") + len(" --data ")
        url = rest.rsplit(" ", 1)[-1]
        body = rest[start: len(rest) - len(url) - 1]
        if "/pc/power/update" in url:
            updates.append(json.loads(body))
    return updates


def _poweroffs(tmp_path):
    return [rest for name, rest in _fakes(tmp_path)
            if name == "systemctl" and rest.startswith("poweroff")]


def test_report_lists_every_reason(rig):
    env, tmp_path = rig
    (tmp_path / "stay-on").write_text("")
    env["FAKE_PGREP"] = "Runner.Worker"
    env["FAKE_DOCKER_PS"] = "kaya-sim"
    env["FAKE_IDLE_MS"] = "60000"
    env["FAKE_GPU_UTIL"] = "90"
    result = _run_script(env, "--report")
    assert result.returncode == 0
    lines = [line.split("\t") for line in result.stdout.splitlines() if line]
    kinds = {kind for kind, _ in lines}
    assert kinds == {"note", "busy"}
    joined = "\n".join(result.stdout.splitlines())
    assert "keyboard/mouse used 1 min ago" in joined
    assert "a CI job is running" in joined
    assert "job containers running: kaya-sim" in joined
    assert "a GPU is 90% busy" in joined
    assert "stay-on exists" in joined
    assert "kernel 7.0.0-35 has no nvidia module" not in joined
    # .stay-on is a note in the report, never a busy reason.
    busy = [value for kind, value in lines if kind == "busy"]
    assert not any("stay-on" in reason for reason in busy)


def test_report_counts_idea_pipeline_containers(rig):
    env, tmp_path = rig
    env["FAKE_IDEA_CONTAINERS"] = "abc123"
    result = _run_script(env, "--report")
    assert result.returncode == 0
    assert "an idea-pipeline build or test container is running" in result.stdout
    docker_calls = [rest for name, rest in _fakes(tmp_path) if name == "docker"]
    assert any("idea-pipeline.idea" in rest for rest in docker_calls)


def test_gpu_is_kayas_while_she_is_replying(rig):
    env, tmp_path = rig
    env["FAKE_GPU_UTIL"] = "90"
    env["FAKE_RELAY_STATUS"] = '{"pending_replies": 2}'
    result = _run_script(env, "--report")
    assert result.returncode == 0
    assert "a GPU is 90% busy" not in result.stdout
    assert "Kaya is finishing 2 replies" in result.stdout
    env["FAKE_RELAY_STATUS"] = '{"pending_replies": 0}'
    result = _run_script(env, "--report")
    assert result.returncode == 0
    assert "a GPU is 90% busy" in result.stdout


def test_manual_idle_powers_off_in_order(rig):
    env, tmp_path = rig
    result = _run_script(env, "--manual")
    assert result.returncode == 0
    names = [name for name, _ in _fakes(tmp_path)]
    assert "systemctl" in names
    assert len(_poweroffs(tmp_path)) == 1
    assert "rtcwake" in names
    updates = _updates(tmp_path)
    assert updates and updates[-1]["final"] is True
    assert "Powering off now" in updates[-1]["text"]
    # In order: "Only Kaya is left", the going-down curl, rtcwake, the final update, poweroff.
    flat = [f"{name} {rest}" for name, rest in _fakes(tmp_path)]
    first_kaya = next(i for i, line in enumerate(flat) if "Only Kaya is left" in line)
    going_down = next(i for i, line in enumerate(flat) if "going-down" in line)
    rtcwake = next(i for i, line in enumerate(flat) if line.startswith("rtcwake"))
    final_update = next(i for i, line in enumerate(flat)
                        if "Powering off now" in line and "/pc/power/update" in line)
    poweroff = next(i for i, line in enumerate(flat) if line.startswith("systemctl poweroff"))
    assert first_kaya < going_down < rtcwake < final_update < poweroff
    # No docker stop anywhere.
    assert not any("docker stop" in line for line in flat)
    # The phase file is cleaned up on exit.
    assert not (tmp_path / "state" / "phase").exists()


def test_manual_cancel_while_waiting(rig):
    env, tmp_path = rig
    env["FAKE_PGREP"] = "Runner.Worker"
    proc = subprocess.Popen([BASH, str(SCRIPT), "--manual"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 10
    phase = tmp_path / "state" / "phase"
    while time.time() < deadline:
        if phase.exists() and phase.read_text().strip() == "waiting":
            break
        time.sleep(0.05)
    else:
        proc.kill()
        raise AssertionError("the script never reached the waiting phase")
    (tmp_path / "state" / "cancel").write_text("cancel")
    proc.wait(timeout=15)
    assert proc.returncode == 0
    assert _poweroffs(tmp_path) == []
    updates = _updates(tmp_path)
    assert any("Waiting for:" in update["text"] for update in updates)
    assert updates and updates[-1]["final"] is True and "Cancelled" in updates[-1]["text"]


def test_manual_refuses_a_kernel_without_nvidia(rig):
    env, tmp_path = rig
    (tmp_path / "modules" / "7.0.0-35" / "nvidia.ko").unlink()
    result = _run_script(env, "--manual")
    assert result.returncode == 0
    assert _poweroffs(tmp_path) == []
    updates = _updates(tmp_path)
    assert updates and updates[-1]["final"] is True and "no nvidia module" in updates[-1]["text"]


def test_manual_gives_up_after_the_cap(rig):
    env, tmp_path = rig
    env["POWER_MANUAL_MAX_WAIT_MINUTES"] = "0"
    env["FAKE_PGREP"] = "Runner.Worker"
    result = _run_script(env, "--manual")
    assert result.returncode == 0
    assert _poweroffs(tmp_path) == []
    updates = _updates(tmp_path)
    assert updates and updates[-1]["final"] is True and "Gave up" in updates[-1]["text"]


def test_scheduled_dry_run_is_unchanged(rig):
    env, tmp_path = rig
    result = _run_script(env, "--dry-run")
    assert result.returncode == 0
    assert "dry run: the PC is idle and would shut down now" in result.stdout
    assert _poweroffs(tmp_path) == []
    assert not any(name == "systemctl" for name, _ in _fakes(tmp_path))
