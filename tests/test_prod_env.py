"""scripts/prod_env.sh: the deploy never blanks ~/kaya-prod/.env, and refuses one prod cannot run on."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prod_env.sh"

GOOD_PI = """KAYA_WEB_USER=user
KAYA_WEB_PASS=secret
KAYA_EDGE=pi
KAYA_TUNNEL=pi
KAYA_PROD_WAHA_URL=http://pi:3000
KAYA_RELAY_TOKEN=relay
KAYA_INFERENCE_BACKEND=ollama
KAYA_PROD_OLLAMA_URL=http://llm-broker:8080/upstream/kaya
XAI_API_KEY=old-xai
"""


def run(cmd, env_file, **secrets):
    env = {key: value for key, value in os.environ.items()
           if key not in ("XAI_API_KEY", "KAYA_WEB_USER", "KAYA_WEB_PASS", "CLOUDFLARE_TUNNEL_TOKEN",
                          "AZURE_OPENAI_API_KEY_gpt_41_mini", "AZURE_OPENAI_API_KEY_gpt_53_chat")}
    env.update(secrets)
    return subprocess.run(["bash", str(SCRIPT), cmd, str(env_file)], env=env, capture_output=True, text=True)


def values(env_file):
    return dict(line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line)


def test_empty_secrets_leave_env_untouched(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(GOOD_PI)
    result = run("update", env_file, XAI_API_KEY="", KAYA_WEB_USER="", CLOUDFLARE_TUNNEL_TOKEN="")
    assert result.returncode == 0
    assert env_file.read_text() == GOOD_PI


def test_set_secrets_replace_their_line_and_keep_the_rest(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(GOOD_PI)
    result = run("update", env_file, XAI_API_KEY="new-xai", CLOUDFLARE_TUNNEL_TOKEN="tok")
    assert result.returncode == 0
    got = values(env_file)
    assert got["XAI_API_KEY"] == "new-xai"
    assert got["CLOUDFLARE_TUNNEL_TOKEN"] == "tok"
    assert got["KAYA_EDGE"] == "pi" and got["KAYA_RELAY_TOKEN"] == "relay"
    assert env_file.read_text().count("XAI_API_KEY=") == 1
    assert "new-xai" not in result.stdout


def test_check_accepts_a_complete_pi_edge(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(GOOD_PI)
    assert run("check", env_file).returncode == 0


def test_check_refuses_the_blank_env_the_workflow_wrote(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("XAI_API_KEY=\nKAYA_WEB_USER=\nKAYA_WEB_PASS=\nCLOUDFLARE_TUNNEL_TOKEN=\n")
    result = run("check", env_file)
    assert result.returncode == 1
    assert "KAYA_EDGE" in result.stderr and "KAYA_WEB_USER" in result.stderr


def test_check_requires_relay_and_waha_url_on_the_pi_edge(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(GOOD_PI.replace("KAYA_RELAY_TOKEN=relay\n", ""))
    result = run("check", env_file)
    assert result.returncode == 1 and "KAYA_RELAY_TOKEN" in result.stderr


def test_check_requires_the_tunnel_token_only_for_a_local_tunnel(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(GOOD_PI.replace("KAYA_TUNNEL=pi", "KAYA_TUNNEL=local"))
    result = run("check", env_file)
    assert result.returncode == 1 and "CLOUDFLARE_TUNNEL_TOKEN" in result.stderr


def test_check_rejects_an_unknown_edge(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(GOOD_PI.replace("KAYA_EDGE=pi", "KAYA_EDGE=cloud"))
    assert run("check", env_file).returncode == 1
