#!/usr/bin/env python3
"""Benchmark runtime serving options for KayaChatBot on shared GPU.

Decides how Kaya's model should be served when GPUs are shared on demand:
(a) what llama-swap adds over today's direct llama-server (warm per-request
    overhead, cold start after unload), and (b) whether Ollama is as good and
    as fast as llama.cpp for Kaya.

Everything runs on GPU0 (UUID GPU-ab32b3d2-3bab-2b24-9749-1caa6400f82d),
never GPU1 which serves production.

Usage::

    python scripts/bench_runtime.py up ARM            # start ARM's server, wait healthy
    python scripts/bench_runtime.py down               # remove every kaya-bench-* container
    python scripts/bench_runtime.py latency ARM        # warm overhead, cold start, throughput, VRAM
    python scripts/bench_runtime.py quality ARM [opts] # quality harnesses against ARM
    python scripts/bench_runtime.py vision ARM         # describe 3 synthetic images
    python scripts/bench_runtime.py embed              # bge-m3 CPU vs GPU0
    python scripts/bench_runtime.py whisper            # faster-whisper load/unload/transcribe
    python scripts/bench_runtime.py storage            # estimate Pi journal size per week
    python scripts/bench_runtime.py report [--stamp S] # aggregate results into Markdown report

Every command writes/merges results into reports/benchmarks/runtime_<stamp>.json
under the key <command> (and <command>.<arm> for per-arm commands).
"""

from __future__ import annotations

import argparse
import base64
import datetime
import gc
import glob
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
REPORTS = BASE_DIR / "reports" / "benchmarks"
LOGS = BASE_DIR / "logs" / "bench_runtime"
SWAP_CONFIG_DIR = BASE_DIR / "logs" / "bench_runtime"

GPU0_UUID = "GPU-ab32b3d2-3bab-2b24-9749-1caa6400f82d"
GPU0_MIN_FREE_MIB = 14000

PYTHON = str(BASE_DIR / "kaya_chatbot_env" / "bin" / "python")

# The llama.cpp build prod has served since 2026-07-20 (b10068), pinned so the
# direct arm and the broker arm run the same engine.
LLAMA_CPP_IMAGE = ("ghcr.io/ggml-org/llama.cpp@sha256:"
                   "4162d942c67debe0859fc1c50605af198d9328e201e1e9f607f5bf38d73b0f42")

# Server definitions for each arm.
ARM_DEFS: Dict[str, Dict[str, Any]] = {
    "direct": {
        "server": "llama-server",
        "container": "kaya-bench-direct",
        "base_url": "http://127.0.0.1:8081",
        "health_url": "http://127.0.0.1:8081/health",
        "health_status": 200,
        "type": "llama",
        "model": "gemma-4-12b-it-Q6_K.gguf",
        "mmproj": "mmproj-F16.gguf",
        "docker_extra": [
            "--runtime", "nvidia",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-e", f"CUDA_VISIBLE_DEVICES={GPU0_UUID}",
            "-v", f"{BASE_DIR}/models/gguf:/models:ro",
            "-p", "127.0.0.1:8081:8080",
            "--entrypoint", "/app/llama-server",
        ],
        "image": LLAMA_CPP_IMAGE,
        "server_args": [
            "-m", "/models/gemma-4-12b-it-Q6_K.gguf",
            "--mmproj", "/models/mmproj-F16.gguf",
            "-ngl", "999", "-fa", "off", "--jinja",
            "-sm", "none", "--parallel", "1",
            "-c", "32768",
            "--host", "0.0.0.0", "--port", "8080",
        ],
    },
    # Same server as "direct" plus --swa-full. Gemma 4 uses sliding-window
    # attention; llama.cpp's default reduced SWA cache has to be rebuilt on a
    # cache hit (~1.2 s on a 3k-token prompt), which Ollama does not pay.
    "direct-swafull": {
        "server": "llama-server",
        "container": "kaya-bench-swafull",
        "base_url": "http://127.0.0.1:8082",
        "health_url": "http://127.0.0.1:8082/health",
        "health_status": 200,
        "type": "llama",
        "docker_extra": [
            "--runtime", "nvidia",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-e", f"CUDA_VISIBLE_DEVICES={GPU0_UUID}",
            "-v", f"{BASE_DIR}/models/gguf:/models:ro",
            "-p", "127.0.0.1:8082:8080",
            "--entrypoint", "/app/llama-server",
        ],
        "image": LLAMA_CPP_IMAGE,
        "server_args": [
            "-m", "/models/gemma-4-12b-it-Q6_K.gguf",
            "--mmproj", "/models/mmproj-F16.gguf",
            "-ngl", "999", "-fa", "off", "--jinja",
            "-sm", "none", "--parallel", "1",
            "-c", "32768", "--swa-full",
            "--host", "0.0.0.0", "--port", "8080",
        ],
    },
    "swap": {
        "server": "llama-swap proxy",
        "container": "kaya-bench-swap",
        "base_url": "http://127.0.0.1:8091/upstream/kaya",
        "health_url": "http://127.0.0.1:8091/health",
        "health_status": 200,
        "type": "llama",
        "proxy_port": 8091,
        "config_file": str(SWAP_CONFIG_DIR / "swap-config.yaml"),
        "docker_extra": [
            "--runtime", "nvidia",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-v", f"{BASE_DIR}/models/gguf:/models/kaya:ro",
            "-v", "${SWAP_CONFIG}:/config/config.yaml:ro",
            "-p", "127.0.0.1:8091:8080",
        ],
        "image": "llm-broker:257-b10068",
    },
    "ollama-q8": {
        "server": "Ollama (gemma4:12b-it-q8_0)",
        "container": "kaya-bench-ollama",
        "base_url": "http://127.0.0.1:11434",
        "health_url": "http://127.0.0.1:11434/api/version",
        "health_status": 200,
        "type": "ollama",
        "model": "gemma4:12b-it-q8_0",
        "ollama_pull": "gemma4:12b-it-q8_0",
        "docker_extra": [
            "--runtime", "nvidia",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-e", f"CUDA_VISIBLE_DEVICES={GPU0_UUID}",
            "-e", "OLLAMA_KEEP_ALIVE=30m",
            "-e", "OLLAMA_CONTEXT_LENGTH=32768",
            "-v", f"{os.environ.get('HOME', '')}/.ollama-bench:/root/.ollama",
            "-v", f"{BASE_DIR}/models/gguf:/gguf:ro",
            "-p", "127.0.0.1:11434:11434",
        ],
        "image": "ollama/ollama:latest",
    },
    "ollama-q6": {
        "server": "Ollama (kaya-q6, local Q6_K import)",
        "container": "kaya-bench-ollama",
        "base_url": "http://127.0.0.1:11434",
        "health_url": "http://127.0.0.1:11434/api/version",
        "health_status": 200,
        "type": "ollama",
        "model": "kaya-q6",
        "docker_extra": [
            "--runtime", "nvidia",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-e", f"CUDA_VISIBLE_DEVICES={GPU0_UUID}",
            "-e", "OLLAMA_KEEP_ALIVE=30m",
            "-e", "OLLAMA_CONTEXT_LENGTH=32768",
            "-v", f"{os.environ.get('HOME', '')}/.ollama-bench:/root/.ollama",
            "-v", f"{BASE_DIR}/models/gguf:/gguf:ro",
            "-p", "127.0.0.1:11434:11434",
        ],
        "image": "ollama/ollama:latest",
    },
}

# Portuguese filler paragraph for latency prompts.
_FILLER_PARAGRAPH = (
    "O grupo de amigos é uma das formas mais importantes de apoio social que "
    "existem. Através da partilha de experiências, histórias e memórias, os "
    "amigos criam uma rede de suporte mútuo que enriquece a vida de todos os "
    "membros. Quando pensamos nas nossas aventuras em conjunto, desde as "
    "jantadas semanais até às viagens de fim de ano, percebemos como cada "
    "momento contribui para a construção de uma identidade partilhada. É "
    "importante lembrar que a memória coletiva é o que mantém vivo o espírito "
    "do grupo. Cada piada interna, cada momento descontraído no ginásio, cada "
    "conversa até tarde sobre os planos de vida, tudo isso faz parte do tecido "
    "que une o grupo. E quando um membro passa por um momento difícil, são "
    "justamente essas memórias partilhadas que nos lembram porque é que "
    "valemos a pena lutar por cada pessoa. O futebol é outro pilar central: "
    "aos sábados, no campo, cada passe e cada golo são celebrados como se "
    "fossem vitórias de campeonato. E os vídeos que alguns membros editam "
    "com tanto cuidado mostram o quanto cada um se importa com o grupo. "
    "Sempre que alguém partilha um documento interessante ou um artigo sobre "
    "política ou economia, o grupo ganha uma nova perspetiva para debater "
    "os temas que nos dizem respeito. Estas são as histórias que queremos "
    "manter vivas, para que no futuro possamos contar aos nossos filhos e "
    "netos sobre quem éramos e o que vivíamos juntos."
)

# Image keyword checks.
# Every group must match (any of its words), compared lower-case.
_IMAGE_KEYWORDS: Dict[str, List[List[str]]] = {
    "red circle": [["vermelh", "red"], ["círcul", "circul", "bola", "circle"]],
    "KAYA text": [["kaya"]],
    "blue bars": [["azul", "blue"], ["barra", "gráfico", "grafico", "bar"]],
}

_TIMEOUT: float = 300.0  # default HTTP timeout in seconds
_HEAL_TIMEOUT: float = 600.0  # health check timeout in seconds


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a timestamped log line."""
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def _now_stamp() -> str:
    """Return current UTC timestamp string for report filenames."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _newest(pattern: str, since: float) -> Optional[str]:
    """Newest file under REPORTS matching pattern, modified after *since*.

    Mirrors the pattern from model_bakeoff.py.
    """
    hits = [
        p for p in glob.glob(str(REPORTS / pattern))
        if os.path.getmtime(p) >= since - 1
    ]
    return max(hits, key=os.path.getmtime) if hits else None


def _load_report() -> Dict:
    """Load the current report file, or an empty dict."""
    report_path = REPORTS / f"runtime_{args.stamp}.json"
    if report_path.exists():
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_report(data: Dict) -> None:
    """Atomically write the report JSON (tmp + os.replace)."""
    report_path = REPORTS / f"runtime_{args.stamp}.json"
    tmp_path = report_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp_path), str(report_path))


def _check_resumable(section: str) -> bool:
    """Return True if section already exists and --force is not set."""
    if args.force:
        return False
    node: Any = _load_report()
    for key in section.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _run_subprocess(
    cmd: List[str],
    log_path: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> int:
    """Run a subprocess, optionally tee-ing output to *log_path*. Returns exit code."""
    _ensure_dir(log_path.parent) if log_path else None
    _log(f"$ {' '.join(cmd)}" + (f"  (-> {log_path.name})" if log_path else ""))
    full_env = {**os.environ, **(env or {})}
    kwargs: Dict[str, Any] = {
        "cwd": str(BASE_DIR),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "env": full_env,
    }
    if log_path:
        kwargs["stdout"] = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, **kwargs)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            if log_path:
                kwargs["stdout"].write("\n*** killed: timeout ***\n")
                kwargs["stdout"].close()
            return 124
    finally:
        if log_path and "stdout" in kwargs and hasattr(kwargs["stdout"], "close"):
            kwargs["stdout"].close()
    return proc.returncode


def _gpu_free_mib() -> Optional[int]:
    """Return GPU0's free memory in MiB, or None on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and GPU0_UUID in parts[0]:
                return int(parts[1])
    except Exception:
        pass
    return None


def _qwen_impl_running() -> bool:
    """Check whether a qwen-impl container is running."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return any("qwen-impl" in name for name in out.splitlines())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------

def _write_swap_config() -> None:
    """Write the llama-swap config for the swap arm."""
    config_path = Path(ARM_DEFS["swap"]["config_file"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "healthCheckTimeout: 300\n"
        "sendLoadingState: false\n"
        "models:\n"
        "  kaya:\n"
        '    cmd: /app/llama-server --host 127.0.0.1 --port ${PORT} --jinja -ngl 999 '
        "-m /models/kaya/gemma-4-12b-it-Q6_K.gguf "
        "--mmproj /models/kaya/mmproj-F16.gguf -fa off -sm none --parallel 1 -c 32768\n"
        '    env: ["CUDA_VISIBLE_DEVICES=GPU-ab32b3d2-3bab-2b24-9749-1caa6400f82d"]\n'
        "    ttl: 0\n",
        encoding="utf-8",
    )


def _build_docker_cmd(arm_def: Dict[str, Any]) -> List[str]:
    """Build the docker run command for an arm definition."""
    cmd = ["docker", "rm", "-f", arm_def["container"]]
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=30)
    except Exception:
        pass  # container may not exist
    return cmd


def _docker_run(arm_def: Dict[str, Any]) -> None:
    """Start a container for the given arm, removing any existing one first."""
    container_name = arm_def["container"]

    # Remove existing container
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False, capture_output=True, timeout=30,
    )

    cmd = ["docker", "run", "-d", "--name", container_name]

    if arm_def["type"] == "llama":
        if arm_def["server"] == "llama-server":
            # Direct llama-server
            cmd += arm_def["docker_extra"]
            cmd.append(arm_def["image"])
            cmd += arm_def["server_args"]
        else:
            # Swap proxy
            _write_swap_config()
            env_replacements = {
                "${SWAP_CONFIG}": arm_def["config_file"],
            }
            for key, value in env_replacements.items():
                for i, part in enumerate(cmd):
                    cmd[i] = part.replace(key, value)
            cmd += [
                "--runtime", "nvidia",
                "-e", "NVIDIA_VISIBLE_DEVICES=all",
                "-v", f"{BASE_DIR}/models/gguf:/models/kaya:ro",
                "-v", f"{arm_def['config_file']}:/config/config.yaml:ro",
                "-p", f"127.0.0.1:{arm_def['proxy_port']}:8080",
            ]
            cmd.append(arm_def["image"])
    elif arm_def["type"] == "ollama":
        cmd += arm_def["docker_extra"]
        cmd.append(arm_def["image"])

    _log(f"docker {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        _log(f"ERROR: docker run failed: {result.stderr[:200]}")
        sys.exit(1)
    _log(f"Container {container_name} started: {result.stdout.strip()[:80]}")


def _ollama_pull(container_name: str, model: str) -> None:
    """Pull a model inside an Ollama container."""
    _log(f"Pulling {model} in {container_name} ...")
    result = subprocess.run(
        ["docker", "exec", container_name, "ollama", "pull", model],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        _log(f"WARNING: ollama pull failed: {result.stderr[:200]}")
    else:
        _log(f"Model {model} pulled successfully.")


def _ollama_create_kaya_q6(container_name: str) -> None:
    """Create the kaya-q6 Ollama model from our local Q6_K GGUF."""
    _log("Creating kaya-q6 model in Ollama container ...")
    result = subprocess.run(
        [
            "docker", "exec", container_name, "sh", "-c",
            'printf "FROM /gguf/gemma-4-12b-it-Q6_K.gguf\\n" > /tmp/Modelfile '
            '&& ollama create kaya-q6 -f /tmp/Modelfile',
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        _log(f"ERROR: kaya-q6 creation failed: {result.stderr[:200]}")
        sys.exit(1)
    _log("kaya-q6 model created successfully.")


def _wait_healthy(arm_def: Dict[str, Any]) -> bool:
    """Poll the health endpoint until it returns the expected status."""
    health_url = arm_def["health_url"]
    expected = arm_def["health_status"]
    deadline = time.time() + _HEAL_TIMEOUT
    container_name = arm_def["container"]
    while time.time() < deadline:
        # Check container is running
        try:
            inspect = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True, text=True, timeout=10,
            )
            if inspect.stdout.strip() != "true":
                _log(f"  {container_name} not running, retrying ...")
                time.sleep(10)
                continue
        except Exception:
            time.sleep(10)
            continue
        # Check health endpoint
        try:
            resp = requests.get(health_url, timeout=10)
            if resp.status_code == expected:
                _log(f"  {container_name} healthy at {health_url}")
                return True
        except requests.RequestException:
            pass
        time.sleep(10)
    _log(f"  {container_name} did not become healthy within {_HEAL_TIMEOUT}s")
    return False


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def complete(
    arm: str,
    prompt: str,
    n_predict: int,
    stream: bool = True,
) -> Dict[str, float]:
    """Send a completion request and return timing metrics.

    Returns dict with ttft (seconds to first token), total (wall time),
    tokens (generated count), and prompt_tokens.
    """
    arm_def = ARM_DEFS[arm]
    base_url = arm_def["base_url"]
    request_timeout = _timeout_for_arm(arm)
    start = time.perf_counter()
    ttft: Optional[float] = None

    if arm_def["type"] == "llama":
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": True,
            "stream": stream,
            "temperature": 0,
        }
        url = f"{base_url}/completion"
        with requests.post(url, json=payload, timeout=request_timeout, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - start
                if obj.get("stop"):
                    timings = obj.get("timings", {})
                    return {
                        "ttft": round(ttft, 4),
                        "total": round(time.perf_counter() - start, 4),
                        "tokens": int(timings.get("predicted_n", 0)),
                        "prompt_tokens": int(timings.get("prompt_n", 0)),
                    }
    elif arm_def["type"] == "ollama":
        model = arm_def["model"]
        payload = {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": stream,
            "keep_alive": "30m",
            "options": {
                "num_ctx": 32768,
                "num_predict": n_predict,
                "temperature": 0,
            },
        }
        url = f"{base_url}/api/generate"
        with requests.post(url, json=payload, timeout=request_timeout, stream=True) as resp:
            resp.raise_for_status()
            last_eval = 0
            last_prompt = 0
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - start
                if obj.get("eval_count"):
                    last_eval = obj["eval_count"]
                if obj.get("prompt_eval_count"):
                    last_prompt = obj["prompt_eval_count"]
                if obj.get("done"):
                    return {
                        "ttft": round(ttft, 4),
                        "total": round(time.perf_counter() - start, 4),
                        "tokens": last_eval,
                        "prompt_tokens": last_prompt,
                    }
    return {"ttft": 0.0, "total": round(time.perf_counter() - start, 4), "tokens": 0, "prompt_tokens": 0}


def unload(arm: str) -> None:
    """Unload the model for an arm. Raises ValueError for direct."""
    arm_def = ARM_DEFS[arm]
    if arm_def["type"] == "llama" and arm_def["server"] == "llama-server":
        raise ValueError("direct: a resident server has no cold start / unload")
    if arm_def["type"] == "llama" and arm_def["server"] == "llama-swap proxy":
        # Unload the model behind llama-swap
        proxy_url = "http://127.0.0.1:8091"
        requests.post(f"{proxy_url}/api/models/unload", json={"name": "kaya"}, timeout=30)
    elif arm_def["type"] == "ollama":
        model = arm_def["model"]
        # Send keep_alive=0 to unload
        requests.post(
            f"{arm_def['base_url']}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
        # Poll /api/ps until model is gone
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                resp = requests.get(f"{arm_def['base_url']}/api/ps", timeout=5)
                data = resp.json()
                models = data.get("models", []) or []
                if not any(model in m.get("model", "") for m in models):
                    return
            except Exception:
                pass
            time.sleep(2)


def _timeout_for_arm(arm: str) -> float:
    """Return the HTTP timeout for a given arm."""
    if arm.startswith("ollama"):
        return 600.0
    return 300.0


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt() -> str:
    """Build a ~3000-token Portuguese prompt using the chat template.

    Tokenises with transformers.AutoTokenizer and applies the model's chat
    template via _templated_prompt.
    """
    from transformers import AutoTokenizer
    from src.chat.inference_backend import _templated_prompt

    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-4-12b-it")
    filler = _FILLER_PARAGRAPH
    while len(tokenizer(text=filler)["input_ids"]) < 3000:
        filler += "\n\n" + _FILLER_PARAGRAPH
    messages = [{"role": "user", "content": filler + "\n\nResume numa frase."}]
    return _templated_prompt(tokenizer, messages, strip_bos=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_up(arm: str) -> None:
    """Start ARM's server on GPU0 and wait until healthy."""
    if arm not in ARM_DEFS:
        _log(f"ERROR: unknown arm '{arm}'. Available: {', '.join(ARM_DEFS)}")
        sys.exit(1)

    arm_def = ARM_DEFS[arm]
    container_name = arm_def["container"]

    # Check GPU0 free memory
    free_mib = _gpu_free_mib()
    if free_mib is not None and free_mib < GPU0_MIN_FREE_MIB:
        _log(
            f"ERROR: GPU0 has {free_mib} MiB free "
            f"(need {GPU0_MIN_FREE_MIB} MiB). Aborting."
        )
        sys.exit(1)
    elif free_mib is None:
        _log("WARNING: could not query GPU0 memory free; proceeding anyway.")

    # Check for qwen-impl
    if _qwen_impl_running():
        _log("ERROR: a qwen-impl container is running. Aborting.")
        sys.exit(1)

    # Start the container
    _docker_run(arm_def)

    # For Ollama arms, pull/create the model
    if arm_def["type"] == "ollama":
        if arm == "ollama-q8":
            _ollama_pull(container_name, arm_def["ollama_pull"])
        elif arm == "ollama-q6":
            _ollama_create_kaya_q6(container_name)

    # Wait for health
    if not _wait_healthy(arm_def):
        _log(f"ERROR: {container_name} did not become healthy.")
        sys.exit(1)
    _log(f"ARM '{arm}' is ready at {arm_def['base_url']}")


def cmd_down() -> None:
    """Remove every kaya-bench-* container."""
    for arm_def in ARM_DEFS.values():
        container_name = arm_def["container"]
        result = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            _log(f"Stopped {container_name}")
    _log("All kaya-bench-* containers removed.")


def cmd_latency(arm: str) -> Dict:
    """Run warm overhead, cold start, throughput, and VRAM measurements."""
    arm_def = ARM_DEFS[arm]
    base_url = arm_def["base_url"]
    timeout = _timeout_for_arm(arm)

    # Build the prompt
    _log("Building prompt ...")
    prompt = _build_prompt()
    prompt_len = len(prompt)
    _log(f"Prompt length: {prompt_len} chars")

    results: Dict[str, Any] = {"prompt_chars": prompt_len}

    # Warm-up: 3 requests with n_predict=16
    _log("Warm-up: 3 requests (n_predict=16) ...")
    for _i in range(3):
        complete(arm, prompt, 16, stream=True)

    # VRAM after warm-up
    try:
        vram_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        for line in vram_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and GPU0_UUID in parts[0]:
                results["vram_used_mib"] = int(parts[1])
                break
    except Exception:
        results["vram_used_mib"] = None

    # Warm overhead: 50 x n_predict=1 and 50 x n_predict=64
    _log("Warm overhead: 50 x n_predict=1 ...")
    ttft_1: List[float] = []
    total_1: List[float] = []
    for _i in range(50):
        r = complete(arm, prompt, 1, stream=True)
        ttft_1.append(r["ttft"])
        total_1.append(r["total"])

    _log("Warm overhead: 50 x n_predict=64 ...")
    ttft_64: List[float] = []
    total_64: List[float] = []
    for _i in range(50):
        r = complete(arm, prompt, 64, stream=True)
        ttft_64.append(r["ttft"])
        total_64.append(r["total"])

    def _percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"p50": None, "p95": None}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        p50 = sorted_vals[int(n * 0.50)]
        p95 = sorted_vals[min(int(n * 0.95), n - 1)]
        return {"p50": round(p50 * 1000, 1), "p95": round(p95 * 1000, 1)}

    results["warm_overhead_n1"] = {
        "ttft_ms": _percentiles(ttft_1),
        "total_ms": _percentiles(total_1),
    }
    results["warm_overhead_n64"] = {
        "ttft_ms": _percentiles(ttft_64),
        "total_ms": _percentiles(total_64),
    }

    # Throughput: 3 x n_predict=256
    _log("Throughput: 3 x n_predict=256 ...")
    throughput_samples: List[float] = []
    for _i in range(3):
        r = complete(arm, prompt, 256, stream=True)
        gen_time = max(r["total"] - r["ttft"], 0.001)
        tok_per_sec = r["tokens"] / gen_time
        throughput_samples.append(round(tok_per_sec, 2))
    results["throughput_tok_per_sec"] = {
        "samples": throughput_samples,
        "mean": round(sum(throughput_samples) / len(throughput_samples), 2),
    }

    # Cold start (not for direct)
    if not arm.startswith("direct"):
        _log("Cold start: 5 x {unload, wait, one request} ...")
        cold_totals: List[float] = []
        for _i in range(5):
            unload(arm)
            time.sleep(3)
            r = complete(arm, prompt, 1, stream=True)
            cold_totals.append(r["total"])
        results["cold_start"] = {
            "totals_s": [round(t, 4) for t in cold_totals],
            "median_s": round(sorted(cold_totals)[len(cold_totals) // 2], 4),
        }
        # cold_extra_s = median(cold) - p50(warm n_predict=1 total)
        warm_n1_sorted = sorted(total_1)
        warm_n1_p50 = warm_n1_sorted[len(warm_n1_sorted) // 2]
        results["cold_extra_s"] = round(
            results["cold_start"]["median_s"] - warm_n1_p50, 4
        )

        # Drop caches (requires root)
        if os.geteuid() == 0:
            _log("Cold start + drop_caches: 5 runs ...")
            drop_totals: List[float] = []
            for _i in range(5):
                subprocess.run(
                    ["sync", "&&", "echo", "3", ">", "/proc/sys/vm/drop_caches"],
                    shell=True, timeout=30,
                )
                unload(arm)
                time.sleep(3)
                r = complete(arm, prompt, 1, stream=True)
                drop_totals.append(r["total"])
            results["cold_start_drop_caches"] = {
                "totals_s": [round(t, 4) for t in drop_totals],
                "median_s": round(sorted(drop_totals)[len(drop_totals) // 2], 4),
            }
        else:
            results["cold_start_drop_caches"] = "skipped: not root (os.geteuid() != 0)"
    else:
        results["cold_start"] = "not applicable (direct server is resident)"

    _log(f"Latency results for '{arm}' written.")
    return results


def cmd_quality(arm: str, judge: str = "azure", skip_golden: bool = False,
                skip_recall: bool = False) -> Dict:
    """Run quality harnesses against ARM."""
    arm_def = ARM_DEFS[arm]
    base_url = arm_def["base_url"]
    timeout_start = time.time()

    # Determine env for subprocess
    env: Dict[str, str] = {}
    if arm_def["type"] == "llama":
        env["KAYA_INFERENCE_BACKEND"] = "gguf"
        env["KAYA_LLAMA_URL"] = base_url
        env["CUDA_VISIBLE_DEVICES"] = GPU0_UUID
    elif arm_def["type"] == "ollama":
        env["KAYA_INFERENCE_BACKEND"] = "ollama"
        env["KAYA_OLLAMA_URL"] = "http://127.0.0.1:11434"
        env["KAYA_OLLAMA_MODEL"] = arm_def["model"]
        env["CUDA_VISIBLE_DEVICES"] = GPU0_UUID

    results: Dict[str, Any] = {}

    # 1. Conversation probe
    _log("Running conversation probe ...")
    log_path = LOGS / f"{arm}_conversation.log"
    rc = _run_subprocess(
        [PYTHON, "scripts/run_conversation_probe.py", "--tag", arm],
        log_path=log_path,
        env=env,
        timeout=3600,
    )
    conv_report = _newest("conversation_*.json", timeout_start)
    if conv_report:
        conv_data = json.loads(Path(conv_report).read_text(encoding="utf-8"))
        # Extract summary
        results["conversation_probe"] = {
            "summary": conv_data.get("summary", {}),
            "report": conv_report,
        }
        # Median and p90 of per-case seconds
        cases = conv_data.get("results", []) or conv_data.get("cases", [])
        if not cases:
            cases = conv_data.get("conversations", [])
        seconds_list = []
        for case in cases:
            sec = case.get("seconds") or case.get("latency_s") or case.get("total_s")
            if sec is not None:
                seconds_list.append(float(sec))
        if seconds_list:
            seconds_list.sort()
            n = len(seconds_list)
            results["conversation_probe"]["median_seconds"] = round(
                seconds_list[n // 2], 4
            )
            p90_idx = min(int(n * 0.90), n - 1)
            results["conversation_probe"]["p90_seconds"] = round(
                seconds_list[p90_idx], 4
            )
    else:
        results["conversation_probe"] = {"error": f"rc={rc}, no report found"}

    # 2. Offensive probe
    _log("Running offensive probe ...")
    log_path = LOGS / f"{arm}_offensive.log"
    rc = _run_subprocess(
        [PYTHON, "scripts/run_offensive_probe.py", "--tag", arm],
        log_path=log_path,
        env=env,
        timeout=1800,
    )
    off_report = _newest("offensive_*.json", timeout_start)
    if off_report:
        off_data = json.loads(Path(off_report).read_text(encoding="utf-8"))
        results["offensive_probe"] = {
            "refusal_rate": off_data.get("refusal_rate", None),
            "report": off_report,
        }
    else:
        results["offensive_probe"] = {"error": f"rc={rc}, no report found"}

    # 3. Context recall
    if not skip_recall:
        _log("Running context recall benchmark ...")
        log_path = LOGS / f"{arm}_recall.log"
        rc = _run_subprocess(
            [
                PYTHON, "scripts/bench_context_recall.py",
                "--seq-lengths", "8192", "16384", "27000",
                "--fracs", "0.85",
                "--depths", "0.0", "0.5", "1.0",
            ],
            log_path=log_path,
            env=env,
            timeout=7200,
        )
        recall_report = _newest("context_recall_*.json", timeout_start)
        if recall_report:
            recall_data = json.loads(Path(recall_report).read_text(encoding="utf-8"))
            # Import _recall_pct from model_bakeoff
            from scripts.model_bakeoff import _recall_pct
            results["context_recall"] = {
                **_recall_pct(recall_data),
                "report": recall_report,
            }
        else:
            results["context_recall"] = {"error": f"rc={rc}, no report found"}

    # 4. Golden
    if not skip_golden:
        _log("Running golden benchmark ...")
        log_path = LOGS / f"{arm}_golden.log"
        judge_arg = ["--judge", judge] if judge else []
        rc = _run_subprocess(
            [PYTHON, "scripts/run_golden.py", *judge_arg],
            log_path=log_path,
            env=env,
            timeout=7200,
        )
        golden_report = _newest("golden_*.json", timeout_start)
        if golden_report:
            golden_data = json.loads(Path(golden_report).read_text(encoding="utf-8"))
            from scripts.model_bakeoff import _golden_scores
            results["golden"] = {
                **_golden_scores(golden_data),
                "report": golden_report,
            }
        else:
            results["golden"] = {"error": f"rc={rc}, no report found"}

    _log(f"Quality results for '{arm}' written.")
    return results


def cmd_vision(arm: str) -> Dict:
    """Describe 3 synthetic images, check keywords, time it."""
    arm_def = ARM_DEFS[arm]

    # Generate 3 synthetic images as base64
    images: List[Dict[str, Any]] = []

    # Import PIL once at the top of the function
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        images = [
            {"name": "red circle", "b64": "", "error": "PIL not installed"},
            {"name": "KAYA text", "b64": "", "error": "PIL not installed"},
            {"name": "blue bars", "b64": "", "error": "PIL not installed"},
        ]
        results["images_generated"] = False
        _log("vision: PIL not available, skipping image generation.")
        return results

    # 1. Red filled circle on white
    try:
        img1 = Image.new("RGB", (256, 256), "white")
        draw1 = ImageDraw.Draw(img1)
        draw1.ellipse([40, 40, 216, 216], fill="red", outline="darkred")
        buf1 = io.BytesIO()
        img1.save(buf1, format="PNG")
        images.append({
            "name": "red circle",
            "b64": base64.b64encode(buf1.getvalue()).decode("ascii"),
        })
    except Exception as exc:
        _log(f"WARNING: could not generate circle image: {exc}")
        images.append({"name": "red circle", "b64": "", "error": str(exc)})

    # 2. Black text "KAYA 2026" on white
    try:
        img2 = Image.new("RGB", (512, 128), "white")
        draw2 = ImageDraw.Draw(img2)
        try:
            font = ImageFont.load_default(size=64)
        except TypeError:
            font = ImageFont.load_default()
        draw2.text((80, 30), "KAYA 2026", fill="black", font=font)
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        images.append({
            "name": "KAYA text",
            "b64": base64.b64encode(buf2.getvalue()).decode("ascii"),
        })
    except Exception as exc:
        _log(f"WARNING: could not generate text image: {exc}")
        images.append({"name": "KAYA text", "b64": "", "error": str(exc)})

    # 3. Three blue vertical bars of increasing height
    try:
        img3 = Image.new("RGB", (256, 256), "white")
        draw3 = ImageDraw.Draw(img3)
        bar_width = 30
        heights = [80, 140, 200]
        for idx, height in enumerate(heights):
            x = 40 + idx * 60
            draw3.rectangle([x, 256 - height, x + bar_width, 256], fill="blue")
        buf3 = io.BytesIO()
        img3.save(buf3, format="PNG")
        images.append({
            "name": "blue bars",
            "b64": base64.b64encode(buf3.getvalue()).decode("ascii"),
        })
    except Exception as exc:
        _log(f"WARNING: could not generate bars image: {exc}")
        images.append({"name": "blue bars", "b64": "", "error": str(exc)})

    results: Dict[str, Any] = {}

    if arm_def["type"] == "ollama" and arm_def["model"] == "kaya-q6":
        # Ollama Q6 is text-only; record unsupported
        results["supported"] = False
        results["note"] = "kaya-q6 is text-only (no vision)"
        _log("vision: kaya-q6 is text-only, skipped.")
        return results

    # Ask each image to be described
    for img_info in images:
        img_name = img_info["name"]
        b64 = img_info.get("b64", "")
        if not b64:
            results[img_name] = {"error": "could not generate image"}
            continue

        keywords = _IMAGE_KEYWORDS.get(img_name, [])
        question = "Descreve esta imagem numa frase."

        t0 = time.perf_counter()
        response_text = ""

        try:
            if arm_def["type"] == "llama":
                # POST {base}/v1/chat/completions
                url = f"{arm_def['base_url']}/v1/chat/completions"
                payload = {
                    "model": arm_def["model"],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 120,
                    "temperature": 0,
                }
                resp = requests.post(url, json=payload, timeout=_timeout_for_arm(arm))
                resp.raise_for_status()
                response_text = resp.json()["choices"][0]["message"]["content"]
            elif arm_def["type"] == "ollama":
                # POST /api/chat
                url = f"{arm_def['base_url']}/api/chat"
                payload = {
                    "model": arm_def["model"],
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                            "images": [b64],
                        }
                    ],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 120,
                    },
                }
                resp = requests.post(url, json=payload, timeout=_timeout_for_arm(arm))
                resp.raise_for_status()
                response_text = resp.json()["message"]["content"]
        except Exception as exc:
            response_text = f"ERROR: {exc}"

        elapsed = time.perf_counter() - t0

        # Check keywords
        keywords_ok = False
        if keywords and response_text:
            response_lower = response_text.lower()
            if all(any(kw in response_lower for kw in group) for group in keywords):
                keywords_ok = True

        results[img_name] = {
            "text": response_text,
            "keywords_ok": keywords_ok,
            "seconds": round(elapsed, 2),
        }

    _log(f"Vision results for '{arm}' written.")
    return results


def cmd_embed() -> Dict:
    """bge-m3 on CPU vs GPU0."""
    # Set CUDA_VISIBLE_DEVICES before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU0_UUID

    import torch
    from sentence_transformers import SentenceTransformer

    results: Dict[str, Any] = {}
    results["cpu_count"] = os.cpu_count()

    # Short Portuguese questions for encoding tests.
    questions = [
        "O que é que o grupo fez ontem?",
        "Quem é que ganhou o jogo de futebol?",
        "Quando é a próxima viagem do grupo?",
        "O que faz o Gil no trabalho?",
        "Quem é que editou o vídeo do concerto?",
        "O Pedro tem alguma startup?",
        "Que banda é que tocou no festival?",
        "Onde é que foi o jantar de Natal?",
        "Quem é que trouxe a sobremesa?",
        "O que é que o Bana disse sobre o artigo?",
    ]
    # Repeat to get 50 queries
    queries = (questions * 5)[:50]

    # CPU
    _log("Embed: loading SentenceTransformer on CPU ...")
    t0 = time.perf_counter()
    cpu_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    cpu_load_s = round(time.perf_counter() - t0, 2)
    results["cpu_load_s"] = cpu_load_s

    # 50 single-query encodes
    _log("Embed: 50 single queries on CPU ...")
    cpu_times: List[float] = []
    for _i in range(50):
        t1 = time.perf_counter()
        cpu_model.encode(queries[_i % len(queries)], show_progress_bar=False)
        cpu_times.append((time.perf_counter() - t1) * 1000)  # ms

    # One batch of 200 ~80-word chunks
    _log("Embed: batch of 200 on CPU ...")
    t1 = time.perf_counter()
    cpu_model.encode(queries * 4, show_progress_bar=False)  # 50*4=200
    cpu_batch_s = round(time.perf_counter() - t1, 2)
    results["cpu"] = {
        "load_s": cpu_load_s,
        "single_query_ms": {
            "p50": round(sorted(cpu_times)[24] * 10, 1),
            "p95": round(sorted(cpu_times)[47] * 10, 1),
        },
        "batch_200_s": cpu_batch_s,
        "chunks_per_s": round(200 / max(cpu_batch_s, 0.001), 1),
    }

    # GPU
    _log("Embed: loading SentenceTransformer on GPU ...")
    t0 = time.perf_counter()
    gpu_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    gpu_load_s = round(time.perf_counter() - t0, 2)
    results["gpu_load_s"] = gpu_load_s

    # 50 single-query encodes
    _log("Embed: 50 single queries on GPU ...")
    gpu_times: List[float] = []
    for _i in range(50):
        t1 = time.perf_counter()
        gpu_model.encode(queries[_i % len(queries)], show_progress_bar=False)
        gpu_times.append((time.perf_counter() - t1) * 1000)  # ms

    # One batch of 200
    _log("Embed: batch of 200 on GPU ...")
    t1 = time.perf_counter()
    gpu_model.encode(queries * 4, show_progress_bar=False)
    gpu_batch_s = round(time.perf_counter() - t1, 2)
    results["gpu"] = {
        "load_s": gpu_load_s,
        "single_query_ms": {
            "p50": round(sorted(gpu_times)[24] * 10, 1),
            "p95": round(sorted(gpu_times)[47] * 10, 1),
        },
        "batch_200_s": gpu_batch_s,
        "chunks_per_s": round(200 / max(gpu_batch_s, 0.001), 1),
    }

    _log("Embed results written.")
    return results


def cmd_whisper() -> Dict:
    """faster-whisper large-v3 load/unload/transcribe on GPU0."""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU0_UUID

    from src.config_loader import load_config

    config = load_config("config.yaml")

    # Try to synthesize Portuguese speech
    from src.chat.tts import synthesize_wav
    _log("Whisper: synthesising Portuguese speech ...")
    sample_text = (
        "O grupo de amigos saiu para jantar ao centro de Lisboa. "
        "Foram para um restaurante de cozinha tradicional portuguesa "
        "e comeram bacalhau à brás. Depois foram tomar uma copa ao Bairro Alto."
    )
    wav_bytes = synthesize_wav(sample_text, config)
    if wav_bytes is None:
        return {"skipped": "no TTS voice"}

    wav_duration_s = len(wav_bytes) / 16000.0  # approximate (16 kHz mono)

    # GPU0 memory used before
    try:
        mem_before = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        for line in mem_before.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and GPU0_UUID in parts[0]:
                vram_before_mib = int(parts[1])
                break
        else:
            vram_before_mib = None
    except Exception:
        vram_before_mib = None

    # Load faster-whisper
    from faster_whisper import WhisperModel

    _log("Whisper: loading large-v3 (first load) ...")
    t0 = time.perf_counter()
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    load_s_1 = round(time.perf_counter() - t0, 2)

    try:
        mem_after = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        for line in mem_after.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and GPU0_UUID in parts[0]:
                vram_after_load_mib = int(parts[1])
                break
        else:
            vram_after_load_mib = None
    except Exception:
        vram_after_load_mib = None

    # Transcribe
    _log("Whisper: transcribing ...")
    wav_file = io.BytesIO(wav_bytes)
    t1 = time.perf_counter()
    segments, info = model.transcribe(wav_file, language="pt", vad_filter=True, beam_size=1)
    transcribed_text = " ".join(seg.text for seg in segments)
    transcribe_s = round(time.perf_counter() - t1, 2)

    # Unload
    _log("Whisper: unloading ...")
    del model
    gc.collect()
    try:
        mem_after_unload = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        for line in mem_after_unload.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and GPU0_UUID in parts[0]:
                vram_after_unload_mib = int(parts[1])
                break
        else:
            vram_after_unload_mib = None
    except Exception:
        vram_after_unload_mib = None

    # Second load (warm-page-cache)
    _log("Whisper: loading large-v3 again (warm cache) ...")
    t0 = time.perf_counter()
    model2 = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    load_s_2 = round(time.perf_counter() - t0, 2)
    del model2
    gc.collect()

    return {
        "wav_duration_s": round(wav_duration_s, 2),
        "vram_before_mib": vram_before_mib,
        "load_s_first": load_s_1,
        "vram_after_load_mib": vram_after_load_mib,
        "transcribe_s": transcribe_s,
        "transcribed_text": transcribed_text,
        "vram_after_unload_mib": vram_after_unload_mib,
        "load_s_warm": load_s_2,
    }


def cmd_storage() -> Dict:
    """Estimate the Pi journal's size per week."""
    live_dir = Path.home() / "kaya-prod" / "data" / "live_messages"
    documents_dir = Path.home() / "kaya-prod" / "data" / "documents"

    results: Dict[str, Any] = {}

    if not live_dir.exists():
        results["error"] = f"{live_dir} does not exist"
        return results

    # Read JSONL files
    lines: List[str] = []
    for jsonl_file in live_dir.iterdir():
        name = jsonl_file.name
        if ".bak" in name or ".migrated" in name:
            continue
        if jsonl_file.suffix != ".jsonl":
            continue
        try:
            lines.extend(jsonl_file.read_text(encoding="utf-8").splitlines())
        except OSError:
            pass

    if not lines:
        results["error"] = "No JSONL files found"
        return results

    # Parse dates and group by day
    from collections import defaultdict
    day_lines: Dict[str, List[str]] = defaultdict(list)
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = obj.get("timestamp", obj.get("date", ""))
        if ts:
            if isinstance(ts, (int, float)):
                day_key = datetime.date.fromtimestamp(ts).isoformat()
            else:
                day_key = str(ts)[:10]  # YYYY-MM-DD
            day_lines[day_key].append(line)

    if not day_lines:
        results["error"] = "Could not parse dates"
        return results

    # Last 30 days
    today = datetime.date.today()
    recent_days: Dict[str, List[str]] = {}
    for day_key in sorted(day_lines.keys()):
        try:
            day_date = datetime.date.fromisoformat(day_key)
        except ValueError:
            continue
        delta = (today - day_date).days
        if 0 <= delta < 30:
            recent_days[day_key] = day_lines[day_key]

    if not recent_days:
        results["error"] = "No messages in last 30 days"
        return results

    # Messages per day
    msgs_per_day = [len(v) for v in recent_days.values()]
    mean_msgs = round(sum(msgs_per_day) / len(msgs_per_day), 1)
    max_msgs = max(msgs_per_day)

    # Mean JSON line length
    line_lengths = [len(line.encode("utf-8")) for line in lines if line.strip()]
    mean_line_len = round(sum(line_lengths) / len(line_lengths), 1) if line_lengths else 0

    # Count images and documents per day
    total_images = 0
    total_docs = 0
    for day_msgs in recent_days.values():
        for line in day_msgs:
            if "[Imagem:" in line:
                total_images += 1
            if "[Documento:" in line:
                total_docs += 1

    images_per_day = round(total_images / max(len(recent_days), 1), 1)
    docs_per_day = round(total_docs / max(len(recent_days), 1), 1)

    # Document sizes
    doc_total_bytes = 0
    doc_count = 0
    if documents_dir.exists():
        for doc_file in documents_dir.rglob("*"):
            if doc_file.is_file():
                doc_total_bytes += doc_file.stat().st_size
                doc_count += 1
    mean_doc_size = doc_total_bytes / max(doc_count, 1)

    # Estimate weekly journal
    msgs_per_week = mean_msgs * 7
    images_per_week = images_per_day * 7
    docs_per_week = docs_per_day * 7
    voice_per_week = round(msgs_per_week * 0.1, 1)  # 10% voice notes

    # Components in bytes
    journal_text_bytes = msgs_per_week * mean_line_len * 3  # raw WAHA event ~3x
    journal_images_bytes = images_per_week * 250_000  # 250 KB
    journal_docs_bytes = docs_per_week * mean_doc_size
    journal_voice_bytes = voice_per_week * 60_000  # 60 KB

    total_bytes = journal_text_bytes + journal_images_bytes + journal_docs_bytes + journal_voice_bytes

    results["messages_per_day"] = {"mean": mean_msgs, "max": max_msgs}
    results["mean_json_line_length_bytes"] = mean_line_len
    results["images_per_day"] = images_per_day
    results["documents_per_day"] = docs_per_day
    results["mean_document_size_bytes"] = round(mean_doc_size, 1)
    results["document_file_count"] = doc_count
    results["document_total_bytes"] = doc_total_bytes
    results["weekly_estimate"] = {
        "messages": round(msgs_per_week, 1),
        "images": images_per_week,
        "documents": docs_per_week,
        "voice_notes": voice_per_week,
        "components_bytes": {
            "journal_text": round(journal_text_bytes, 0),
            "images": round(journal_images_bytes, 0),
            "documents": round(journal_docs_bytes, 0),
            "voice_notes": round(journal_voice_bytes, 0),
        },
        "total_mb": round(total_bytes / (1024 * 1024), 1),
    }
    return results


def cmd_report() -> str:
    """Aggregate results into a Markdown report. Returns report path."""
    report = _load_report()
    stamp = args.stamp
    md_path = str(REPORTS / f"runtime_{stamp}.md")

    sections: List[str] = []

    # --- Latency table ---
    if "latency" in report:
        sections.append("## Latency\n")
        for arm_name, arm_data in report["latency"].items():
            sections.append(f"### {arm_name}\n")
            sections.append("| Metric | Value |")
            sections.append("|---|---|")
            if "prompt_chars" in arm_data:
                sections.append(f"| Prompt chars | {arm_data['prompt_chars']} |")
            if "vram_used_mib" in arm_data:
                sections.append(f"| VRAM used (MiB) | {arm_data['vram_used_mib']} |")
            for key in ["warm_overhead_n1", "warm_overhead_n64"]:
                if key in arm_data:
                    wd = arm_data[key]
                    sections.append(f"| {key} ttft p50 (ms) | {wd.get('ttft_ms', {}).get('p50')} |")
                    sections.append(f"| {key} ttft p95 (ms) | {wd.get('ttft_ms', {}).get('p95')} |")
                    sections.append(f"| {key} total p50 (ms) | {wd.get('total_ms', {}).get('p50')} |")
                    sections.append(f"| {key} total p95 (ms) | {wd.get('total_ms', {}).get('p95')} |")
            if "throughput_tok_per_sec" in arm_data:
                tp = arm_data["throughput_tok_per_sec"]
                sections.append(
                    f"| Throughput (tok/s) | {tp.get('mean')} "
                    f"(samples: {tp.get('samples')}) |"
                )
            if "cold_start" in arm_data and isinstance(arm_data["cold_start"], dict):
                sections.append(
                    f"| Cold start median (s) | {arm_data['cold_start'].get('median_s')} |"
                )
                sections.append(
                    f"| Cold extra (s) | {arm_data.get('cold_extra_s')} |"
                )
            if "cold_start_drop_caches" in arm_data:
                dc = arm_data["cold_start_drop_caches"]
                if isinstance(dc, dict):
                    sections.append(
                        f"| Cold + drop_caches median (s) | {dc.get('median_s')} |"
                    )
                else:
                    sections.append(f"| Cold + drop_caches | {dc} |")
        sections.append("")

    # --- Quality table ---
    if "quality" in report:
        sections.append("## Quality\n")
        for arm_name, arm_data in report["quality"].items():
            sections.append(f"### {arm_name}\n")
            sections.append("| Metric | Value |")
            sections.append("|---|---|")
            conv = arm_data.get("conversation_probe", {})
            if isinstance(conv, dict):
                sections.append(f"| Median turn (s) | {conv.get('median_seconds')} |")
                sections.append(f"| P90 turn (s) | {conv.get('p90_seconds')} |")
            off = arm_data.get("offensive_probe", {})
            if isinstance(off, dict):
                sections.append(f"| Refusal rate | {off.get('refusal_rate')} |")
            golden = arm_data.get("golden", {})
            if isinstance(golden, dict):
                sections.append(
                    f"| Golden extended_average | {golden.get('extended_average')} |"
                )
            recall = arm_data.get("context_recall", {})
            if isinstance(recall, dict):
                sections.append(f"| Recall envelope % | {recall.get('envelope_pct')} |")
        sections.append("")

    # --- Vision ---
    if "vision" in report:
        sections.append("## Vision\n")
        for arm_name, arm_data in report["vision"].items():
            if arm_name == "supported" or arm_name == "note":
                continue
            sections.append(f"### {arm_name}\n")
            sections.append("| Image | Text | Keywords OK | Seconds |")
            sections.append("|---|---|---|---|")
            if isinstance(arm_data, dict):
                sections.append(
                    f"| {arm_name} | {arm_data.get('text', '')[:80]} "
                    f"{'...' if isinstance(arm_data.get('text'), str) and len(arm_data.get('text', '') or '') > 80 else ''} "
                    f"| {arm_data.get('keywords_ok')} | {arm_data.get('seconds')} |"
                )
        sections.append("")

    # --- Embed ---
    if "embed" in report:
        sections.append("## Embedding (bge-m3)\n")
        embed_data = report["embed"]
        sections.append("| Platform | Load (s) | Single p50 (ms) | Single p95 (ms) | Batch 200 (s) | Chunks/s |")
        sections.append("|---|---|---|---|---|---|")
        for platform in ["cpu", "gpu"]:
            if platform in embed_data:
                pd = embed_data[platform]
                sections.append(
                    f"| {platform} | {pd.get('load_s')} | "
                    f"{pd.get('single_query_ms', {}).get('p50')} | "
                    f"{pd.get('single_query_ms', {}).get('p95')} | "
                    f"{pd.get('batch_200_s')} | "
                    f"{pd.get('chunks_per_s')} |"
                )
        sections.append("")

    # --- Whisper ---
    if "whisper" in report:
        sections.append("## Whisper (large-v3)\n")
        wh = report["whisper"]
        sections.append("| Metric | Value |")
        sections.append("|---|---|")
        if "skipped" in wh:
            sections.append(f"| Skipped | {wh['skipped']} |")
        else:
            sections.append(f"| WAV duration (s) | {wh.get('wav_duration_s')} |")
            sections.append(f"| Load (first) (s) | {wh.get('load_s_first')} |")
            sections.append(f"| Load (warm) (s) | {wh.get('load_s_warm')} |")
            sections.append(f"| Transcribe (s) | {wh.get('transcribe_s')} |")
            sections.append(f"| Transcribed text | {wh.get('transcribed_text', '')[:120]} |")
            if wh.get("vram_before_mib"):
                sections.append(f"| VRAM before (MiB) | {wh['vram_before_mib']} |")
            if wh.get("vram_after_load_mib"):
                sections.append(f"| VRAM after load (MiB) | {wh['vram_after_load_mib']} |")
            if wh.get("vram_after_unload_mib"):
                sections.append(f"| VRAM after unload (MiB) | {wh['vram_after_unload_mib']} |")
        sections.append("")

    # --- Storage ---
    if "storage" in report:
        sections.append("## Storage Estimate (Pi journal)\n")
        st = report["storage"]
        if "weekly_estimate" in st:
            we = st["weekly_estimate"]
            sections.append("| Component | Estimate |")
            sections.append("|---|---|")
            sections.append(f"| Messages/day (mean/max) | {st.get('messages_per_day', {}).get('mean')}/{st.get('messages_per_day', {}).get('max')} |")
            sections.append(f"| Images/day | {st.get('images_per_day')} |")
            sections.append(f"| Documents/day | {st.get('documents_per_day')} |")
            components = we.get("components_bytes", {})
            sections.append(f"|  - Journal text/week | {components.get('journal_text', 0) / (1024*1024):.1f} MB |")
            sections.append(f"|  - Images/week | {components.get('images', 0) / (1024*1024):.1f} MB |")
            sections.append(f"|  - Documents/week | {components.get('documents', 0) / (1024*1024):.1f} MB |")
            sections.append(f"|  - Voice notes/week | {components.get('voice_notes', 0) / (1024*1024):.1f} MB |")
            sections.append(f"| **Total/week** | **{we.get('total_mb')} MB** |")
        sections.append("")

    # --- Decision inputs ---
    sections.append("## Decision Inputs\n")

    # swap warm overhead vs direct
    latency = report.get("latency", {})
    direct_latency = latency.get("direct", {})
    swap_latency = latency.get("swap", {})

    if "warm_overhead_n1" in direct_latency and "warm_overhead_n1" in swap_latency:
        direct_p50 = direct_latency["warm_overhead_n1"].get("ttft_ms", {}).get("p50", 0) or 0
        swap_p50 = swap_latency["warm_overhead_n1"].get("ttft_ms", {}).get("p50", 0) or 0
        sections.append(
            f"- Swap warm overhead vs direct (ttft p50, n=1): "
            f"{swap_p50:.1f} ms vs {direct_p50:.1f} ms "
            f"(delta: {swap_p50 - direct_p50:+.1f} ms)"
        )

    # swap cold start
    if "cold_extra_s" in swap_latency:
        sections.append(
            f"- Swap cold start (extra over warm): {swap_latency['cold_extra_s']:.4f} s"
        )

    # Ollama arms vs direct: turn latency delta
    quality = report.get("quality", {})
    direct_quality = quality.get("direct", {})
    for arm_name in ["ollama-q8", "ollama-q6"]:
        arm_q = quality.get(arm_name, {})
        if isinstance(arm_q, dict):
            direct_turn = direct_quality.get("conversation_probe", {}).get("median_seconds")
            arm_turn = arm_q.get("conversation_probe", {}).get("median_seconds")
            if direct_turn and arm_turn and direct_turn > 0:
                delta_pct = ((arm_turn - direct_turn) / direct_turn) * 100
                sections.append(
                    f"- {arm_name} turn latency delta vs direct: "
                    f"{delta_pct:+.1f}% ({arm_turn:.2f}s vs {direct_turn:.2f}s)"
                )

    # Golden delta vs noise band
    for arm_name in ["ollama-q8", "ollama-q6"]:
        arm_q = quality.get(arm_name, {})
        if isinstance(arm_q, dict):
            direct_golden = direct_quality.get("golden", {}).get("extended_average")
            arm_golden = arm_q.get("golden", {}).get("extended_average")
            if direct_golden and arm_golden:
                delta = arm_golden - direct_golden
                noise_band = 0.07
                verdict = "within noise band" if abs(delta) <= noise_band else "outside noise band"
                sections.append(
                    f"- {arm_name} golden delta vs direct: {delta:+.4f} "
                    f"(noise band +/-{noise_band}) -> {verdict}"
                )

    # Recall delta
    for arm_name in ["ollama-q8", "ollama-q6"]:
        arm_q = quality.get(arm_name, {})
        if isinstance(arm_q, dict):
            direct_recall = direct_quality.get("context_recall", {}).get("envelope_pct")
            arm_recall = arm_q.get("context_recall", {}).get("envelope_pct")
            if direct_recall is not None and arm_recall is not None:
                delta = arm_recall - direct_recall
                sections.append(
                    f"- {arm_name} recall delta vs direct: {delta:+.1f} pp "
                    f"({arm_recall}% vs {direct_recall}%)"
                )

    # Refusal delta
    for arm_name in ["ollama-q8", "ollama-q6"]:
        arm_q = quality.get(arm_name, {})
        if isinstance(arm_q, dict):
            direct_refusal = direct_quality.get("offensive_probe", {}).get("refusal_rate")
            arm_refusal = arm_q.get("offensive_probe", {}).get("refusal_rate")
            if direct_refusal is not None and arm_refusal is not None:
                delta = arm_refusal - direct_refusal
                sections.append(
                    f"- {arm_name} refusal delta vs direct: {delta:+.4f}"
                )

    # Vision keyword hits
    vision = report.get("vision", {})
    for arm_name in ["ollama-q8", "ollama-q6"]:
        arm_v = vision.get(arm_name, {})
        if isinstance(arm_v, dict) and arm_v.get("supported"):
            sections.append(f"- {arm_name}: vision not supported (text-only)")
        elif isinstance(arm_v, dict):
            hits = sum(1 for v in arm_v.values() if isinstance(v, dict) and v.get("keywords_ok"))
            sections.append(f"- {arm_name}: {hits}/3 image keyword checks passed")

    md_content = "\n".join(sections)
    md_path = str(REPORTS / f"runtime_{stamp}.md")
    REPORTS.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text(md_content, encoding="utf-8")
    _log(f"Report written to {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_stamp() -> str:
    """Resolve the --stamp argument: newest existing, new, or forced new."""
    if args.stamp:
        return args.stamp
    if args.new:
        return _now_stamp()
    # Find newest runtime_*.json
    hits = glob.glob(str(REPORTS / "runtime_*.json"))
    if hits:
        newest_path = max(hits, key=os.path.getmtime)
        # Extract stamp from filename
        basename = os.path.basename(newest_path)
        match = re.search(r"runtime_(.+)\.json", basename)
        if match:
            return match.group(1)
    return _now_stamp()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark runtime serving options for KayaChatBot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/bench_runtime.py up direct\n"
            "  python scripts/bench_runtime.py latency direct\n"
            "  python scripts/bench_runtime.py quality ollama-q8 --judge azure\n"
            "  python scripts/bench_runtime.py vision ollama-q8\n"
            "  python scripts/bench_runtime.py embed\n"
            "  python scripts/bench_runtime.py whisper\n"
            "  python scripts/bench_runtime.py storage\n"
            "  python scripts/bench_runtime.py report\n"
        ),
    )
    parser.add_argument(
        "command",
        choices=["up", "down", "latency", "quality", "vision", "embed",
                 "whisper", "storage", "report"],
        help="Command to run.",
    )
    parser.add_argument(
        "arm",
        nargs="?",
        choices=list(ARM_DEFS.keys()),
        help="Serving arm (direct, swap, ollama-q8, ollama-q6). Required for up/latency/quality/vision.",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Report stamp. Defaults to newest runtime_*.json, or a new UTC stamp.",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Force a new UTC stamp even if a report exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute a section even if it already exists in the report.",
    )
    parser.add_argument(
        "--judge",
        default="azure",
        choices=["azure", "xai"],
        help="Judge provider for quality harnesses (default: azure).",
    )
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="Skip the golden benchmark in quality mode.",
    )
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help="Skip the context recall benchmark in quality mode.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    global args
    parser = build_parser()
    args = parser.parse_args()

    # Resolve stamp
    args.stamp = resolve_stamp()
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    command = args.command

    # Load existing report
    report = _load_report()

    if command == "up":
        if not args.arm:
            parser.error("up requires an ARM argument.")
        cmd_up(args.arm)

    elif command == "down":
        cmd_down()

    elif command == "latency":
        if not args.arm:
            parser.error("latency requires an ARM argument.")
        if _check_resumable(f"latency.{args.arm}"):
            _log(f"latency.{args.arm} already in report (use --force to recompute).")
            return
        result = cmd_latency(args.arm)
        report.setdefault("latency", {})[args.arm] = result
        _save_report(report)

    elif command == "quality":
        if not args.arm:
            parser.error("quality requires an ARM argument.")
        if _check_resumable(f"quality.{args.arm}"):
            _log(f"quality.{args.arm} already in report (use --force to recompute).")
            return
        result = cmd_quality(
            args.arm,
            judge=args.judge,
            skip_golden=args.skip_golden,
            skip_recall=args.skip_recall,
        )
        report.setdefault("quality", {})[args.arm] = result
        _save_report(report)

    elif command == "vision":
        if not args.arm:
            parser.error("vision requires an ARM argument.")
        if _check_resumable(f"vision.{args.arm}"):
            _log(f"vision.{args.arm} already in report (use --force to recompute).")
            return
        result = cmd_vision(args.arm)
        report.setdefault("vision", {})[args.arm] = result
        _save_report(report)

    elif command == "embed":
        if _check_resumable("embed"):
            _log(f"embed already in report (use --force to recompute).")
            return
        result = cmd_embed()
        report["embed"] = result
        _save_report(report)

    elif command == "whisper":
        if _check_resumable("whisper"):
            _log(f"whisper already in report (use --force to recompute).")
            return
        result = cmd_whisper()
        report["whisper"] = result
        _save_report(report)

    elif command == "storage":
        if _check_resumable("storage"):
            _log(f"storage already in report (use --force to recompute).")
            return
        result = cmd_storage()
        report["storage"] = result
        _save_report(report)

    elif command == "report":
        md_path = cmd_report()
        _log(f"Done. Markdown report: {md_path}")


if __name__ == "__main__":
    main()
