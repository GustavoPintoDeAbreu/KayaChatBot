"""Generation backends: a llama.cpp server or Ollama, both over HTTP.

Every generation site (``engine.generate_reply``, the web UI stream,
``suggestions``) builds a ``messages`` list and asks a backend to turn it into
text. Two backends implement the same interface:

- ``LlamaCppBackend`` — sends the chat-templated prompt to a llama.cpp
  ``llama-server`` (``/completion``).
- ``OllamaBackend`` — sends the same prompt to Ollama's ``/api/generate`` in raw
  mode; the serving path since 2026-09-25.

The weights live in the server; this process only needs the tokenizer (for
templating) and the RAG retriever. Selected by ``KAYA_INFERENCE_BACKEND`` or
``inference.backend`` (default ``gguf``). The in-process ``hf`` backend was
removed with fine-tuning on 2026-10-26.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Names Kaya to the GPU broker, so labwatch books her tokens to her (a bare
# python-requests is unattributable). Override with KAYA_USER_AGENT.
_UA = {"User-Agent": os.environ.get("KAYA_USER_AGENT", "kaya")}

# ``http://llm-broker:8080/upstream/kaya``: the shared llama-swap broker, which
# starts the named model on first request (see ~/llm-broker/README.md).
_BROKER_UPSTREAM = re.compile(r"^(?P<base>.+)/upstream/(?P<model>[^/]+)/?$")

BACKENDS = ("gguf", "ollama")


def resolve_backend(config: Dict[str, Any]) -> str:
    """The active backend: ``KAYA_INFERENCE_BACKEND`` env wins, else config, else gguf."""
    return (
        os.environ.get("KAYA_INFERENCE_BACKEND")
        or config.get("inference", {}).get("backend")
        or "gguf"
    ).lower()


def resolve_llama_url(config: Dict[str, Any]) -> str:
    """The llama-server URL: ``KAYA_LLAMA_URL`` env wins, else config, else localhost.

    The env override lets a benchmark arm point at the ``llama-bench`` server
    (127.0.0.1:8081) serving a candidate model, without editing the committed
    ``inference.gguf.server_url`` that prod resolves. Used by
    ``scripts/model_bakeoff.py``.
    """
    return (
        os.environ.get("KAYA_LLAMA_URL")
        or config.get("inference", {}).get("gguf", {}).get("server_url")
        or "http://127.0.0.1:8080"
    )


def resolve_ollama_url(config: Dict[str, Any]) -> str:
    """Ollama's URL: ``KAYA_OLLAMA_URL`` env wins, else config, else localhost."""
    return (os.environ.get("KAYA_OLLAMA_URL")
            or (config.get("inference", {}).get("ollama", {}) or {}).get("server_url")
            or "http://127.0.0.1:11434")


def resolve_ollama_model(config: Dict[str, Any]) -> str:
    """The Ollama model name: ``KAYA_OLLAMA_MODEL`` env wins, else config."""
    return (os.environ.get("KAYA_OLLAMA_MODEL")
            or (config.get("inference", {}).get("ollama", {}) or {}).get("model")
            or "gemma4:12b-it-q8_0")


def resolve_server_url(config: Dict[str, Any]) -> str:
    """Where the active runtime answers, for the OpenAI-style calls (vision, synopsis)."""
    if resolve_backend(config) == "ollama":
        return resolve_ollama_url(config)
    return resolve_llama_url(config)


def openai_chat_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fields every ``/v1/chat/completions`` call adds so either runtime answers.

    Gemma 4 thinks by default, and a thinking reply spends the whole token budget
    in a reasoning channel and comes back with empty content. llama.cpp is told
    through the chat template (``enable_thinking``), Ollama through
    ``reasoning_effort``; each ignores the other's switch, which is what lets one
    payload serve both. ``model`` is required by Ollama and ignored by llama.cpp.
    """
    return {
        "model": resolve_ollama_model(config),
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def broker_target(url: str) -> Optional[Tuple[str, str]]:
    """``(broker_base, model)`` when ``url`` is a broker upstream URL, else None."""
    match = _BROKER_UPSTREAM.match(url or "")
    return (match.group("base"), match.group("model")) if match else None


def record_activity(config: Dict[str, Any]) -> None:
    """Write the time of a GPU turn for the broker's GPU1 lending guards."""
    bcfg = (config.get("inference", {}) or {}).get("broker", {}) or {}
    activity_file = bcfg.get("activity_file", "data/kaya_last_active")
    try:
        Path(activity_file).parent.mkdir(parents=True, exist_ok=True)
        Path(activity_file).write_text(f"{time.time():.0f}\n")
    except OSError as exc:
        logger.warning("could not record activity in %s: %s", activity_file, exc)


def _borrowers(loaded, model: str, patterns) -> List[str]:
    """Loaded models other than Kaya that borrow GPU1. Entries are fnmatch patterns
    ("*-g1"), so a new borrower named by the broker's convention needs no change here."""
    return sorted(name for name in loaded
                  if name and name != model and any(fnmatch.fnmatchcase(name, p) for p in patterns))


def _loaded(base: str) -> set:
    running = requests.get(f"{base}/running", headers=_UA, timeout=5).json().get("running", []) or []
    return {entry.get("model") for entry in running if isinstance(entry, dict)}


_lent_cache: Dict[str, Any] = {"at": 0.0, "value": False}


def gpu1_lent(config: Dict[str, Any]) -> bool:
    """True while another model borrows GPU1 (an idea build, a coding session).

    Work that is not a reply (describing a photo nobody asked about, a bio
    refresh) must not take the card back then: the build would be thrown off
    and Kaya's idle clock restarted. An unreachable broker counts as lent, since
    calling Kaya then could make llama-swap swap her in. A plain llama-server is
    never lent. ``KAYA_GPU1_LENT=1|0`` forces the answer (the simulator).
    Cached for ``inference.broker.lent_cache_seconds`` (10).
    """
    forced = os.environ.get("KAYA_GPU1_LENT", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    target = broker_target(resolve_server_url(config))
    if target is None:
        return False
    bcfg = (config.get("inference", {}) or {}).get("broker", {}) or {}
    ttl = float(bcfg.get("lent_cache_seconds", 10))
    now = time.monotonic()
    if _lent_cache["at"] and now - _lent_cache["at"] < ttl:
        return _lent_cache["value"]
    base, model = target
    try:
        value = bool(_borrowers(_loaded(base), model, bcfg.get("evict_before_use", []) or []))
    except (requests.RequestException, ValueError) as exc:
        logger.warning("broker unreachable (%s); treating GPU1 as lent", exc)
        value = True
    _lent_cache.update(at=now, value=value)
    return value


def ensure_model_loaded(config: Dict[str, Any], urgent: bool = True) -> bool:
    """Make the serving model ready before GPU work. Best effort; never raises.

    ``urgent`` (a reply, or anything somebody is waiting for): behind the broker
    the model may be unloaded, or GPU1 may be lent to a coding model or a
    two-card model. llama-swap would evict those by itself when Kaya is
    requested, but it waits for their in-flight requests first, and a long agent
    turn could then hold a WhatsApp reply for minutes. So the borrowers are
    unloaded explicitly, then the model is loaded, and the time is recorded for
    the broker's ``llm-use-g1``/``llm-use-big`` guards, which lend GPU1 only after
    Kaya has been quiet for a while. Against a plain llama-server this only
    records the time. Returns True.

    Not urgent (enrichment nobody asked for): never evicts and never records
    activity (2026-10-26: only a reply takes the card back). Returns whether
    the model may be called now, i.e. GPU1 is not lent.
    """
    if not urgent:
        return not gpu1_lent(config)
    record_activity(config)
    target = broker_target(resolve_server_url(config))
    if target is None:
        return True
    base, model = target
    bcfg = (config.get("inference", {}) or {}).get("broker", {}) or {}
    ollama = resolve_backend(config) == "ollama"
    try:
        for borrower in _borrowers(_loaded(base), model, bcfg.get("evict_before_use", []) or []):
            logger.info("unloading %s to take GPU1 back", borrower)
            requests.post(f"{base}/api/models/unload/{borrower}", headers=_UA, timeout=60)
        _lent_cache.update(at=0.0, value=False)
        load_timeout = float(bcfg.get("load_timeout", 300))
        if ollama:
            # A generate with no prompt loads the model and returns; /api/version
            # would only start the server process, leaving the load to the reply.
            requests.post(f"{base}/upstream/{model}/api/generate",
                          json={"model": resolve_ollama_model(config)},
                          headers=_UA, timeout=load_timeout).raise_for_status()
        else:
            requests.get(f"{base}/upstream/{model}/health",
                         headers=_UA, timeout=load_timeout).raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("broker preparation failed (%s); generating anyway", exc)
    return True


def _templated_prompt(tokenizer, messages: List[Dict[str, str]], strip_bos: bool = False) -> str:
    """Apply the model's chat template. Optionally drop a leading BOS.

    llama.cpp auto-prepends BOS per the GGUF metadata; the HF template also emits
    a leading ``<bos>``. Sending both yields a double-BOS that measurably degrades
    quality, so the llama.cpp backend strips ours.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if strip_bos:
        bos = getattr(tokenizer, "bos_token", None)
        if bos and prompt.startswith(bos):
            prompt = prompt[len(bos):]
    return prompt


class InferenceBackend(ABC):
    """Turn a ``messages`` list into generated text (streaming or not)."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], *, max_new_tokens: int, sampling: Dict[str, Any]) -> str:
        ...

    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], *, max_new_tokens: int, sampling: Dict[str, Any]) -> Iterator[str]:
        ...


class LlamaCppBackend(InferenceBackend):
    """Generation via a llama.cpp ``llama-server`` HTTP endpoint."""

    def __init__(self, tokenizer, server_url: str, timeout: float = 180.0):
        self.tokenizer = tokenizer
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def _payload(self, messages, max_new_tokens, sampling, stream):
        return {
            "prompt": _templated_prompt(self.tokenizer, messages, strip_bos=True),
            "n_predict": max_new_tokens,
            "temperature": sampling.get("temperature", 1.0),
            "top_p": sampling.get("top_p", 0.95),
            "top_k": sampling.get("top_k", 64),
            # llama.cpp calls it repeat_penalty; it has no no_repeat_ngram_size.
            "repeat_penalty": sampling.get("repetition_penalty", 1.0),
            # Reuse the longest common prefix with the previous request on this
            # slot. It arrived as False with the backend and carried no reason —
            # a copied default, not a decision — and it was costing a full
            # re-prefill on every call, twice per turn.
            #
            # The router's system prompt is ~2,350 tokens and byte-identical on
            # EVERY message, including "😂", to produce a one-word label. Measured
            # against the live server with a shared 3,015-token prefix:
            # 3.01s -> 1.45s per call.
            #
            # --parallel 1 means one slot, so there is exactly one cached prefix
            # at a time and the two calls of a turn alternate. The router still
            # wins every time (its prefix is constant); the reply call wins only
            # when its own prefix happens to match, which for an open-ended turn
            # it deliberately does not — sample_facts reshuffles the profiles by
            # design. That is the anti-repetition fix and it is worth more than
            # the cache hit.
            "cache_prompt": True,
            "stream": stream,
        }

    def generate(self, messages, *, max_new_tokens, sampling):
        resp = requests.post(
            f"{self.server_url}/completion",
            json=self._payload(messages, max_new_tokens, sampling, False),
            headers=_UA, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("content", "")

    def generate_stream(self, messages, *, max_new_tokens, sampling):
        with requests.post(
            f"{self.server_url}/completion",
            json=self._payload(messages, max_new_tokens, sampling, True),
            headers=_UA, timeout=self.timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunk = obj.get("content", "")
                if chunk:
                    yield chunk
                if obj.get("stop"):
                    break


class OllamaBackend(InferenceBackend):
    """Generation via Ollama's ``/api/generate`` in raw mode.

    The serving path since 2026-09-25 (reports/benchmarks/runtime_20260924T232255Z.md):
    llama.cpp pays ~1.2 s to rebuild Gemma 4's sliding-window cache on every cache
    hit, twice per turn, and Ollama does not, so a turn went from 9.5 s to 4.6 s at
    equal quality. Raw mode sends the same HF-templated prompt ``LlamaCppBackend``
    sends, so the runtime is the only thing that changed. Every sampling option and
    the context size are passed explicitly: Ollama's defaults (a 4096 context among
    them) are not llama-server's. ``keep_alive`` -1 leaves unloading to the broker.
    """

    def __init__(self, tokenizer, server_url: str, model: str, timeout: float = 180.0,
                 num_ctx: int = 32768, keep_alive: str = "-1"):
        self.tokenizer = tokenizer
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        # Ollama takes a number of seconds or a duration ("30m"); the string "-1"
        # is a 400, so a bare number from config is sent as a number.
        self.keep_alive = int(keep_alive) if str(keep_alive).lstrip("-").isdigit() else keep_alive

    def _payload(self, messages, max_new_tokens, sampling, stream):
        return {
            "model": self.model,
            "prompt": _templated_prompt(self.tokenizer, messages, strip_bos=True),
            "raw": True,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": max_new_tokens,
                "temperature": sampling.get("temperature", 1.0),
                "top_p": sampling.get("top_p", 0.95),
                "top_k": sampling.get("top_k", 64),
                "repeat_penalty": sampling.get("repetition_penalty", 1.0),
            },
        }

    def generate(self, messages, *, max_new_tokens, sampling):
        resp = requests.post(
            f"{self.server_url}/api/generate",
            json=self._payload(messages, max_new_tokens, sampling, False),
            headers=_UA, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def generate_stream(self, messages, *, max_new_tokens, sampling):
        with requests.post(
            f"{self.server_url}/api/generate",
            json=self._payload(messages, max_new_tokens, sampling, True),
            headers=_UA, timeout=self.timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = obj.get("response", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break


def build_backend(config: Dict[str, Any], model, tokenizer) -> InferenceBackend:
    """Pick the backend (``KAYA_INFERENCE_BACKEND`` env or ``inference.backend``); unknown values raise."""
    backend = resolve_backend(config)
    if backend == "gguf":
        gcfg = config.get("inference", {}).get("gguf", {})
        url = resolve_llama_url(config)
        print(f"✓ Inference backend: gguf (llama.cpp @ {url})")
        return LlamaCppBackend(tokenizer, url, timeout=gcfg.get("timeout", 180.0))
    if backend == "ollama":
        ocfg = config.get("inference", {}).get("ollama", {}) or {}
        url, name = resolve_ollama_url(config), resolve_ollama_model(config)
        print(f"✓ Inference backend: ollama ({name} @ {url})")
        return OllamaBackend(tokenizer, url, name, timeout=ocfg.get("timeout", 180.0),
                             num_ctx=int(ocfg.get("num_ctx", 32768)),
                             keep_alive=str(ocfg.get("keep_alive", "-1")))
    raise ValueError(f"Unknown inference backend {backend!r}: expected one of {BACKENDS} "
                     "(the in-process 'hf' backend was removed)")
