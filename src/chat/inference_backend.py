"""Pluggable generation backend: in-process HF model vs a llama.cpp server.

Every generation site (``engine.generate_reply``, the web UI stream,
``suggestions``, the CLI) builds a ``messages`` list and asks a backend to turn it
into text. Two backends implement the same interface:

- ``HFBackend`` — the current path: Unsloth ``FastModel`` / PEFT model running
  in-process on the GPU. Behaviour is byte-for-byte the previous code.
- ``LlamaCppBackend`` — sends the chat-templated prompt to a llama.cpp
  ``llama-server`` over HTTP. ~15x faster generation at Q6_K parity quality; the
  heavy model lives in the server (a separate process/container), so this process
  only needs the tokenizer (for templating) and the RAG retriever.

Selected by ``inference.backend`` in ``config.yaml`` (default ``hf``), so turning
the migration on is a one-line config flip and ``hf`` stays the current path.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any, Dict, Iterator, List

import requests


def resolve_backend(config: Dict[str, Any]) -> str:
    """The active backend: ``KAYA_INFERENCE_BACKEND`` env wins, else config, else hf.

    The env override lets prod run ``gguf`` (set in the compose service) while the
    committed config default stays ``hf`` for local dev.
    """
    return (
        os.environ.get("KAYA_INFERENCE_BACKEND")
        or config.get("inference", {}).get("backend")
        or "hf"
    ).lower()


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


class HFBackend(InferenceBackend):
    """In-process HF/Unsloth model (current behaviour). GPU-resident."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _gen_kwargs(self, max_new_tokens: int, sampling: Dict[str, Any]) -> Dict[str, Any]:
        temperature = sampling.get("temperature", 1.0)
        return dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=sampling.get("top_p", 0.95),
            top_k=sampling.get("top_k", 64),
            repetition_penalty=sampling.get("repetition_penalty", 1.0),
            no_repeat_ngram_size=sampling.get("no_repeat_ngram_size", 0),
            use_cache=True,
        )

    def _inputs(self, messages):
        prompt = _templated_prompt(self.tokenizer, messages)
        return self.tokenizer(text=[prompt], return_tensors="pt").to(self.device)

    def generate(self, messages, *, max_new_tokens, sampling):
        inputs = self._inputs(messages)
        outputs = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens, sampling))
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_stream(self, messages, *, max_new_tokens, sampling):
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0
        )
        gen_kwargs = dict(**inputs, streamer=streamer, **self._gen_kwargs(max_new_tokens, sampling))
        Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True).start()
        for token in streamer:
            yield token


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
            "cache_prompt": False,
            "stream": stream,
        }

    def generate(self, messages, *, max_new_tokens, sampling):
        resp = requests.post(
            f"{self.server_url}/completion",
            json=self._payload(messages, max_new_tokens, sampling, False),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("content", "")

    def generate_stream(self, messages, *, max_new_tokens, sampling):
        with requests.post(
            f"{self.server_url}/completion",
            json=self._payload(messages, max_new_tokens, sampling, True),
            timeout=self.timeout,
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


def build_backend(config: Dict[str, Any], model, tokenizer) -> InferenceBackend:
    """Pick the backend (``KAYA_INFERENCE_BACKEND`` env or ``inference.backend``)."""
    backend = resolve_backend(config)
    if backend == "gguf":
        gcfg = config.get("inference", {}).get("gguf", {})
        url = gcfg.get("server_url", "http://127.0.0.1:8080")
        print(f"✓ Inference backend: gguf (llama.cpp @ {url})")
        return LlamaCppBackend(tokenizer, url, timeout=gcfg.get("timeout", 180.0))
    print("✓ Inference backend: hf (in-process model)")
    return HFBackend(model, tokenizer)
