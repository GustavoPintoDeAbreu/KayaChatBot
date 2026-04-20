"""
Offline Biography Generator & Multi-Model Comparison Tool
==========================================================
Generates structured biographical profiles for each Kaya group member using
multiple model backends, then produces a side-by-side comparison report.

Models supported:
  gemma4      — Local finetuned Gemma 4 E4B (Unsloth FastModel)
  qwen3       — Local finetuned Qwen3-14B (Unsloth FastLanguageModel)
  grok        — xAI Grok cloud API (xai_sdk)
  azure       — Azure OpenAI GPT-4.1-mini
  azure_gpt53 — Azure OpenAI GPT-5.3-chat (Responses API)

Local model loading uses Unsloth and auto-detects the base model class from
adapter_config.json so the correct loader is selected regardless of directory
name. One local model is loaded at a time (VRAM constraint, RTX 3090 24 GB).

Self-Censorship Notes
----------------------
  - Azure: content filters are managed at the Azure Portal resource level,
    not via API flags. --uncensored prepends the preamble to the system prompt,
    but the Azure deployment's filter policy is the real control point.
  - xAI/Grok: no safety API toggle; Grok is less restricted by design.
    --uncensored prepends the preamble to the system message, which may help.
  - Local (Gemma4, Qwen3): fully offline; --uncensored injects the preamble
    from config.yaml:chat.uncensored_system_prompt at generation time.

Usage
------
  python src/data/generate_bios_offline.py
  python src/data/generate_bios_offline.py --models gemma4,qwen3,grok,azure,azure_gpt53
  python src/data/generate_bios_offline.py --models grok --uncensored
  python src/data/generate_bios_offline.py --models gemma4 --gemma4-path models/kaya_gemma4_e4b
  python src/data/generate_bios_offline.py --compare-only   # rebuild comparison from existing JSONs
"""

import argparse
import gc
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports" / "bios"
CONFIG_PATH = BASE_DIR / "config.yaml"

# Default model directories (auto-detected from adapter_config.json)
DEFAULT_GEMMA4_PATH = BASE_DIR / "models" / "kaya_gemma4_e4b"
DEFAULT_QWEN3_PATHS = [
    BASE_DIR / "models" / "kaya_v2_synthetic",
    BASE_DIR / "models" / "kaya_qwen3_14b",
]

ALL_MODELS = ["gemma4", "qwen3", "grok", "azure", "azure_gpt53"]
LOCAL_MODELS = {"gemma4", "qwen3"}

# Maximum tokens of chat history to include per member bio request.
# Local models: keep small to fit in 4096-token context.
# Cloud models: can take larger chunks.
MAX_CONTEXT_TOKENS_LOCAL = 1800
MAX_CONTEXT_TOKENS_CLOUD = 6000

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BIO_SYSTEM_PROMPT = """You are a meticulous archivist analysing private chat logs of a Portuguese friend group called "Kaya". Your task is to build a structured biographical profile for a specific member based solely on what the chat messages reveal.

Rules:
- Base EVERYTHING on the chat excerpts provided. Do NOT invent facts.
- Use null for fields without clear evidence.
- Write biography_summary and free_text_bio in English.
- Keep biography_summary under 3 sentences.
- free_text_bio should be a natural 2-4 paragraph narrative covering personality, interests, lifestyle, and social role.
- Do NOT self-censor about normal social behaviour, banter, humour, or personal details mentioned in the chats."""

BIO_USER_TEMPLATE = """Based on these chat excerpts from the Kaya group, write a biography for: {name}

=== CHAT EXCERPTS ===
{context}
=== END EXCERPTS ===

Return ONLY valid JSON with this exact structure (no markdown fences):
{{
  "name": "{name}",
  "age": "approximate age or null",
  "occupation": "job/studies or null",
  "living_place": "city/country or null",
  "marital_status": "single/relationship/married/etc or null",
  "state_of_mind": "brief mood/attitude from messages or null",
  "interests": ["interest1", "interest2"],
  "frequently_discussed_topics": ["topic1", "topic2"],
  "biography_summary": "2-3 sentence factual bio",
  "free_text_bio": "2-4 paragraph narrative biography"
}}"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_members(config: Dict[str, Any]) -> List[str]:
    """Return the list of member names from group_members.json."""
    members_path = DATA_DIR / "group_members.json"
    if not members_path.exists():
        raise FileNotFoundError(f"group_members.json not found at {members_path}")
    with open(members_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [m["name"] for m in data.get("members", [])]


def load_messages() -> List[Dict[str, Any]]:
    """Load cleaned messages from all_messages_cleaned.jsonl."""
    path = DATA_DIR / "all_messages_cleaned.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"all_messages_cleaned.jsonl not found at {path}")
    msgs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return msgs


def collect_member_context(
    messages: List[Dict[str, Any]],
    member_name: str,
    max_tokens: int,
    aliases: Optional[List[str]] = None,
) -> str:
    """Collect chat messages relevant to a member, up to max_tokens chars (~4 chars/token)."""
    char_limit = max_tokens * 4
    name_lower = member_name.lower()
    patterns = [name_lower] + [a.lower() for a in (aliases or [])]

    relevant = []
    for msg in messages:
        sender = msg.get("sender", "").lower()
        content = msg.get("content", "").lower()
        if any(p in sender or p in content for p in patterns):
            relevant.append(msg)

    # Build context string
    lines = []
    total_chars = 0
    for msg in relevant:
        line = f"[{msg.get('timestamp','')[:10]}] {msg.get('sender','')}: {msg.get('content','')}"
        if total_chars + len(line) > char_limit:
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object from a text response."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        # Try to salvage partial JSON by truncating at last complete field
        return None


def resolve_local_path(
    cli_path: Optional[str],
    defaults: List[Path],
    model_name: str,
) -> Path:
    """Resolve the local model directory, checking defaults if cli_path not given."""
    if cli_path:
        p = Path(cli_path)
        if not p.is_absolute():
            p = BASE_DIR / p
        if not p.exists():
            raise FileNotFoundError(f"{model_name} model path not found: {p}")
        return p
    for d in defaults:
        if d.exists() and (d / "adapter_config.json").exists():
            return d
    raise FileNotFoundError(
        f"Could not find {model_name} model. Searched: {defaults}. "
        f"Pass --{model_name.replace('3','')}3-path or --gemma4-path."
    )


def detect_base_model(model_dir: Path) -> str:
    """Read adapter_config.json and return the base_model_name_or_path."""
    cfg_path = model_dir / "adapter_config.json"
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return data.get("base_model_name_or_path", "")
    return ""


def is_gemma4_model(base_model_id: str) -> bool:
    return "gemma-4" in base_model_id.lower() or "gemma4" in base_model_id.lower()


# ---------------------------------------------------------------------------
# Local model inference
# ---------------------------------------------------------------------------

def load_local_model(model_dir: Path, max_seq_length: int = 4096):
    """Load a finetuned local model using the correct Unsloth loader."""
    base_model_id = detect_base_model(model_dir)
    print(f"   Detected base model: {base_model_id}")

    if is_gemma4_model(base_model_id):
        print("   Using Unsloth FastModel (Gemma 4)...")
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(
            model_name=str(model_dir),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastModel.for_inference(model)
    else:
        print("   Using Unsloth FastLanguageModel (Qwen3/other)...")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_dir),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

    return model, tokenizer, base_model_id


def unload_local_model(model, tokenizer):
    """Unload model and free VRAM."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("   Model unloaded, VRAM freed.")


def generate_local(
    model,
    tokenizer,
    base_model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 1024,
) -> str:
    """Generate text with a local model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Gemma4 tokenizer needs text= kwarg
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(text=input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Cloud model inference
# ---------------------------------------------------------------------------

def generate_cloud(
    provider_name: str,
    config: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Generate text with a cloud provider (azure, azure_gpt53, grok)."""
    sys.path.insert(0, str(BASE_DIR / "src"))
    load_dotenv(BASE_DIR / ".env")

    if provider_name in ("azure", "azure_gpt53"):
        from src.llm_providers.azure_provider import AzureProvider
        config_key = "azure_gpt53" if provider_name == "azure_gpt53" else "azure"
        provider = AzureProvider(config, config_key=config_key)
    elif provider_name == "grok":
        from src.llm_providers.xai_provider import XAIProvider
        provider = XAIProvider(config)
    else:
        raise ValueError(f"Unknown cloud provider: {provider_name}")

    return provider.generate_text(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Per-member bio generation
# ---------------------------------------------------------------------------

def generate_member_bio(
    model_name: str,
    member_name: str,
    context: str,
    system_prompt: str,
    model=None,
    tokenizer=None,
    base_model_id: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate and parse a bio for one member with one model."""
    user_prompt = BIO_USER_TEMPLATE.format(name=member_name, context=context)

    try:
        if model_name in LOCAL_MODELS:
            raw = generate_local(model, tokenizer, base_model_id, system_prompt, user_prompt)
        else:
            raw = generate_cloud(model_name, config, system_prompt, user_prompt)

        parsed = parse_json_response(raw)
        if parsed is None:
            print(f"   ⚠ Could not parse JSON for {member_name}. Raw (first 200 chars):")
            print(f"     {raw[:200]}")
            return {"name": member_name, "_parse_error": True, "_raw": raw[:500]}
        parsed["name"] = member_name
        return parsed

    except Exception as e:
        print(f"   ✗ Error generating bio for {member_name}: {e}")
        return {"name": member_name, "_error": str(e)}


# ---------------------------------------------------------------------------
# Comparison markdown
# ---------------------------------------------------------------------------

def build_comparison_md(output_dir: Path, member_names: List[str]) -> str:
    """Read all bios_*.json files and generate a side-by-side comparison."""
    json_files = sorted(output_dir.glob("bios_*.json"))
    if not json_files:
        return "No bio JSON files found in reports/bios/."

    models_data: Dict[str, Dict[str, Dict]] = {}
    for jf in json_files:
        model_label = jf.stem[len("bios_"):]  # strip "bios_" prefix
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        # data is {"members": {"Name": {...}}}
        models_data[model_label] = data.get("members", {})

    model_labels = list(models_data.keys())
    lines = [
        "# Kaya Bio Generator — Model Comparison",
        "",
        f"Generated from: {', '.join(f'`bios_{m}.json`' for m in model_labels)}",
        "",
        "---",
        "",
    ]

    for member in member_names:
        lines.append(f"## {member}")
        lines.append("")

        for model_label in model_labels:
            bio = models_data[model_label].get(member, {})
            lines.append(f"### {model_label}")
            lines.append("")

            if not bio or bio.get("_error") or bio.get("_parse_error"):
                err = bio.get("_error") or bio.get("_raw", "parse error")
                lines.append(f"_Error: {err[:200]}_")
                lines.append("")
                continue

            # Structured fields
            field_rows = []
            for field in ("age", "occupation", "living_place", "marital_status", "state_of_mind"):
                val = bio.get(field)
                if val:
                    field_rows.append(f"| {field} | {val} |")
            if field_rows:
                lines.append("| Field | Value |")
                lines.append("|---|---|")
                lines.extend(field_rows)
                lines.append("")

            interests = bio.get("interests") or []
            topics = bio.get("frequently_discussed_topics") or []
            if interests:
                lines.append(f"**Interests:** {', '.join(interests)}")
                lines.append("")
            if topics:
                lines.append(f"**Topics:** {', '.join(topics)}")
                lines.append("")

            summary = bio.get("biography_summary") or ""
            if summary:
                lines.append(f"**Summary:** {summary}")
                lines.append("")

            free_text = bio.get("free_text_bio") or ""
            if free_text:
                lines.append("**Narrative:**")
                lines.append("")
                for para in free_text.split("\n\n"):
                    para = para.strip()
                    if para:
                        lines.append(textwrap.fill(para, width=100))
                        lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate member biographies using multiple models and compare."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(ALL_MODELS),
        help=f"Comma-separated list of models to use. Options: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--gemma4-path",
        type=str,
        default=None,
        help=f"Path to finetuned Gemma 4 model dir (default: {DEFAULT_GEMMA4_PATH})",
    )
    parser.add_argument(
        "--qwen3-path",
        type=str,
        default=None,
        help="Path to finetuned Qwen3 model dir (default: auto-detected from models/)",
    )
    parser.add_argument(
        "--uncensored",
        action="store_true",
        default=True,
        help="Prepend uncensored system prompt preamble (default: enabled).",
    )
    parser.add_argument(
        "--no-uncensored",
        dest="uncensored",
        action="store_false",
        help="Disable uncensored preamble (use vanilla extraction prompt only).",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        default=False,
        help="Skip generation; only rebuild bio_comparison.md from existing JSONs.",
    )
    parser.add_argument(
        "--max-context-local",
        type=int,
        default=MAX_CONTEXT_TOKENS_LOCAL,
        help=f"Max context tokens for local models (default: {MAX_CONTEXT_TOKENS_LOCAL})",
    )
    parser.add_argument(
        "--max-context-cloud",
        type=int,
        default=MAX_CONTEXT_TOKENS_CLOUD,
        help=f"Max context tokens for cloud models (default: {MAX_CONTEXT_TOKENS_CLOUD})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    member_names = load_members(config)

    # --compare-only: rebuild markdown only
    if args.compare_only:
        print("Rebuilding bio_comparison.md from existing JSON files...")
        md = build_comparison_md(REPORTS_DIR, member_names)
        out = REPORTS_DIR / "bio_comparison.md"
        out.write_text(md, encoding="utf-8")
        print(f"✅ Saved: {out}")
        return

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in requested_models if m not in ALL_MODELS]
    if invalid:
        print(f"❌ Unknown models: {invalid}. Valid: {ALL_MODELS}")
        sys.exit(1)

    print(f"\nKaya Offline Bio Generator")
    print(f"Models: {requested_models}")
    print(f"Members: {member_names}")
    print(f"Uncensored preamble: {args.uncensored}")
    print(f"Output: {REPORTS_DIR}")

    # Build system prompt
    base_system = BIO_SYSTEM_PROMPT
    if args.uncensored:
        preamble = config.get("chat", {}).get("uncensored_system_prompt", "")
        if preamble:
            base_system = preamble + "\n\n" + base_system
            print("   (uncensored preamble prepended)")

    messages = load_messages()

    # -----------------------------------------------------------------------
    # Generate bios for each requested model
    # -----------------------------------------------------------------------
    for model_name in requested_models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        is_local = model_name in LOCAL_MODELS
        max_ctx = args.max_context_local if is_local else args.max_context_cloud

        results: Dict[str, Any] = {}

        if is_local:
            # Resolve model path
            try:
                if model_name == "gemma4":
                    model_dir = resolve_local_path(
                        args.gemma4_path, [DEFAULT_GEMMA4_PATH], "gemma4"
                    )
                else:  # qwen3
                    model_dir = resolve_local_path(
                        args.qwen3_path, DEFAULT_QWEN3_PATHS, "qwen3"
                    )
            except FileNotFoundError as e:
                print(f"   ⚠ Skipping {model_name}: {e}")
                continue

            print(f"   Model dir: {model_dir}")
            print("   Loading model (this may take a minute)...")

            try:
                lm, tok, base_id = load_local_model(model_dir, max_seq_length=4096)
            except Exception as e:
                print(f"   ✗ Failed to load model: {e}")
                continue

            for member in member_names:
                print(f"   Generating bio for: {member}...", end=" ", flush=True)
                ctx = collect_member_context(messages, member, max_ctx)
                if not ctx.strip():
                    print("no context found, skipping.")
                    results[member] = {"name": member, "_error": "no context found"}
                    continue
                bio = generate_member_bio(
                    model_name, member, ctx, base_system,
                    model=lm, tokenizer=tok, base_model_id=base_id,
                )
                results[member] = bio
                print("done.")

            unload_local_model(lm, tok)

        else:
            # Cloud model — no loading needed
            for member in member_names:
                print(f"   Generating bio for: {member}...", end=" ", flush=True)
                ctx = collect_member_context(messages, member, max_ctx)
                if not ctx.strip():
                    print("no context found, skipping.")
                    results[member] = {"name": member, "_error": "no context found"}
                    continue
                bio = generate_member_bio(
                    model_name, member, ctx, base_system, config=config
                )
                results[member] = bio
                print("done.")

        # Save per-model output
        out_path = REPORTS_DIR / f"bios_{model_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_name, "members": results}, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved: {out_path}")

    # -----------------------------------------------------------------------
    # Build comparison markdown
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Building comparison report...")
    md = build_comparison_md(REPORTS_DIR, member_names)
    comp_path = REPORTS_DIR / "bio_comparison.md"
    comp_path.write_text(md, encoding="utf-8")
    print(f"✅ Saved: {comp_path}")

    print("\nDone. Files in reports/bios/:")
    for f in sorted(REPORTS_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
