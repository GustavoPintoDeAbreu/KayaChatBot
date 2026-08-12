#!/usr/bin/env python3
"""Score CONVERSATIONAL quality — the thing the golden suite structurally cannot.

The golden judge grades every answer on factual grounding and detail. That is the
right lens for "Quem é o Peter?" and the wrong one for "Hey! How are you?", which
scored **1.33/5** for a correct, natural, short reply. Optimising against that
metric is part of how the bot ended up answering `Ahahhha` with a 77-word analysis
of group dynamics.

This probe scores the opposite properties, deterministically — no LLM judge, so it
is free, fast and repeatable:

  routing      did the router pick the intended mode?
  brevity      is the reply within the mode's word budget?
  restraint    did it volunteer group members nobody asked about?
  substance    did it actually say something (not empty, not a refusal)?
  in_voice     did it stay in character, or slip into assistant register
               ("A primeira parte da pergunta não se enquadra…", "como uma IA…")?
  no_dash      did it avoid dashes used as clause separators? The group asked for
               commas at most, and the model reaches for " — " constantly.
  complied     for cases that carry `must_contain_any`, did it actually do the
               thing? A polite deflection can pass every check above.

    # needs a llama.cpp server for the model under test
    KAYA_INFERENCE_BACKEND=gguf KAYA_LLAMA_URL=http://127.0.0.1:8081 \
      kaya_chatbot_env/bin/python scripts/run_conversation_probe.py
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.chat.engine import get_engine, build_system_prompt

DEFAULT_PROBE = "data/conversation_probe.json"

# Assistant register: the bot stepping out of the conversation to talk about what
# it is and what it will not do. Every one of these is drawn from a real logged
# reply, most notably "A primeira parte da pergunta não se enquadra em resposta
# factual baseada na web." — which arrived, spoken aloud, after a roast request.
_ROBOTIC_RE = re.compile(
    r"não se enquadra"
    r"|(?:a\s+)?primeira parte d[ao] (?:pergunta|mensagem)"
    r"|(?:segunda|outra) parte d[ao] (?:pergunta|mensagem)"
    r"|não pos(?:so)? responder a essa parte"
    r"|como (?:uma? )?(?:ia|intelig[êe]ncia artificial|modelo de linguagem|assistente virtual)\b"
    r"|enquanto (?:ia|assistente)\b"
    r"|sou (?:apenas |só )?(?:uma? )?(?:ia|bot|modelo|assistente)\b"
    r"|não tenho (?:a )?capacidade de"
    r"|as minhas diretrizes|os meus desenvolvedores|não tenho permissão"
    r"|limite de conhecimento|knowledge cutoff"
    r"|\bas an ai\b|\bi'?m (?:just |only )?an? (?:ai|assistant|bot|language model)\b"
    r"|\bi can'?t (?:help|assist|comply) with (?:that|this)\b"
    r"|\bi'?m not able to\b",
    re.IGNORECASE,
)

# A dash separating clauses (whitespace on the left). Intra-word hyphens such as
# "dá-me" are not a violation and must not match.
_CLAUSE_DASH_RE = re.compile(r"\s[—–]|\s-{1,2}(?=\s)")


def _member_names(config, base_dir: Path):
    """Group member first names, used to detect unsolicited fact-dumping."""
    members_file = config.get("data", {}).get("group_members_file")
    if not members_file:
        return set()
    path = Path(members_file)
    if not path.is_absolute():
        path = base_dir / members_file
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data.get("members", data) if isinstance(data, dict) else data
    out = set()
    for name in (names.keys() if isinstance(names, dict) else names):
        first = str(name).strip().split()[0]
        if len(first) > 2:
            out.add(first.lower())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Conversational quality probe.")
    ap.add_argument("--probe-file", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--speaker", default="Gustavo")
    ap.add_argument("--tag", default="conv")
    args = ap.parse_args()

    base_dir = Path(__file__).parent.parent
    config_path = str(base_dir / "config.yaml")
    config = load_config(config_path)
    if args.model_dir:
        config["training"]["output_dir"] = args.model_dir

    probe_path = Path(args.probe_file or (base_dir / DEFAULT_PROBE))
    cases = json.loads(probe_path.read_text(encoding="utf-8"))["cases"]
    members = _member_names(config, base_dir)

    engine = get_engine(config)
    system_prompt = build_system_prompt(
        config, config_path,
        include_uncensored=config.get("chat", {}).get("uncensored_mode", False),
    )

    rows, t_start = [], time.time()
    print(f"{'id':<16}{'mode':<9}{'want':<9}{'words':>6}{'sec':>6}  reply")
    print("-" * 108)

    for case in cases:
        t0 = time.perf_counter()
        reply = engine.respond(case["message"], args.speaker, None, system_prompt)
        elapsed = time.perf_counter() - t0

        text = (reply.text or "").strip()
        words = len(text.split())
        mode = getattr(reply.route, "mode", "?") if reply.route else "?"
        want = case["expected_mode"]

        # Members named in the reply that were not named in the prompt — the
        # "answered 😂 with a paragraph about Gil" failure.
        asked = case["message"].lower()
        volunteered = sorted(
            n for n in members
            if re.search(rf"\b{re.escape(n)}\b", text.lower()) and n not in asked
        )

        robotic = _ROBOTIC_RE.search(text)
        wanted_substrings = [s.lower() for s in case.get("must_contain_any", [])]
        row = {
            "id": case["id"], "message": case["message"],
            "expected_mode": want, "actual_mode": mode,
            "routing_ok": mode == want,
            "words": words, "max_words": case["max_words"],
            "brevity_ok": words <= case["max_words"],
            "volunteered_members": volunteered,
            # Only factual answers are expected to name members; that is the point
            # of the mode. `general` must not — the question is not about them.
            "restraint_ok": want == "factual" or not volunteered,
            "substance_ok": words > 0,
            "robotic_match": robotic.group(0) if robotic else "",
            "in_voice_ok": robotic is None,
            "no_dash_ok": _CLAUSE_DASH_RE.search(text) is None,
            "complied_ok": (
                not wanted_substrings
                or any(sub in text.lower() for sub in wanted_substrings)
            ),
            "seconds": round(elapsed, 2),
            "reply": text,
        }
        rows.append(row)

        checks = ("routing_ok", "brevity_ok", "restraint_ok",
                  "in_voice_ok", "no_dash_ok", "complied_ok")
        failed = [c[:-3] for c in checks if not row[c]]
        flag = f"  <-- FAIL: {','.join(failed)}" if failed else ""
        print(f"{case['id']:<16}{mode:<9}{want:<9}{words:>6}{elapsed:>6.1f}  {text[:52]!r}{flag}")

    def rate(key):
        return sum(1 for r in rows if r[key]) / len(rows) if rows else 0.0

    summary = {
        "routing_accuracy": round(rate("routing_ok"), 4),
        "brevity_rate": round(rate("brevity_ok"), 4),
        "restraint_rate": round(rate("restraint_ok"), 4),
        "substance_rate": round(rate("substance_ok"), 4),
        "in_voice_rate": round(rate("in_voice_ok"), 4),
        "no_dash_rate": round(rate("no_dash_ok"), 4),
        "compliance_rate": round(rate("complied_ok"), 4),
        "median_seconds": round(sorted(r["seconds"] for r in rows)[len(rows) // 2], 2),
        "cases": len(rows),
    }

    print("-" * 108)
    for k, v in summary.items():
        print(f"  {k:<20}{v}")

    out_dir = base_dir / config.get("benchmark", {}).get("output_dir", "reports/benchmarks/")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"conversation_{args.tag}_{stamp}.json"
    out_path.write_text(json.dumps({
        "_meta": {"timestamp": stamp, "model_dir": config["training"]["output_dir"],
                  "elapsed_min": round((time.time() - t_start) / 60, 1)},
        "summary": summary, "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport saved → {out_path}")


if __name__ == "__main__":
    main()
