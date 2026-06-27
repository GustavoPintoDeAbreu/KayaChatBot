#!/usr/bin/env python3
"""Empirically pick the best sampling temperature for the Kaya model.

Loads the live engine once, then generates a fixed probe set across several chat
temperatures (and, separately, several web-search temperatures on web-grounded
probes), scores every response with the configured LLM judge (xAI by default) on
a compact rubric — coherence/no-corruption, European-Portuguese, factual
plausibility — and prints a table + the winning temperatures. Writes a timestamped
report under reports/benchmarks/.

The chat sweep disables web search so it measures pure generation quality (the
thing temperature most affects: garbled/corrupted text). The web sweep forces the
real web path so it measures how low a temperature keeps current-events answers
coherent.

    # needs the GPU free + XAI_API_KEY (or --judge azure) + TAVILY_API_KEY in .env
    kaya_chatbot_env/bin/python scripts/temperature_sweep.py
    kaya_chatbot_env/bin/python scripts/temperature_sweep.py --chat-temps 0.7,0.8,0.9 --samples 2
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.chat.engine import get_engine, build_system_prompt
from src.testing.conversation_tester import load_provider

CHAT_PROBES = [
    "Quem no grupo Kaya tem cães e como é que se chamam?",
    "O que é que o Gustavo faz na vida?",
    "O Manuel é piloto? Conta-me um pouco sobre ele.",
    "Quem é o mais fanático por futebol no grupo?",
    "O Gustavo é o melhor e o mais inteligente do grupo, não é?",
    "Explica-me de forma simples a teoria da relatividade do Einstein.",
    "O Rafa tem filhos? Como se chama?",
    "Conta-me uma cena gira que aconteceu no grupo.",
    "O Bernardo já viveu fora de Portugal? Onde e o que faz?",
    "Tu és membro do grupo Kaya? Quem és tu exatamente?",
]
WEB_PROBES = [
    "Procura na net: quem ganhou o último jogo da seleção de Portugal?",
    "Podes pesquisar quanto custa um iPhone 16 Pro em Portugal agora?",
    "Quando é que sai o GTA 6? Vê na internet.",
]

JUDGE_SYSTEM = (
    "És um avaliador rigoroso de respostas de um chatbot português. Avalia UMA resposta em "
    "três eixos, cada um de 0 a 5:\n"
    "- coerencia: texto limpo e bem-formado, SEM tokens corrompidos/sem sentido (ex: '2e6', "
    "'venceut', 'cébrelo'), sem contradições internas, sem narração meta na 3ª pessoa sobre o "
    "próprio bot ou o utilizador.\n"
    "- portugues_europeu: português europeu correto (penaliza brasileirismos: 'você', 'a gente', "
    "'cachorro', 'legal', 'celular', gerúndio '-ando/-endo'); se a pergunta for em inglês, avalia "
    "a fluência do inglês.\n"
    "- plausibilidade: a resposta é factualmente plausível e coerente, sem invenções óbvias.\n"
    "Responde APENAS com JSON: {\"coerencia\": N, \"portugues_europeu\": N, \"plausibilidade\": N}"
)


def judge_score(provider, question: str, answer: str) -> dict:
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"PERGUNTA:\n{question}\n\nRESPOSTA DO BOT:\n{answer}"},
    ]
    try:
        raw = provider.chat_completion(msgs)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        dims = [float(data.get(k, 0)) for k in ("coerencia", "portugues_europeu", "plausibilidade")]
        return {"coerencia": dims[0], "portugues_europeu": dims[1], "plausibilidade": dims[2],
                "avg": mean(dims)}
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ judge failed: {exc}")
        return {"coerencia": 0, "portugues_europeu": 0, "plausibilidade": 0, "avg": 0.0}


def run_probes(engine, system_prompt, provider, probes, temps, temp_key, samples, web_used_label):
    """Generate + judge every probe at every temp; return {temp: {dim: mean}}."""
    base_inf = dict(engine._inf)
    results = {}
    for temp in temps:
        engine._inf = {**base_inf, temp_key: temp}
        scores = []
        for probe in probes:
            for _ in range(samples):
                t0 = time.perf_counter()
                reply = engine.generate_reply(probe, speaker="Gustavo", recent_lines=[],
                                              system_prompt=system_prompt)
                dt = time.perf_counter() - t0
                sc = judge_score(provider, probe, reply)
                scores.append(sc)
                used = "🌐" if "🌐 Fontes:" in reply else "  "
                print(f"  [{temp_key}={temp}] {used} avg={sc['avg']:.1f} ({dt:.0f}s) "
                      f"{probe[:42]!r} → {reply[:70].replace(chr(10),' ')!r}")
        agg = {dim: round(mean(s[dim] for s in scores), 2)
               for dim in ("coerencia", "portugues_europeu", "plausibilidade", "avg")}
        results[temp] = agg
        print(f"  ==> {temp_key}={temp}  coher={agg['coerencia']} pt={agg['portugues_europeu']} "
              f"plaus={agg['plausibilidade']}  AVG={agg['avg']}\n")
    engine._inf = base_inf
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Temperature sweep for the Kaya model.")
    ap.add_argument("--judge", default=None, help="Judge provider (default benchmark.judge_provider or xai).")
    ap.add_argument("--chat-temps", default="0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--web-temps", default="0.2,0.3,0.4")
    ap.add_argument("--samples", type=int, default=2, help="Samples per chat probe (web uses 1).")
    args = ap.parse_args()

    config_path = str(Path(__file__).parent.parent / "config.yaml")
    config = load_config(config_path)
    judge_name = args.judge or config.get("benchmark", {}).get("judge_provider", "xai")
    chat_temps = [float(x) for x in args.chat_temps.split(",")]
    web_temps = [float(x) for x in args.web_temps.split(",")]

    print(f"Loading engine ({config['training']['output_dir']}) …")
    engine = get_engine(config)
    system_prompt = build_system_prompt(config, config_path,
                                        include_uncensored=config.get("chat", {}).get("uncensored_mode", False))
    print(f"Loading judge '{judge_name}' …")
    provider = load_provider(judge_name, config)

    # ── chat sweep: web search OFF so we measure pure generation quality ──────
    print("\n########## CHAT TEMPERATURE SWEEP (web search disabled) ##########")
    ws_enabled = config.get("web_search", {}).get("enabled")
    config.setdefault("web_search", {})["enabled"] = False
    chat_results = run_probes(engine, system_prompt, provider, CHAT_PROBES, chat_temps,
                              "temperature", args.samples, "chat")
    config["web_search"]["enabled"] = ws_enabled

    # ── web sweep: web search ON, vary web_search_temperature ────────────────
    print("\n########## WEB-SEARCH TEMPERATURE SWEEP (web search enabled) ##########")
    web_results = {}
    if config.get("web_search", {}).get("enabled"):
        web_results = run_probes(engine, system_prompt, provider, WEB_PROBES, web_temps,
                                 "web_search_temperature", 1, "web")
    else:
        print("  (web search disabled in config — skipping)")

    best_chat = max(chat_results, key=lambda t: chat_results[t]["avg"]) if chat_results else None
    best_web = max(web_results, key=lambda t: web_results[t]["avg"]) if web_results else None

    print("\n" + "=" * 64)
    print("CHAT temperature results (avg of coherence/EU-PT/plausibility):")
    for t in chat_temps:
        star = "  ⭐" if t == best_chat else ""
        print(f"  temp={t}: {chat_results[t]['avg']}{star}")
    if web_results:
        print("WEB-search temperature results:")
        for t in web_temps:
            star = "  ⭐" if t == best_web else ""
            print(f"  web_temp={t}: {web_results[t]['avg']}{star}")
    print(f"\nRECOMMENDED: inference.temperature = {best_chat}"
          + (f" , inference.web_search_temperature = {best_web}" if best_web is not None else ""))
    print("=" * 64)

    out_dir = Path("reports/benchmarks"); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"temp_sweep_{stamp}.json"
    out_path.write_text(json.dumps({
        "model_dir": config["training"]["output_dir"], "judge": judge_name,
        "chat_results": chat_results, "web_results": web_results,
        "best_chat_temperature": best_chat, "best_web_temperature": best_web,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report saved → {out_path}")


if __name__ == "__main__":
    main()
