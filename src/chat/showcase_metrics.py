"""The showcase page's numbers: labwatch's public feed, fetched here — never by the browser.

The page is public and the browser must not reach the LAN, so Kaya's server reads labwatch's
allowlisted feed (GET /api/catalog/public), caches it for ~10 minutes, and injects only those
numbers into the page as an inline <script> that static_assets.page_csp hashes at serve time.
Stdlib + httpx: the Pi gateway imports it (tests/gateway/test_torch_free.py)."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import httpx

LABWATCH_PUBLIC_URL_DEFAULT = "http://192.168.1.238:8095/api/catalog/public"
CACHE_TTL_S = 600
FAILURE_TTL_S = 60
FETCH_TIMEOUT_S = 3.0

PROJECT_FEED_NAMES: Dict[str, str] = {
    "kaya": "Kaya",
    "labwatch": "labwatch",
    "redditcast": "redditcast",
    "flight-deals": "flight-deals",
}
FEED_NAME_TO_PROJECT: Dict[str, str] = {v: k for k, v in PROJECT_FEED_NAMES.items()}

# (key as LabApi.public emits it, English label, Portuguese label, unit)
LAB_STATS = [
    ("local_ai_tokens_generated_this_week", "tokens generated this week", "tokens gerados esta semana", ""),
    ("local_ai_tokens_read_this_week", "tokens read this week", "tokens lidos esta semana", ""),
    ("euros_saved_vs_cloud_this_week", "saved vs. the cloud this week", "poupado face à nuvem esta semana", "€"),
    ("electricity_kwh_this_week", "electricity this week", "eletricidade esta semana", " kWh"),
    ("electricity_cost_eur_this_week", "electricity cost this week", "custo da eletricidade esta semana", "€"),
]

# The per-project stats the feed publishes, translated; anything unknown stays English.
STAT_LABEL_PT: Dict[str, str] = {
    "messages Kaya read this week": "mensagens que Kaya leu esta semana",
    "replies Kaya sent this week": "respostas que Kaya enviou esta semana",
    "voice replies this week": "respostas por voz esta semana",
    "episodes made this week": "episódios feitos esta semana",
    "price sweeps this week": "pesquisas de preços esta semana",
    "deal alerts sent this week": "alertas de promoções esta semana",
}

# How a request flows, per project: plain words, no infrastructure.
FLOWS: Dict[str, Dict[str, list]] = {
    "kaya": {"en": ["A friend writes in the group", "Kaya reads it and sorts it",
                    "It searches the group's history", "It answers — text or voice"],
             "pt": ["Um amigo escreve no grupo", "Kaya lê e classifica",
                    "Procura no histórico do grupo", "Responde — por texto ou voz"]},
    "lab-assistant": {"en": ["A voice note arrives", "It is transcribed at home",
                             "A local model picks the tool", "It answers, or waits for your yes"],
                      "pt": ["Chega um áudio", "É transcrito em casa",
                             "Um modelo local escolhe a ferramenta", "Responde, ou espera pelo teu sim"]},
    "lab-notes": {"en": ["You dictate a note", "It is transcribed",
                         "It lands on the right notebook page"],
                  "pt": ["Ditas uma nota", "É transcrita",
                         "Vai para a página certa do caderno"]},
    "idea-pipeline": {"en": ["You send an idea", "Agents ask their questions",
                             "A plan, then code in a sandbox", "A pull request waits for your yes"],
                      "pt": ["Envias uma ideia", "Os agentes fazem perguntas",
                             "Um plano e código numa sandbox", "Um pull request espera pelo teu sim"]},
    "lab-medic": {"en": ["It reads every log and site", "It fingerprints the errors",
                         "A recurring bug becomes a fix", "Nothing ships until you say yes"],
                  "pt": ["Lê os registos e os sites", "Dá uma impressão digital aos erros",
                         "Um erro recorrente vira correção", "Nada sai sem o teu sim"]},
    "labwatch": {"en": ["An agent samples the machines", "A hub keeps the numbers in memory",
                        "Money, power and tokens are worked out", "One dashboard shows it all"],
                 "pt": ["Um agente recolhe das máquinas", "Um hub guarda os números em memória",
                        "Dinheiro, energia e tokens são calculados", "Um painel mostra tudo"]},
    "flight-deals": {"en": ["It sweeps airline prices", "Compares to what each trip usually costs",
                            "Flags the outliers", "One message a day when it's cheap"],
                     "pt": ["Pesquisa preços de voos", "Compara com o custo habitual",
                            "Assinala as exceções", "Uma mensagem por dia quando é barato"]},
    "price-scout": {"en": ["It follows the items you want", "Keeps each price history",
                           "A target or a new low is hit", "You get a ping"],
                    "pt": ["Segue os artigos que queres", "Guarda o histórico de preços",
                           "Atinge o alvo ou um novo mínimo", "Recebes um aviso"]},
    "series-watch": {"en": ["It watches the release slates", "A season is confirmed, dated or out",
                           "You get told, with where to watch"],
                     "pt": ["Vigia os calendários de estreias", "Uma temporada é confirmada, datada ou estreia",
                            "Ficas a saber, com onde ver"]},
    "game-alerts": {"en": ["It watches game news", "A limited-time event is announced",
                           "You get a ping, then another at go-live"],
                    "pt": ["Vigia as notícias dos jogos", "Um evento temporário é anunciado",
                           "Recebes um aviso, e outro no arranque"]},
    "redditcast": {"en": ["The week's top posts are collected", "A model writes a two-host script",
                          "It is voiced at home", "A voice note lands on Monday"],
                   "pt": ["Recolhe os melhores posts da semana", "Um modelo escreve um guião a duas vozes",
                          "É narrado em casa", "Um áudio chega na segunda"]},
    "subs": {"en": ["Subtitles come in", "Translated in validated batches",
                    "Each line is checked", "European Portuguese out"],
             "pt": ["Entram legendas", "Traduzidas em lotes validados",
                    "Cada linha é verificada", "Sai português europeu"]},
}

# The original labwatch /idea prompt, owner-approved for the showcase page.
LABWATCH_PROMPT = (
    "Build me a self-hosted dashboard for my home lab: one PC with two GPUs that runs local LLMs behind a model broker, "
    "and a Raspberry Pi that is always on. An agent on the PC samples every 10 s (GPU watts/util/VRAM/temp, CPU power from RAPL, "
    "docker containers, the broker's token events) and pushes to a hub on the Pi, which keeps live data in memory, writes SQLite "
    "in small batches (the Pi boots from an SSD), and keeps minutes for 48 h and days forever. Show: power and money (the wall plug's "
    "watts, kWh, my tariff's EUR, energy split per model/project), LLM tokens and tok/s for every model, local or cloud, and what the "
    "same work would have cost on a hosted API, my bots' traffic (counts only, never message text), each side project's health, and the "
    "health of every container and service with alert rules that ping me on WhatsApp. Counters travel as deltas with cursors so restarts "
    "never double-count. Add a catalog of the lab (projects, services, ports, how to use each) served over MCP so my coding agents ask it "
    "before building something that already exists. Vanilla JS pages, no CDN, LAN only."
)

_CACHE: Optional[Tuple[float, dict]] = None
_FAILED_AT: Optional[float] = None


def labwatch_url() -> str:
    """labwatch's public feed, or KAYA_LABWATCH_PUBLIC_URL for a different host."""
    return os.environ.get("KAYA_LABWATCH_PUBLIC_URL", LABWATCH_PUBLIC_URL_DEFAULT)


def fetch_public(url: Optional[str] = None, timeout: float = FETCH_TIMEOUT_S) -> Optional[dict]:
    """The feed's JSON, or the last numbers it gave, or None when there are none.

    A failure is cached too (FAILURE_TTL_S): while labwatch is down a page load must not
    wait the full fetch timeout every time. During that window the last good payload is served
    even if it is older than CACHE_TTL_S — stale numbers beat no numbers, and the page cannot
    tell the difference. An explicit ``url`` reads and writes nothing: it is a test or another
    lab, and its numbers must not be served to the page as if they were this one's."""
    global _CACHE, _FAILED_AT
    now = time.time()
    if url is None and _CACHE is not None and now - _CACHE[0] < CACHE_TTL_S:
        return _CACHE[1]
    if url is None and _FAILED_AT is not None and now - _FAILED_AT < FAILURE_TTL_S:
        return _CACHE[1] if _CACHE is not None else None
    try:
        response = httpx.get(url or labwatch_url(), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, OSError):
        if url is None:
            _FAILED_AT = time.time()
            return _CACHE[1] if _CACHE is not None else None
        return None
    if url is None:
        _CACHE = (time.time(), payload)
        _FAILED_AT = None
    return payload


def build_view(payload: Optional[dict]) -> dict:
    """The page's view model from the feed, stable whatever the feed says."""
    lab_stats: List[dict] = []
    products: Dict[str, Dict[str, list]] = {}
    if payload is not None:
        lab_values = (payload.get("lab") or {}).get("stats", {})
        for key, en, pt, unit in LAB_STATS:
            lab_stats.append({"key": key, "value": lab_values.get(key), "en": en, "pt": pt, "unit": unit})
        for entry in payload.get("products") or []:
            project = FEED_NAME_TO_PROJECT.get(entry.get("name"))
            if project is None:
                continue
            stats = [{"en": stat["label"],
                      "pt": STAT_LABEL_PT.get(stat.get("label"), stat.get("label")),
                      "value": stat["value"]}
                     for stat in entry.get("stats") or [] if stat.get("value") is not None]
            if stats:
                products[project] = {"stats": stats}
    return {"lab": {"stats": lab_stats}, "products": products, "flows": FLOWS,
            "prompt": {"labwatch": LABWATCH_PROMPT}}


def render_injection(view: dict) -> str:
    """The view as an inline <script>. JSON escaping is not HTML escaping: a </script> inside a
    label would close the tag and run whatever followed it, so the angle brackets stay escaped.
    The two line separators are JSON-legal but end a JavaScript line, so they escape too."""
    body = "window.KAYA_SHOWCASE = " + json.dumps(view, ensure_ascii=False, separators=(",", ":")) + ";"
    for char, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                          (chr(0x2028), "\\u2028"), (chr(0x2029), "\\u2029")):
        body = body.replace(char, escaped)
    return "<script>" + body + "</script>"


def inject(html: str, view: dict) -> str:
    """The page with the injection before its </body>, once."""
    script = render_injection(view)
    if script in html:
        return html
    index = html.find("</body>")
    if index < 0:
        return html + script
    return html[:index] + script + html[index:]


def _dry_run() -> None:
    """What the page would get from the feed, without writing anything."""
    payload = fetch_public()
    view = build_view(payload)
    print(f"feed: {labwatch_url()}")
    if payload is None:
        print("feed unreachable — the page keeps its flows and prompt and shows no metric bars")
    else:
        print(f"feed answered: {len(view['products'])} projects carry their own stats")
    print(f"lab-wide stats: {len(view['lab']['stats'])}")
    print(f"flows for {len(view['flows'])} projects; prompt for labwatch")


def main(argv: Optional[list] = None) -> int:
    if list(argv or []) == ["--dry-run"]:
        _dry_run()
        return 0
    print("usage: python -m src.chat.showcase_metrics --dry-run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
