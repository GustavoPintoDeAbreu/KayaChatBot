"""The showcase page's numbers: labwatch's public feed, fetched here — never by the browser.

The page is public and the browser must not reach the LAN, so Kaya's server reads labwatch's
allowlisted feed (GET /api/catalog/public), caches it for ~10 minutes, and injects only those
numbers into the page as an inline <script> that static_assets.page_csp hashes at serve time.
Each project shows only its own counters: the feed's lab-wide totals are not passed on, because
handing them to every card without counters made every project show the same tokens.
Stdlib + httpx: the Pi gateway imports it (tests/gateway/test_torch_free.py)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx

LABWATCH_PUBLIC_URL_DEFAULT = "http://192.168.1.238:8095/api/catalog/public"
CACHE_TTL_S = 600
FAILURE_TTL_S = 60
FETCH_TIMEOUT_S = 3.0
LIVE_PAGE_URL_DEFAULT = "https://sigmakayachat.pt/"
LIVE_TIMEOUT_S = 10.0

PROJECT_FEED_NAMES: Dict[str, str] = {
    "kaya": "Kaya",
    "labwatch": "labwatch",
    "redditcast": "redditcast",
    "flight-deals": "flight-deals",
    "lab-notes": "lab-notes",
    "idea-pipeline": "idea-pipeline",
    "lab-medic": "lab-medic",
    "price-scout": "price-scout",
    "Subtitle translator": "subs",
}
FEED_NAME_TO_PROJECT: Dict[str, str] = {v: k for k, v in PROJECT_FEED_NAMES.items()}

# The per-project stats the feed publishes, translated; anything unknown stays English.
STAT_LABEL_PT: Dict[str, str] = {
    "notes added, all time": "notas adicionadas, desde sempre",
    "ideas sent, all time": "ideias enviadas, desde sempre",
    "average idea to pull request, minutes": "minutos em média da ideia ao pull request",
    "local code Claude graded pass, %": "% de código local aprovado pelo Claude",
    "fixes shipped by the medic, all time": "correções enviadas pelo médico, desde sempre",
    "subtitle files translated, all time": "ficheiros de legendas traduzidos, desde sempre",
    "items followed": "artigos seguidos",
    "messages Kaya read this week": "mensagens que Kaya leu esta semana",
    "replies Kaya sent this week": "respostas que Kaya enviou esta semana",
    "voice replies this week": "respostas por voz esta semana",
    "episodes made this week": "episódios feitos esta semana",
    "episodes generated, all time": "episódios feitos, desde sempre",
    "price sweeps this week": "pesquisas de preços esta semana",
    "deal alerts sent this week": "alertas de promoções esta semana",
}

# How a request flows, per project: plain words, no infrastructure.
FLOWS: Dict[str, Dict[str, list]] = {
    "kaya": {
        "en": [
            "A friend writes in the WhatsApp group",
            "Kaya reads it and sorts it by intention",
            "It retrieves what the group already said",
            "It answers, text or voice"
        ],
        "pt": [
            "Um amigo escreve no grupo de WhatsApp",
            "Kaya lê e classifica pela intenção",
            "Procura no histórico do grupo",
            "Responde, por texto ou por voz"
        ]
    },
    "lab-assistant": {
        "en": [
            "A voice note arrives, by DM",
            "Kaya transcribes it at home",
            "Kaya picks the tool",
            "It answers, or waits for my yes"
        ],
        "pt": [
            "Chega um áudio, por DM",
            "É transcrito em casa",
            "Kaya escolhe a ferramenta",
            "Responde, ou espera pelo teu sim"
        ]
    },
    "lab-notes": {
        "en": [
            "You dictate a note, or write it down, on WhatsApp",
            "Kaya transcribes it",
            "It lands on the right OneNote page"
        ],
        "pt": [
            "Ditas uma nota, ou escreves, no WhatsApp",
            "É transcrita",
            "Vai para a página certa do caderno"
        ]
    },
    "idea-pipeline": {
        "en": [
            "I send an idea from my phone, by voice",
            "A planning agent asks the questions and writes a plan",
            "The implementation agent builds it in a sandbox",
            "A QA agent reviews it and opens a pull request for my review"
        ],
        "pt": [
            "Envio uma ideia pelo telemóvel, por voz ou texto",
            "Um agente de planeamento faz as perguntas e escreve um plano",
            "O agente de implementação constrói-o numa sandbox",
            "Um agente de QA revê-o e abre um pull request para eu rever"
        ]
    },
    "lab-medic": {
        "en": [
            "It reads every log and site",
            "It fingerprints the errors",
            "A recurring bug becomes a fix",
            "Nothing ships until you say yes"
        ],
        "pt": [
            "Lê os registos e os sites",
            "Dá uma impressão digital aos erros",
            "Um erro recorrente vira correção",
            "Nada sai sem o teu sim"
        ]
    },
    "labwatch": {
        "en": [
            "An agent samples the machines",
            "A hub keeps the numbers in memory",
            "Money, power and tokens are worked out",
            "One dashboard shows it all"
        ],
        "pt": [
            "Um agente recolhe das máquinas",
            "Um hub guarda os números em memória",
            "Dinheiro, energia e tokens são calculados",
            "Um painel mostra tudo"
        ]
    },
    "flight-deals": {
        "en": [
            "It sweeps airline prices",
            "Compares to what each trip usually costs",
            "Flags the outliers",
            "One message a day when it's cheap"
        ],
        "pt": [
            "Pesquisa preços de voos",
            "Compara com o custo habitual",
            "Assinala as exceções",
            "Uma mensagem por dia quando é barato"
        ]
    },
    "price-scout": {
        "en": [
            "It follows the items you want",
            "Keeps each price history",
            "A target or a new low is hit",
            "You get a ping"
        ],
        "pt": [
            "Segue os artigos que queres",
            "Guarda o histórico de preços",
            "Atinge o alvo ou um novo mínimo",
            "Recebes um aviso"
        ]
    },
    "series-watch": {
        "en": [
            "It watches the release slates",
            "A season is confirmed, dated or out",
            "You get told, with where to watch"
        ],
        "pt": [
            "Vigia os calendários de estreias",
            "Uma temporada é confirmada, datada ou estreia",
            "Ficas a saber, com onde ver"
        ]
    },
    "game-alerts": {
        "en": [
            "It watches game news",
            "A limited-time event is announced",
            "You get a ping, then another at go-live"
        ],
        "pt": [
            "Vigia as notícias dos jogos",
            "Um evento temporário é anunciado",
            "Recebes um aviso, e outro no arranque"
        ]
    },
    "lab-browser": {
        "en": [
            "I ask by voice: reply to that post, how many unread emails",
            "A local model works in my browser, one checked step at a time",
            "It answers, or shows me the draft and waits for my word"
        ],
        "pt": [
            "Peço por voz: responde a esse post, quantos emails por ler",
            "Um modelo local trabalha no meu browser, um passo verificado de cada vez",
            "Responde, ou mostra-me o rascunho e espera pela minha palavra"
        ]
    },
    "paywatch": {
        "en": [
            "The payslip arrives by email",
            "Each line is checked against the rules and the days I worked",
            "All good, or a note ready for HR"
        ],
        "pt": [
            "O recibo chega por email",
            "Cada linha é conferida com as regras e os dias trabalhados",
            "Tudo certo, ou uma nota pronta para os RH"
        ]
    },
    "personal-hub": {
        "en": [
            "Each project keeps its own numbers",
            "The dashboard asks them when I open it",
            "One page per year, with charts"
        ],
        "pt": [
            "Cada projeto guarda os seus números",
            "O painel pergunta-lhes quando o abro",
            "Uma página por ano, com gráficos"
        ]
    },
    "redditcast": {
        "en": [
            "The week's top posts are collected",
            "A model writes a two-host script",
            "It is voiced at home",
            "A voice note lands on Monday"
        ],
        "pt": [
            "Recolhe os melhores posts da semana",
            "Um modelo escreve um guião a duas vozes",
            "É narrado em casa",
            "Um áudio chega na segunda"
        ]
    },
    "subs": {
        "en": [
            "Subtitles come in",
            "Translated in validated batches",
            "Each line is checked",
            "European Portuguese out"
        ],
        "pt": [
            "Entram legendas",
            "Traduzidas em lotes validados",
            "Cada linha é verificada",
            "Sai português europeu"
        ]
    }
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


def live_page_url() -> str:
    """The deployed showcase page, or KAYA_SHOWCASE_LIVE_URL for a different one."""
    return os.environ.get("KAYA_SHOWCASE_LIVE_URL", LIVE_PAGE_URL_DEFAULT)


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
    products: Dict[str, Dict[str, list]] = {}
    if payload is not None:
        for entry in payload.get("products") or []:
            project = FEED_NAME_TO_PROJECT.get(entry.get("name"))
            if project is None:
                continue
            stats = [{"en": stat["label"],
                      "pt": STAT_LABEL_PT.get(stat.get("label"), stat.get("label")),
                      "value": stat["value"],
                      "scope": stat.get("scope", "week")}
                     for stat in entry.get("stats") or [] if stat.get("value") is not None]
            if stats:
                products[project] = {"stats": stats}
    return {"products": products, "flows": FLOWS,
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


def _at_a_glance(html: str) -> Dict[str, Optional[int]]:
    """The glance card's project count per language, as a served page states it."""
    counts: Dict[str, Optional[int]] = {}
    for lang, pattern in (("en", r"<b>(\d+) projects?</b>"), ("pt", r"<b>(\d+) projetos</b>")):
        match = re.search(pattern, html)
        counts[lang] = int(match.group(1)) if match else None
    return counts


def _project_cards(html: str) -> int:
    """How many distinct project cards a page shows (both language docs list the same ones)."""
    return len(set(re.findall(r'data-project="([a-z-]+)"', html)))


def _showcase_script(html: str) -> Optional[dict]:
    """The injected window.KAYA_SHOWCASE view, as served (None when the page carries none)."""
    match = re.search(r"window\.KAYA_SHOWCASE = (\{.*?\});</script>", html, flags=re.S)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except ValueError:
        return None


def _dry_run() -> None:
    """What the page would get from the feed, what this checkout's page says and what the
    deployed site says — reads only, writes nothing."""
    payload = fetch_public()
    view = build_view(payload)
    print(f"feed: {labwatch_url()}")
    if payload is None:
        print("feed unreachable — every metrics card shows its schedule, flow and no counters")
    else:
        print(f"feed answered: {len(view['products'])} projects carry their own stats")
    print(f"flows for {len(view['flows'])} projects; prompt for labwatch")

    local = Path(__file__).with_name("static") / "landing.html"
    if local.exists():
        html = local.read_text(encoding="utf-8")
        print(f"local page: lists {_project_cards(html)} project cards; "
              f"at a glance says {_at_a_glance(html)}")

    try:
        response = httpx.get(live_page_url(), timeout=LIVE_TIMEOUT_S, follow_redirects=True)
    except (httpx.HTTPError, ValueError, OSError):
        print(f"live page: {live_page_url()} unreachable")
        return
    title = re.search(r"<title>(.*?)</title>", response.text)
    print(f"live page: {live_page_url()} -> HTTP {response.status_code}, "
          f"{len(response.content)} bytes, title {title.group(1) if title else '(none)'}")
    if response.status_code != 200:
        return
    print(f"live page lists {_project_cards(response.text)} project cards; "
          f"at a glance says {_at_a_glance(response.text)}")
    script = _showcase_script(response.text)
    if script is None:
        print("live page carries no window.KAYA_SHOWCASE script")
        return
    products = script.get("products") or {}
    with_stats = [name for name in sorted(products) if products[name].get("stats")]
    print(f"live showcase script: {len(with_stats)} products carry stats")
    for name in with_stats[:3]:
        first = products[name]["stats"][0]
        print(f"  {name}: '{first['en']}' -> pt '{first['pt']}'")
    flow = ((script.get("flows") or {}).get("kaya") or {}).get("pt") or []
    print(f"  kaya pt flow: {flow}")


def main(argv: Optional[list] = None) -> int:
    if list(argv or []) == ["--dry-run"]:
        _dry_run()
        return 0
    print("usage: python -m src.chat.showcase_metrics --dry-run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
