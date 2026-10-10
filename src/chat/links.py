"""Reading the links the group shares.

About one link a day lands in the group: Portuguese news (Observador, JN, CM,
Expresso, Público, CNN Portugal), X posts and Reddit threads. Until now the bot
saw only the address, so "o que achas disto?" under a link was answered blind,
and a month later nothing about the article could be found in memory.

A link becomes text the way a photo does. ``read`` fetches the page, extracts
its main text and has the LOCAL model write a one or two sentence synopsis;
``render`` turns that into ``[Link: título (site) — sinopse]``, which goes into
the message itself, so the message log, the ingester, the router and retrieval
need no changes. The fuller ``excerpt`` is handed to a reply that asks about it.

What leaves the box is the URL and nothing else: the page's own site sees a
GET, and for X posts ``api.fxtwitter.com`` sees the post id. No group text is
ever sent anywhere; the synopsis is written locally.

**Every request goes through ``_get``, and ``_get`` refuses non-public hosts on
every hop.** This PC sits on a LAN where the Pi's WAHA and gateway accept it,
so a link to ``http://192.168.1.238:3000/…``, or a public page redirecting
there, must never be fetched. A handler that opened its own client would skip
the check; none does.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

_URL = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_TRAILING = ".,;:!?)]}>'\"…"
_MAX_HOPS = 5
_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/126.0 Safari/537.36")
_TWEET_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com", "xcancel.com",
                "fxtwitter.com", "vxtwitter.com", "fixupx.com"}

SYNOPSIS_PROMPT = (
    "Resume em uma ou duas frases, em português europeu, o que diz o texto abaixo. "
    "Usa só o que está no texto: sem opiniões, sem factos que lá não estejam, e "
    "nomes de pessoas só se aparecerem no texto. Diz directamente o que aconteceu "
    "ou o que se afirma, sem começar por 'O texto' ou 'O artigo'. Responde só com o resumo."
)


@dataclass
class LinkContent:
    """What a link turned out to say."""
    url: str
    site: str
    title: str = ""
    text: str = ""
    kind: str = ""
    ok: bool = False
    synopsis: str = ""


@dataclass
class _Response:
    status: int
    headers: Dict[str, str]
    body: bytes
    url: str

    @property
    def text(self) -> str:
        charset = "utf-8"
        match = re.search(r"charset=([\w\-]+)", self.headers.get("content-type", ""))
        if match:
            charset = match.group(1)
        try:
            return self.body.decode(charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.body)


def settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """``chat.links`` with every default applied."""
    raw = ((config.get("chat", {}) or {}).get("links", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "max_per_message": int(raw.get("max_per_message", 2)),
        "timeout": float(raw.get("timeout", 8.0)),
        "max_bytes": int(raw.get("max_bytes", 2_000_000)),
        "max_words_text": int(raw.get("max_words_text", 3000)),
        "max_words_context": int(raw.get("max_words_context", 600)),
        "fxtwitter": bool(raw.get("fxtwitter", True)),
        "synopsis_max_tokens": int(raw.get("synopsis_max_tokens", 120)),
        "cache_dir": str(raw.get("cache_dir", "data/link_cache")),
        "skip_domains": list(raw.get("skip_domains",
                                     ["maps.app.goo.gl", "sigmakayachat.pt"])),
    }


def is_enabled(config: Dict[str, Any]) -> bool:
    return settings(config)["enabled"]


def extract_urls(text: str, limit: int = 2) -> List[str]:
    """The http(s) links in a message, trailing punctuation removed, first ``limit``."""
    found: List[str] = []
    for match in _URL.finditer(text or ""):
        url = match.group(0)
        while url and url[-1] in _TRAILING:
            if url[-1] == ")" and url.count("(") >= url.count(")"):
                break
            url = url[:-1]
        if url and url not in found:
            found.append(url)
        if len(found) >= limit:
            break
    return found


def _bare_host(host: str) -> str:
    host = (host or "").lower().rstrip(".")
    for prefix in ("www.", "m.", "old.", "mobile."):
        if host.startswith(prefix) and host.count(".") > 1:
            return host[len(prefix):]
    return host


def is_public_host(host: str) -> bool:
    """Whether every address ``host`` resolves to is on the public internet.

    False for anything private, loopback, link-local, multicast, reserved or
    unspecified, for a name that does not resolve, and for LAN-style names.
    """
    host = (host or "").strip("[]").lower().rstrip(".")
    if not host or host == "localhost" or host.endswith((".local", ".lan", ".home", ".internal")):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0].split("%")[0])
        except ValueError:
            return False
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_multicast or address.is_reserved or address.is_unspecified):
            return False
    return True


def _request(url: str, headers: Dict[str, str], timeout: float,
             max_bytes: int) -> Optional[_Response]:
    """One hop, redirects NOT followed, body read up to ``max_bytes``."""
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        with client.stream("GET", url, headers=headers) as response:
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) >= max_bytes:
                    break
            return _Response(status=response.status_code,
                             headers={key.lower(): value for key, value in response.headers.items()},
                             body=bytes(body[:max_bytes]), url=url)


def _get(url: str, config: Dict[str, Any],
         headers: Optional[Dict[str, str]] = None) -> Optional[_Response]:
    """GET with the host checked on every hop. The only way this module touches the network."""
    cfg = settings(config)
    merged = {"User-Agent": _BROWSER_UA, "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8"}
    merged.update(headers or {})
    current = url
    for _ in range(_MAX_HOPS + 1):
        parts = urlsplit(current)
        if parts.scheme not in ("http", "https") or not is_public_host(parts.hostname or ""):
            logger.warning("refused to fetch a non-public address (%s)", parts.hostname)
            return None
        try:
            response = _request(current, merged, cfg["timeout"], cfg["max_bytes"])
        except Exception as exc:  # noqa: BLE001 — a dead link is not an error worth raising
            logger.warning("could not fetch %s: %s", current, exc)
            return None
        if response is None:
            return None
        if response.status in (301, 302, 303, 307, 308) and response.headers.get("location"):
            current = urljoin(current, response.headers["location"])
            continue
        if 200 <= response.status < 300:
            return response
        logger.info("fetching %s returned %s", current, response.status)
        return None
    logger.warning("too many redirects for %s", url)
    return None


def _cap_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    return text.strip() if len(words) <= max_words else " ".join(words[:max_words])


def _tweet(url: str, host: str, config: Dict[str, Any]) -> Optional[LinkContent]:
    match = re.search(r"/status(?:es)?/(\d+)", urlsplit(url).path)
    if not match or not settings(config)["fxtwitter"]:
        return None
    response = _get(f"https://api.fxtwitter.com/status/{match.group(1)}", config)
    if response is None:
        return None
    try:
        tweet = response.json().get("tweet") or {}
    except ValueError:
        return None
    author = tweet.get("author") or {}
    title = (f"{author.get('name', '')} (@{author.get('screen_name', '')})"
             if author.get("screen_name") else author.get("name", ""))
    text = tweet.get("text") or ""
    quote = tweet.get("quote") or {}
    if quote.get("text"):
        quoted_by = (quote.get("author") or {}).get("screen_name", "")
        text += f"\n\nCitando @{quoted_by}: {quote['text']}"
    return LinkContent(url=url, site=host, title=title.strip(), text=text.strip(), kind="tweet")


def _reddit(url: str, host: str, config: Dict[str, Any]) -> Optional[LinkContent]:
    path = urlsplit(url).path.rstrip("/")
    if "/comments/" not in path:
        return None
    response = _get(f"https://www.reddit.com{path}.json", config,
                    {"User-Agent": "KayaBot/1.0 (link reader)"})
    if response is None:
        return _reddit_oembed(url, host, path, config)
    try:
        listing = response.json()
        post = listing[0]["data"]["children"][0]["data"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    comments = []
    try:
        comments = [child["data"] for child in listing[1]["data"]["children"]
                    if child.get("kind") == "t1" and child.get("data", {}).get("body")]
    except (KeyError, IndexError, TypeError):
        pass
    comments.sort(key=lambda comment: int(comment.get("score") or 0), reverse=True)
    text = (post.get("selftext") or "").strip()
    for comment in comments[:3]:
        text += f"\n\nComentário: {comment['body'].strip()}"
    title = f"r/{post.get('subreddit', '')}: {post.get('title', '')}".strip()
    return LinkContent(url=url, site=host, title=title, text=text.strip(), kind="reddit")


def _reddit_oembed(url: str, host: str, path: str,
                   config: Dict[str, Any]) -> Optional[LinkContent]:
    """Reddit answers its ``.json`` with 403 from this address; oEmbed still gives the post."""
    from urllib.parse import quote

    response = _get(f"https://www.reddit.com/oembed?url={quote('https://www.reddit.com' + path, safe='')}",
                    config, {"User-Agent": "KayaBot/1.0 (link reader)"})
    if response is None:
        return None
    try:
        data = response.json()
    except ValueError:
        return None
    subreddit = re.search(r"/r/([^/]+)/", path)
    title = (data.get("title") or "").strip()
    if not title:
        return None
    prefix = f"r/{subreddit.group(1)}: " if subreddit else ""
    words = title.split()
    return LinkContent(url=url, site=host, title=prefix + " ".join(words[:20]),
                       text=title if len(words) > 20 else "", kind="reddit")


def _video(url: str, host: str, config: Dict[str, Any]) -> Optional[LinkContent]:
    from urllib.parse import quote

    response = _get(f"https://www.youtube.com/oembed?url={quote(url, safe='')}&format=json",
                    config)
    if response is None:
        return None
    try:
        data = response.json()
    except ValueError:
        return None
    title = data.get("title") or ""
    if data.get("author_name"):
        title = f"{title} ({data['author_name']})"
    return LinkContent(url=url, site=host, title=title, kind="video")


def _meta(html_text: str, name: str) -> str:
    for pattern in (
        rf'<meta[^>]+(?:property|name)=["\']{name}["\'][^>]+content=["\']([^"\']+)',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']{name}["\']',
    ):
        match = re.search(pattern, html_text, re.IGNORECASE)
        if match:
            import html

            return html.unescape(match.group(1)).strip()
    return ""


def _article(url: str, host: str, config: Dict[str, Any]) -> Optional[LinkContent]:
    response = _get(url, config)
    if response is None:
        return None
    if "html" not in response.headers.get("content-type", "html"):
        return None
    html_text = response.text
    title, text = "", ""
    try:
        import trafilatura

        text = trafilatura.extract(html_text, include_comments=False,
                                   include_tables=False, favor_precision=True) or ""
        metadata = trafilatura.extract_metadata(html_text)
        title = (getattr(metadata, "title", None) or "") if metadata else ""
    except Exception as exc:  # noqa: BLE001 — fall back to the meta tags
        logger.warning("trafilatura failed on %s: %s", url, exc)
    title = title or _meta(html_text, "og:title")
    if not title:
        match = re.search(r"<title[^>]*>([^<]+)</title>", html_text, re.IGNORECASE)
        title = match.group(1).strip() if match else ""
    if not text:
        text = _meta(html_text, "og:description") or _meta(html_text, "description")
    return LinkContent(url=url, site=host, title=title, text=text, kind="article")


def _from_preview(url: str, host: str, preview: Optional[Dict[str, Any]]) -> Optional[LinkContent]:
    """WhatsApp's own preview, minus the parts that say nothing."""
    if not preview:
        return None
    title = (preview.get("title") or "").strip()
    description = (preview.get("description") or "").strip()
    if _bare_host(title) == host:
        title = ""
    if description in (url, preview.get("url", "")):
        description = ""
    if not (title or description):
        return None
    return LinkContent(url=url, site=host, title=title, text=description, kind="preview")


def fetch(url: str, config: Dict[str, Any],
          preview: Optional[Dict[str, Any]] = None) -> LinkContent:
    """What ``url`` says. Never raises; ``ok`` is False when nothing was found."""
    cfg = settings(config)
    parts = urlsplit(url)
    host = _bare_host(parts.hostname or "")
    if parts.scheme not in ("http", "https") or not host:
        return LinkContent(url=url, site=host)
    content: Optional[LinkContent] = None
    if host not in {_bare_host(domain) for domain in cfg["skip_domains"]}:
        try:
            if host in _TWEET_HOSTS or host.startswith("nitter."):
                content = _tweet(url, host, config)
            elif host in ("reddit.com", "redd.it"):
                content = _reddit(url, host, config)
            elif host in ("youtube.com", "youtu.be"):
                content = _video(url, host, config)
            else:
                content = _article(url, host, config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reading %s failed: %s", url, exc)
            content = None
    if content is None or not (content.title or content.text):
        content = _from_preview(url, host, preview) or LinkContent(url=url, site=host)
    content.text = _cap_words(content.text, cfg["max_words_text"])
    content.ok = bool(content.title or content.text)
    return content


def synopsis(content: LinkContent, config: Dict[str, Any]) -> str:
    """One or two sentences, from the local model, about the text only. ``""`` when too short.

    Under 15 words there is nothing to summarise that the title does not already
    say, and asking anyway is how a model starts writing about the headline.
    """
    if len((content.text or "").split()) < 15:
        return ""
    from src.chat.inference_backend import openai_chat_fields, resolve_server_url

    source = f"{content.title}\n\n{_cap_words(content.text, 1200)}".strip()
    try:
        import requests

        response = requests.post(
            f"{resolve_server_url(config).rstrip('/')}/v1/chat/completions",
            json={"messages": [{"role": "user",
                                "content": f"{SYNOPSIS_PROMPT}\n\n---\n{source}\n---"}],
                  "max_tokens": settings(config)["synopsis_max_tokens"],
                  "temperature": 0.2, **openai_chat_fields(config)},
            timeout=60,
        )
        response.raise_for_status()
        answer = (response.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("link synopsis failed for %s: %s", content.url, exc)
        return ""
    answer = " ".join(answer.split()).strip("\"'“”«» ")
    return answer


def _cache_path(url: str, config: Dict[str, Any]) -> Path:
    directory = Path(settings(config)["cache_dir"])
    if not directory.is_absolute():
        directory = BASE_DIR / directory
    return directory / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"


def read(url: str, config: Dict[str, Any],
         preview: Optional[Dict[str, Any]] = None) -> LinkContent:
    """``fetch`` plus ``synopsis``, cached per URL. A failed read is not cached."""
    path = _cache_path(url, config)
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        return LinkContent(**{key: cached[key] for key in LinkContent.__dataclass_fields__
                              if key in cached})
    except (OSError, ValueError, TypeError):
        pass
    content = fetch(url, config, preview)
    if not content.ok:
        return content
    content.synopsis = synopsis(content, config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(content), ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        logger.warning("could not cache %s: %s", url, exc)
    return content


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def render(content: LinkContent) -> str:
    """``[Link: título (site) — sinopse]``, the form it takes inside the message."""
    title, synopsis_text = _one_line(content.title), _one_line(content.synopsis)
    if not title:
        title = " ".join(_one_line(content.text).split()[:25])
    if not title:
        return f"[Link: {content.site}]"
    if synopsis_text:
        return f"[Link: {title} ({content.site}) — {synopsis_text}]"
    return f"[Link: {title} ({content.site})]"


def excerpt(content: LinkContent, max_words: int) -> str:
    """Title and the first ``max_words`` words, for a reply that asks about the link."""
    parts = [part for part in (content.title.strip(),
                               _cap_words(content.text, max_words)) if part]
    head = f"{content.url}\n" if parts else ""
    return head + "\n".join(parts)
