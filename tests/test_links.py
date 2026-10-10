"""Reading shared links, with no network.

`links._request` (one hop) and `socket.getaddrinfo` are the only seams patched,
so the SSRF guard in `_get` is exercised for real on every test that fetches.
"""
import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import links
from src.chat.links import LinkContent, _Response

PUBLIC = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
CONFIG = {"chat": {"links": {"enabled": True}}}


def resolve_to(mapping):
    def fake(host, *args, **kwargs):
        address = mapping.get(host, "93.184.216.34")
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, 0))]
    return fake


@pytest.fixture
def network(monkeypatch):
    """Route `_request` to canned responses keyed by URL; record every URL requested."""
    responses, requested = {}, []

    def fake_request(url, headers, timeout, max_bytes):
        requested.append(url)
        return responses.get(url) or _Response(404, {}, b"", url)

    monkeypatch.setattr(links, "_request", fake_request)
    monkeypatch.setattr(socket, "getaddrinfo", resolve_to({}))
    return responses, requested


def ok(url, body, content_type="text/html; charset=utf-8"):
    payload = body if isinstance(body, bytes) else (
        json.dumps(body).encode() if not isinstance(body, str) else body.encode())
    return _Response(200, {"content-type": content_type}, payload, url)


def test_extract_urls_strips_punctuation_dedupes_and_limits():
    text = ("vê isto: https://observador.pt/a/b. e https://x.com/u/status/1… "
            "(https://observador.pt/a/b) https://jn.pt/c")
    assert links.extract_urls(text, 2) == ["https://observador.pt/a/b", "https://x.com/u/status/1"]
    assert links.extract_urls("sem links", 2) == []


def test_wikipedia_parentheses_survive():
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    assert links.extract_urls(f"lê {url}.", 2) == [url]


@pytest.mark.parametrize("address", ["127.0.0.1", "192.168.1.238", "10.0.0.5",
                                     "169.254.1.1", "::1", "0.0.0.0", "172.17.0.2"])
def test_non_public_addresses_are_refused(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", resolve_to({"evil.example": address}))
    assert links.is_public_host("evil.example") is False


@pytest.mark.parametrize("host", ["localhost", "pi5.local", "router.lan", ""])
def test_lan_names_are_refused_without_resolving(host):
    assert links.is_public_host(host) is False


def test_public_address_is_allowed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", resolve_to({}))
    assert links.is_public_host("observador.pt") is True


def test_unresolvable_host_is_refused(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("nope")
    monkeypatch.setattr(socket, "getaddrinfo", fail)
    assert links.is_public_host("nope.example") is False


def test_first_hop_to_a_lan_address_is_never_requested(network, monkeypatch):
    _, requested = network
    monkeypatch.setattr(socket, "getaddrinfo", resolve_to({"192.168.1.238": "192.168.1.238"}))
    content = links.fetch("http://192.168.1.238:3000/api/sessions", CONFIG)
    assert requested == []
    assert not content.ok


def test_redirect_to_a_lan_address_is_not_followed(network, monkeypatch):
    responses, requested = network
    monkeypatch.setattr(socket, "getaddrinfo", resolve_to({"pi5.example": "192.168.1.238"}))
    responses["https://short.example/x"] = _Response(
        302, {"location": "http://pi5.example:8088/journal"}, b"", "https://short.example/x")
    content = links.fetch("https://short.example/x", CONFIG)
    assert requested == ["https://short.example/x"]
    assert not content.ok


def test_redirect_loop_stops(network):
    responses, requested = network
    responses["https://loop.example/a"] = _Response(
        302, {"location": "/a"}, b"", "https://loop.example/a")
    assert not links.fetch("https://loop.example/a", CONFIG).ok
    assert len(requested) == 6


def test_article_text_and_title_extracted(network):
    responses, _ = network
    paragraph = ("O Governo aprovou hoje um investimento de 60 milhões de euros para "
                 "reaver as reservas petrolíferas estratégicas que foram vendidas. ") * 4
    responses["https://observador.pt/2026/09/24/reservas"] = ok(
        "https://observador.pt/2026/09/24/reservas",
        f"<html><head><title>Governo quer reaver reservas</title></head><body>"
        f"<article><h1>Governo quer reaver reservas</h1><p>{paragraph}</p>"
        f"<p>{paragraph}</p></article></body></html>")
    content = links.fetch("https://observador.pt/2026/09/24/reservas", CONFIG)
    assert content.ok and content.kind == "article" and content.site == "observador.pt"
    assert "reservas" in content.title.lower()
    assert "60 milhões" in content.text


def test_article_falls_back_to_meta_description(network):
    responses, _ = network
    responses["https://jn.pt/a"] = ok("https://jn.pt/a", (
        '<html><head><meta property="og:title" content="Ventura avança com moção">'
        '<meta property="og:description" content="Moção de censura ao Governo."></head>'
        "<body></body></html>"))
    content = links.fetch("https://jn.pt/a", CONFIG)
    assert content.title == "Ventura avança com moção"
    assert content.text == "Moção de censura ao Governo."


def test_tweet_read_through_fxtwitter_with_quote(network):
    responses, requested = network
    responses["https://api.fxtwitter.com/status/2103729888096068029"] = ok(
        "https://api.fxtwitter.com/status/2103729888096068029",
        {"tweet": {"text": "A musa do Pingo Doce",
                   "author": {"name": "Malaghetto", "screen_name": "Malaghetto74"},
                   "quote": {"text": "vídeo original", "author": {"screen_name": "outro"}}}},
        "application/json")
    content = links.fetch("https://x.com/Malaghetto74/status/2103729888096068029?s=46", CONFIG)
    assert content.kind == "tweet" and content.title == "Malaghetto (@Malaghetto74)"
    assert "A musa do Pingo Doce" in content.text and "Citando @outro: vídeo original" in content.text
    assert requested == ["https://api.fxtwitter.com/status/2103729888096068029"]


def test_xcancel_counts_as_x(network):
    responses, requested = network
    responses["https://api.fxtwitter.com/status/42"] = ok(
        "https://api.fxtwitter.com/status/42", {"tweet": {"text": "olá", "author": {}}},
        "application/json")
    assert links.fetch("https://xcancel.com/a/status/42", CONFIG).text == "olá"


def test_reddit_takes_top_three_comments_by_score(network):
    responses, _ = network
    listing = [
        {"data": {"children": [{"data": {"title": "GTA 6 leak", "selftext": "novo vídeo",
                                         "subreddit": "GTA6"}}]}},
        {"data": {"children": [
            {"kind": "t1", "data": {"body": "low", "score": 1}},
            {"kind": "t1", "data": {"body": "top", "score": 90}},
            {"kind": "t1", "data": {"body": "mid", "score": 40}},
            {"kind": "t1", "data": {"body": "second", "score": 60}},
            {"kind": "more", "data": {}},
        ]}},
    ]
    responses["https://www.reddit.com/r/GTA6/comments/1vteu8s/new_leak.json"] = ok(
        "https://www.reddit.com/r/GTA6/comments/1vteu8s/new_leak.json", listing,
        "application/json")
    content = links.fetch("https://www.reddit.com/r/GTA6/comments/1vteu8s/new_leak/?share=1",
                          CONFIG)
    assert content.title == "r/GTA6: GTA 6 leak"
    assert [line for line in content.text.split("\n\n")] == [
        "novo vídeo", "Comentário: top", "Comentário: second", "Comentário: mid"]


def test_preview_used_when_the_fetch_fails(network):
    content = links.fetch("https://cmjornal.pt/x", CONFIG, preview={
        "url": "https://cmjornal.pt/x", "title": "Suspeito detido",
        "description": "Homem disparou contra turista."})
    assert content.ok and content.kind == "preview" and content.title == "Suspeito detido"


def test_empty_preview_parts_are_ignored(network):
    content = links.fetch("https://observador.pt/y", CONFIG, preview={
        "url": "https://observador.pt/y", "title": "observador.pt",
        "description": "https://observador.pt/y"})
    assert not content.ok


def test_skip_domains_are_not_fetched(network):
    _, requested = network
    links.fetch("https://maps.app.goo.gl/aKGor5tB9JuQTAEY9", CONFIG)
    assert requested == []


def test_non_http_is_not_fetched(network):
    _, requested = network
    assert not links.fetch("ftp://example.com/x", CONFIG).ok
    assert requested == []


def test_text_is_capped():
    config = {"chat": {"links": {"max_words_text": 5}}}
    assert links._cap_words("um dois três quatro cinco seis", 5) == "um dois três quatro cinco"
    assert links.settings(config)["max_words_text"] == 5


def test_synopsis_skips_short_text_without_calling_the_model(monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise AssertionError("the model must not be called")
    monkeypatch.setattr(requests, "post", boom)
    assert links.synopsis(LinkContent("u", "s", title="t", text="curto demais"), CONFIG) == ""


def test_synopsis_cleans_the_answer(monkeypatch):
    import requests

    class Answer:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '"O Governo quer\nreaver reservas."'}}]}

    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return Answer()
    monkeypatch.setattr(requests, "post", fake_post)
    content = LinkContent("u", "observador.pt", title="Reservas", text="palavra " * 40)
    assert links.synopsis(content, CONFIG) == "O Governo quer reaver reservas."
    assert "Reservas" in sent["messages"][0]["content"]


def test_read_caches_a_success_and_not_a_failure(network, tmp_path, monkeypatch):
    responses, requested = network
    config = {"chat": {"links": {"cache_dir": str(tmp_path)}}}
    monkeypatch.setattr(links, "synopsis", lambda content, config: "resumo")
    responses["https://jn.pt/a"] = ok("https://jn.pt/a",
                                      '<meta property="og:title" content="Título">')
    first = links.read("https://jn.pt/a", config)
    second = links.read("https://jn.pt/a", config)
    assert first.synopsis == second.synopsis == "resumo"
    assert requested.count("https://jn.pt/a") == 1
    links.read("https://jn.pt/missing", config)
    links.read("https://jn.pt/missing", config)
    assert requested.count("https://jn.pt/missing") == 2


def test_render_and_excerpt():
    full = LinkContent("https://x.com/a/status/1", "x.com", title="Um\npost",
                       text="texto do post", synopsis="Diz algo.")
    assert links.render(full) == "[Link: Um post (x.com) — Diz algo.]"
    full.synopsis = ""
    assert links.render(full) == "[Link: Um post (x.com)]"
    assert links.render(LinkContent("u", "jn.pt", text="só texto aqui")) == "[Link: só texto aqui (jn.pt)]"
    assert links.render(LinkContent("u", "jn.pt")) == "[Link: jn.pt]"
    assert links.excerpt(LinkContent("u", "s", title="T", text="a b c d"), 2) == "u\nT\na b"


def test_reddit_falls_back_to_oembed_when_json_is_blocked(network):
    responses, _ = network
    responses["https://www.reddit.com/oembed?url=https%3A%2F%2Fwww.reddit.com%2Fr%2FHistory%2Fcomments%2F1w36ecd%2Fsaddam"] = ok(
        "o", {"title": "Saddam's purge in a conference hall", "author_name": "x"},
        "application/json")
    content = links.fetch("https://www.reddit.com/r/History/comments/1w36ecd/saddam/", CONFIG)
    assert content.ok and content.title == "r/History: Saddam's purge in a conference hall"
