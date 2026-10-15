"""Recaps of one topic read the log, not the top-k chunks nearest the question.

A planning thread goes back and forth for days. The recap selects the messages on
the topic (keyword seeds, their neighbours in the same chat, the time windows of
the semantic hits) and hands the model a dated, attributed transcript.
"""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import recap
from src.chat.response_utils import clean_response
from src.data.message_log import MessageLog

BASE = 1797000000
GROUP = "120363000000000001@g.us"
OTHER = "120363000000000002@g.us"
LISBON = ZoneInfo("Europe/Lisbon")


def _row(text, timestamp, chat_id=GROUP, sender="Gil"):
    return {"text": text, "timestamp": timestamp, "chat_id": chat_id, "sender": sender}


def _log(tmp_path, rows, scope="shared"):
    log = MessageLog(str(tmp_path / "live"))
    for index, (text, timestamp, chat_id, sender) in enumerate(rows):
        log.append(chat_id=chat_id, message_id=f"{scope}-{index}", sender=sender,
                   text=text, timestamp=timestamp, scope=scope)
    return tmp_path / "live"


def _config(tmp_path, **recap_over):
    return {"whatsapp": {"message_log_dir": str(tmp_path / "live")},
            "power": {"timezone": "Europe/Lisbon"},
            "chat": {"recap": {"common_floor": 5, **recap_over}}}


# ── keywords ─────────────────────────────────────────────────────────────────
def test_keywords_drop_the_request_and_fold_accents():
    assert recap.topic_keywords("o que decidimos sobre o almoço de natal?") == ["almoco", "natal"]


def test_keywords_are_capped_and_deduplicated():
    words = recap.topic_keywords("natal natal praia barco comida vinho festa jantar sitio carro")
    assert words.count("natal") == 1 and len(words) == 6


# ── selection ────────────────────────────────────────────────────────────────
def _lunch_rows():
    rows = [_row("almoço amanhã?", BASE + index * 3600) for index in range(10)]
    rows += [_row("o natal é na casa da Rita", BASE + 50000),
             _row("natal sem prendas este ano", BASE + 60000)]
    return sorted(rows, key=lambda row: row["timestamp"])


def test_a_rare_keyword_seeds_on_its_own():
    selected, seeds = recap.select_rows(_lunch_rows(), ["almoco", "natal"], common_floor=5,
                                        neighbour_rows=0)
    seed_texts = {selected[index]["text"] for index in seeds}
    assert seed_texts == {"o natal é na casa da Rita", "natal sem prendas este ano"}


def test_a_common_keyword_needs_a_second_one():
    rows = _lunch_rows() + [_row("almoço de natal no sábado", BASE + 70000)]
    selected, seeds = recap.select_rows(rows, ["almoco", "natal"], common_floor=5, neighbour_rows=0)
    seed_texts = {selected[index]["text"] for index in seeds}
    assert "almoço de natal no sábado" in seed_texts
    assert "almoço amanhã?" not in seed_texts


def test_a_single_keyword_topic_seeds_on_any_match():
    rows = [_row("natal outra vez", BASE + index * 3600) for index in range(10)]
    selected, seeds = recap.select_rows(rows, ["natal"], common_floor=5)
    assert len(seeds) == 10


def test_neighbours_come_from_the_same_chat_within_the_gap():
    rows = [_row("almoço de natal no Ramiro?", BASE),
            _row("ponto alto", BASE + 30, chat_id=OTHER),
            _row("e o Zé da Mouraria?", BASE + 60),
            _row("vou dormir", BASE + 7200)]
    selected, _ = recap.select_rows(rows, ["almoco", "natal"])
    texts = [row["text"] for row in selected]
    assert "e o Zé da Mouraria?" in texts
    assert "vou dormir" not in texts and "ponto alto" not in texts


def test_semantic_windows_pull_in_rows_without_keywords():
    rows = [_row("almoço de natal?", BASE), _row("bora ao Ramiro", BASE + 500)]
    selected, seeds = recap.select_rows(rows, ["almoco", "natal"], windows=[(BASE + 400, BASE + 600)],
                                        neighbour_rows=0)
    texts = [row["text"] for row in selected]
    assert "bora ao Ramiro" in texts
    assert texts.index("bora ao Ramiro") not in seeds


# ── semantic hits ────────────────────────────────────────────────────────────
def test_chunk_windows_parse_live_chunks_and_keep_archive_text():
    chunks = [
        {"metadata": {"source": "live", "scope": "shared"},
         "timestamp_start": "2026-07-01T12:00:00", "timestamp_end": "2026-07-01T12:30:00", "text": "x"},
        {"metadata": {"scope": "shared"}, "text": "antigo"},
        {"metadata": {"source": "live", "scope": "dm:abc"},
         "timestamp_start": "2026-07-01T12:00:00", "timestamp_end": "2026-07-01T12:30:00", "text": "y"},
    ]
    windows, archive = recap.chunk_windows(chunks, "shared")
    assert windows == [(1782907200, 1782909000)]
    assert archive == ["antigo"]


# ── the log ──────────────────────────────────────────────────────────────────
def test_load_rows_respects_readable_scopes(tmp_path):
    _log(tmp_path, [("almoço de natal", BASE, GROUP, "Gil")], scope="shared")
    _log(tmp_path, [("segredo do natal", BASE + 10, "351911111111@c.us", "Rafa")], scope="dm:abc")
    shared = [row["text"] for row in recap.load_rows(tmp_path / "live", "shared")]
    dm = [row["text"] for row in recap.load_rows(tmp_path / "live", "dm:abc")]
    assert shared == ["almoço de natal"]
    assert dm == ["almoço de natal", "segredo do natal"]


def test_load_rows_skips_failed_image_descriptions(tmp_path):
    """A refused description is stripped by ``ingest.strip_failed_descriptions``;
    a row left empty by that is not a message about anything."""
    refused = "[Imagem: Não consigo ver a imagem.]"
    assert not recap.strip_failed_descriptions(refused).strip()
    _log(tmp_path, [(refused, BASE, GROUP, "Gil"), ("almoço de natal", BASE + 5, GROUP, "Rafa")])
    assert [row["text"] for row in recap.load_rows(tmp_path / "live", "shared")] == ["almoço de natal"]


def test_format_line_uses_local_time_and_the_resolver():
    class Resolver:
        def resolve(self, raw):
            return {"Gil João": "Gil"}.get(raw, raw)

    line = recap.format_line({"sender": "Gil João", "text": "bora", "timestamp": 1782907200},
                             Resolver(), LISBON)
    assert line == "[01/07 13:00] Gil: bora"
    quoted = recap.format_line({"sender": "Gil", "text": "sim", "timestamp": 1782907200,
                                "reply_to_text": "Ramiro?"}, None, LISBON)
    assert '(a responder a "' in quoted


def test_fit_budget_drops_neighbours_before_seeds_oldest_first():
    lines = ["aaaa", "bbbb", "cccc", "dddd"]
    kept, dropped = recap.fit_budget(lines, {1, 3}, max_chars=10)
    assert kept == ["bbbb", "dddd"] and dropped == 2


def test_protect_speaker_lines():
    text = "*Quem disse o quê*\nGustavo: Ramiro\nGil: Zé"
    protected = recap.protect_speaker_lines(text, "Gustavo")
    assert protected == "*Quem disse o quê*\n- Gustavo: Ramiro\nGil: Zé"
    assert recap.protect_speaker_lines(text, "") == text
    assert "Gil: Zé" in clean_response(protected, user_name="Gustavo")


# ── gather ───────────────────────────────────────────────────────────────────
def test_a_request_with_no_subject_recaps_the_chat(tmp_path):
    """Every word of "o que é que decidimos?" is a request word, so it names no
    subject: the recap reads this chat's recent messages, never another chat's."""
    _log(tmp_path, [("bora ao Ramiro", BASE, GROUP, "Gil"),
                    ("noutro grupo", BASE + 10, OTHER, "Rafa")])
    now = datetime.fromtimestamp(BASE + 3600, tz=LISBON)
    for request in ("o que é que decidimos?", "o que se discutiu?", "tl;dr?", "o que é que cada um quer?"):
        block, instruction, stats = recap.gather(_config(tmp_path), request, "shared",
                                                 request=request, chat_id=GROUP, now=now)
        assert stats["recap_mode"] == "chat" and stats["recap_keywords"] == []
        assert "Gil: bora ao Ramiro" in block and "noutro grupo" not in block
        assert instruction.startswith(f"Responde a «{request}»") and "assuntos principais" in instruction


def test_a_named_period_bounds_the_chat_recap(tmp_path):
    now = datetime(2026, 12, 18, 12, 0, tzinfo=LISBON)
    yesterday = int(datetime(2026, 12, 17, 20, 0, tzinfo=LISBON).timestamp())
    last_week = int(datetime(2026, 12, 10, 20, 0, tzinfo=LISBON).timestamp())
    _log(tmp_path, [("ontem falámos do Ramiro", yesterday, GROUP, "Gil"),
                    ("isto foi há uma semana", last_week, GROUP, "Rafa")])
    block, instruction, stats = recap.gather(_config(tmp_path), "o que é que falámos ontem?", "shared",
                                             request="o que é que falámos ontem?", chat_id=GROUP, now=now)
    assert stats["recap_period"] == "ontem" and stats["recap_keywords"] == []
    assert "ontem falámos do Ramiro" in block and "há uma semana" not in block
    assert "(ontem)" in instruction


def test_a_subject_in_the_request_keeps_the_topic_recap(tmp_path):
    _log(tmp_path, [("almoço de natal no Ramiro?", BASE, GROUP, "Rafa")])
    _, _, stats = recap.gather(_config(tmp_path), "almoço de natal", "shared",
                               request="o que decidimos sobre o almoço de natal?", chat_id=GROUP)
    assert stats["recap_mode"] == "topic" and stats["recap_keywords"] == ["almoco", "natal"]


def test_an_empty_chat_says_so(tmp_path):
    _log(tmp_path, [("noutro grupo", BASE, OTHER, "Rafa")])
    block, instruction, _ = recap.gather(_config(tmp_path), "", "shared", request="resume",
                                         chat_id=GROUP, now=datetime.fromtimestamp(BASE, tz=LISBON))
    assert block == "" and "não inventes" in instruction


def test_gather_with_no_matches_says_so(tmp_path):
    _log(tmp_path, [("bom dia", BASE, GROUP, "Gil")])
    block, instruction, _ = recap.gather(_config(tmp_path), "viagem a Roma", "shared")
    assert block == "" and "viagem a Roma" in instruction and "não inventes" in instruction


def test_gather_builds_an_attributed_dated_block(tmp_path):
    _log(tmp_path, [("almoço de natal no Ramiro?", BASE, GROUP, "Rafa"),
                    ("e o Zé da Mouraria?", BASE + 60, GROUP, "Gil")])
    block, instruction, stats = recap.gather(_config(tmp_path), "almoço de natal", "shared")
    assert "Gil: e o Zé da Mouraria?" in block
    assert "*Decidido*" in instruction and stats["recap_rows"] > 0


def test_gather_survives_a_failing_retriever(tmp_path):
    class Broken:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("chroma down")

    _log(tmp_path, [("almoço de natal no Ramiro?", BASE, GROUP, "Rafa")])
    block, _, _ = recap.gather(_config(tmp_path), "almoço de natal", "shared", retriever=Broken())
    assert "Ramiro" in block


def test_gather_passes_the_scope_to_retrieval(tmp_path):
    class Recording:
        scopes = []

        def retrieve(self, query, top_k=None, scope=None, **kwargs):
            self.scopes.append(scope)
            return []

    retriever = Recording()
    recap.gather(_config(tmp_path), "almoço de natal", "dm:abc", retriever=retriever)
    assert retriever.scopes == ["dm:abc"]


def test_whatsapp_bold_leaves_single_asterisks_alone():
    assert recap.whatsapp_bold("**Decidido**\n*Em aberto*") == "*Decidido*\n*Em aberto*"
    assert recap.whatsapp_bold("**\\*Decidido\\***") == "*Decidido*"


def test_a_topic_recap_prefers_the_chat_it_is_asked_in(tmp_path):
    _log(tmp_path, [("o restaurante Pombal é bom", BASE, OTHER, "Rafa"),
                    ("restaurante: voto no Ramiro", BASE + 10, GROUP, "Gil")])
    block, _, stats = recap.gather(_config(tmp_path), "restaurante", "shared",
                                   request="o que disse o Gil sobre o restaurante?", chat_id=GROUP)
    assert stats["recap_this_chat"] and "voto no Ramiro" in block and "Pombal" not in block
    block, _, stats = recap.gather(_config(tmp_path), "Pombal", "shared",
                                   request="o que se disse do Pombal?", chat_id=GROUP)
    assert not stats["recap_this_chat"] and "Pombal é bom" in block
