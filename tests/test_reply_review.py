"""The bot's own replies must not come back to it as facts.

2026-09-28: at 17:08 the bot mocked Rafa's "three failed startups"; at 17:25 the
rolling summary recorded "O Rafa, empreendedor que lançou três startups
falhadas"; at 17:26, asked what the "jogo do titz" was, the bot answered with
that sentence almost word for word. These tests pin the ledger that records
each reply, the verdicts on it, and the two readers that must respect them: the
summary writer and the ingester.
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import reply_review
from src.chat.gpu_lock import GpuBusyError
from src.chat.reply_review import ReplyLedger, reply_key
from src.chat.summary import ChatSummaryStore, SummaryWriter
from src.data.ingest import build_chunks

CHAT = "351900000000-1500000000@g.us"
ROAST = ("O Rafa é o exemplo perfeito do empreendedorismo de fracasso: lançou três "
         "startups que bateram na parede antes mesmo de começarem a andar.")
ANSWER = "O jantar foi no Ponto Final, no sábado, e o Bernardo pagou a conta toda."


# ── keys ─────────────────────────────────────────────────────────────────────
def test_a_truncated_quote_finds_the_reply_it_came_from():
    quoted = " ".join(ROAST.split()[:14]) + " …"
    assert reply_key(quoted) == reply_key(ROAST)


def test_the_session_prefix_is_not_part_of_the_key():
    assert reply_key(f"Kaya Bot: {ROAST}") == reply_key(ROAST)


def test_punctuation_and_case_do_not_matter():
    assert reply_key("Pure fire only.") == reply_key("pure fire only")


# ── the ledger ───────────────────────────────────────────────────────────────
def test_a_roast_is_recorded_as_a_joke(tmp_path):
    ledger = ReplyLedger(str(tmp_path))
    ledger.record(CHAT, ROAST, "roast")

    entry = ledger.lookup(CHAT, f"Kaya Bot: {ROAST}")
    assert entry["mode"] == "roast"
    assert reply_review.is_joke(entry)
    assert entry["verdict"] is None


def test_ledgers_are_per_chat(tmp_path):
    ledger = ReplyLedger(str(tmp_path))
    ledger.record(CHAT, ROAST, "roast")
    assert ledger.lookup("outro@g.us", ROAST) is None


def test_a_bug_report_condemns_only_the_bot_lines(tmp_path):
    ledger = ReplyLedger(str(tmp_path))
    marked = ledger.mark_bug_report(CHAT, [f"Kaya Bot: {ROAST}", "Gil: Nem ombro tens"])

    assert marked == 1
    assert ledger.lookup(CHAT, ROAST)["verdict"] == reply_review.MAU
    assert ledger.lookup(CHAT, "Gil: Nem ombro tens") is None


def test_the_ledger_keeps_only_the_newest_entries(tmp_path):
    ledger = ReplyLedger(str(tmp_path), max_entries=3)
    for number in range(5):
        ledger.record(CHAT, f"resposta número {number} bem diferente", "factual")

    entries = ledger.entries(CHAT)
    assert len(entries) == 3
    assert reply_key("resposta número 4 bem diferente") in entries
    assert reply_key("resposta número 0 bem diferente") not in entries


def test_a_corrupt_ledger_reads_as_empty(tmp_path):
    ledger = ReplyLedger(str(tmp_path))
    ledger.record(CHAT, ROAST, "roast")
    Path(ledger._path(CHAT)).write_text("{nope", encoding="utf-8")
    assert ledger.lookup(CHAT, ROAST) is None


# ── what the readers see ─────────────────────────────────────────────────────
def test_summary_line_labels_jokes_and_drops_bad_replies():
    line = f"Kaya Bot: {ROAST}"
    assert reply_review.summary_line(line, None) == line
    assert "piada, não é facto" in reply_review.summary_line(line, {"mode": "roast"})
    assert reply_review.summary_line(line, {"mode": "factual", "verdict": "fora"}) is None
    assert reply_review.summary_line("Gil: olá", {"verdict": "mau"}) == "Gil: olá"


def test_keep_quote_needs_a_known_good_reply():
    assert reply_review.keep_quote({"mode": "factual", "verdict": None})
    assert reply_review.keep_quote({"mode": "factual", "verdict": "ok"})
    assert not reply_review.keep_quote(None)
    assert not reply_review.keep_quote({"mode": "roast", "verdict": "ok"})
    assert not reply_review.keep_quote({"mode": "factual", "verdict": "inventado"})


def test_parse_review_tolerates_the_shapes_the_model_writes():
    raw = ("1: FORA - falou do Rafa em vez de responder\n"
           "[2] ok — respondeu\n"
           "3. **INVENTADO** - o Rafa diz que acabou as startups\n"
           "Resposta 3: OK - repetido, o primeiro ganha\n"
           "9: OK - fora de alcance\n"
           "lixo")
    parsed = reply_review.parse_review(raw, 3)
    assert parsed[1][0] == "fora"
    assert parsed[2][0] == "ok"
    assert parsed[3][0] == "inventado"
    assert 9 not in parsed


def test_the_review_prompt_shows_the_reaction_after_each_reply():
    lines = ["Gil: o que é o jogo do titz?", f"Kaya Bot: {ROAST}",
             "Gil: Não explicaste o jogo do titz", "Gustavo: esquece, alucinação"]
    prompt = reply_review.build_review_prompt(lines, [1])
    assert "o que é o jogo do titz?" in prompt.split("Resposta 1")[0]
    assert "alucinação" in prompt.split("Resposta 1")[1]


# ── the summary writer ───────────────────────────────────────────────────────
class ScriptedBackend:
    """Answers the review first, then the summary."""

    def __init__(self, *answers, raises_first=None):
        self.answers = list(answers)
        self.raises_first = raises_first
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if self.raises_first is not None and len(self.calls) == 1:
            raise self.raises_first
        return self.answers.pop(0) if self.answers else "Resumo."


def _writer(tmp_path, backend, ledger):
    config = {
        "chat": {"summary": {"enabled": True, "every_lines": 1, "lock_timeout_seconds": 1},
                 "concurrency": {"max_concurrent": 1, "acquire_timeout": 1}},
        "whatsapp": {},
    }
    store = ChatSummaryStore(str(tmp_path / "summaries"))
    return SummaryWriter(config, backend, store=store, ledger=ledger)


HISTORY = [
    "Rafa: I made 3 startups not just ideas",
    f"Kaya Bot: {ROAST}",
    "Rafa: Well I finished them",
    "Gil: e o jogo do titz?",
    f"Kaya Bot: {ANSWER}",
    "Gustavo: yah esquece",
]


def _summary_prompt(backend):
    return backend.calls[-1][-1]["content"]


def test_a_bad_reply_never_reaches_the_summary(tmp_path):
    ledger = ReplyLedger(str(tmp_path / "replies"))
    ledger.record(CHAT, ROAST, "roast")
    ledger.record(CHAT, ANSWER, "factual")
    backend = ScriptedBackend("1: INVENTADO - o Rafa diz que acabou\n2: OK - respondeu",
                              "Resumo novo.")

    _writer(tmp_path, backend, ledger)._update_one(CHAT, HISTORY, {})

    prompt = _summary_prompt(backend)
    assert "empreendedorismo de fracasso" not in prompt
    assert ANSWER in prompt
    assert "Well I finished them" in prompt
    assert ledger.lookup(CHAT, ROAST)["verdict"] == "inventado"
    assert ledger.lookup(CHAT, ANSWER)["source"] == "review"


def test_a_joke_reaches_the_summary_labelled_as_one(tmp_path):
    ledger = ReplyLedger(str(tmp_path / "replies"))
    ledger.record(CHAT, ROAST, "roast")
    backend = ScriptedBackend("1: OK - era um roast pedido\n2: OK - ok", "Resumo.")

    _writer(tmp_path, backend, ledger)._update_one(CHAT, HISTORY, {})

    assert f"Kaya Bot (piada, não é facto): {ROAST}" in _summary_prompt(backend)


def test_the_marker_comes_from_the_unfiltered_history(tmp_path):
    """Filter the last line and the marker must still point past it, or every
    later update re-reads the whole window."""
    ledger = ReplyLedger(str(tmp_path / "replies"))
    history = HISTORY[:-1]
    backend = ScriptedBackend("1: FORA - x\n2: FORA - y", "Resumo.")
    writer = _writer(tmp_path, backend, ledger)

    writer._update_one(CHAT, history, {})

    assert writer.store.load(CHAT)["marker"] == history[-3:]


def test_replies_already_judged_are_not_reviewed_again(tmp_path):
    ledger = ReplyLedger(str(tmp_path / "replies"))
    ledger.mark(CHAT, ROAST, "mau", "bug report", source="bug")
    ledger.mark(CHAT, ANSWER, "ok", "", source="review")
    backend = ScriptedBackend("Resumo.")

    _writer(tmp_path, backend, ledger)._update_one(CHAT, HISTORY, {})

    assert len(backend.calls) == 1, "only the summary was generated"
    assert "empreendedorismo" not in _summary_prompt(backend)


def test_a_failed_review_still_summarises_with_what_is_known(tmp_path):
    ledger = ReplyLedger(str(tmp_path / "replies"))
    ledger.record(CHAT, ROAST, "roast")
    backend = ScriptedBackend("Resumo.", raises_first=RuntimeError("model down"))
    writer = _writer(tmp_path, backend, ledger)

    writer._update_one(CHAT, HISTORY, {})

    assert writer.store.summary_for(CHAT) == "Resumo."
    assert "piada, não é facto" in _summary_prompt(backend)


def test_a_busy_gpu_defers_the_whole_update(tmp_path):
    import src.chat.gpu_lock as gpu_lock

    ledger = ReplyLedger(str(tmp_path / "replies"))
    backend = ScriptedBackend("1: OK - x", "Resumo.")
    writer = _writer(tmp_path, backend, ledger)

    def busy(*args, **kwargs):
        raise GpuBusyError("held by a reply")

    real_section = gpu_lock.gpu_section
    gpu_lock.gpu_section = busy
    try:
        writer._update_one(CHAT, HISTORY, {})
    finally:
        gpu_lock.gpu_section = real_section

    assert backend.calls == []
    assert writer.store.summary_for(CHAT) == ""


def test_without_a_ledger_the_writer_behaves_as_before(tmp_path):
    backend = ScriptedBackend("Resumo.")

    _writer(tmp_path, backend, None)._update_one(CHAT, HISTORY, {})

    assert len(backend.calls) == 1
    assert ROAST in _summary_prompt(backend)


def test_the_writer_is_told_the_bots_lines_are_not_facts(tmp_path):
    backend = ScriptedBackend("Resumo.")
    _writer(tmp_path, backend, None)._update_one(CHAT, HISTORY, {})
    assert "nunca é um facto" in backend.calls[-1][0]["content"]


# ── the ingester ─────────────────────────────────────────────────────────────
def _message(uid, text, **extra):
    return {"id": uid, "chat_id": CHAT, "sender": "Gil", "text": text,
            "timestamp": 1790612840, "scope": "shared", **extra}


def test_a_quoted_roast_is_not_embedded_as_group_memory(tmp_path):
    ledger = ReplyLedger(str(tmp_path))
    ledger.record(CHAT, ROAST, "roast")
    ledger.record(CHAT, ANSWER, "factual")
    messages = [
        _message("a", "Nem ombro tens", reply_to_id="X1", reply_to_bot=True,
                 reply_to_text=" ".join(ROAST.split()[:14]) + " …"),
        _message("b", "boa", reply_to_id="X2", reply_to_bot=True, reply_to_text=ANSWER),
        _message("c", "concordo", reply_to_id="X3", reply_to_text="o Peter disse isto"),
    ]
    entries = ledger.entries(CHAT)

    chunks, _ = build_chunks(
        messages, "shared",
        keep_bot_quote=lambda m: reply_review.keep_quote(
            entries.get(reply_key(m.get("reply_to_text", "")))))

    text = chunks[0]["text"]
    assert "empreendedorismo" not in text
    assert "Gil (a responder ao bot): Nem ombro tens" in text
    assert ANSWER in text, "a factual answer that was not judged bad keeps its context"
    assert '(a responder a "o Peter disse isto")' in text, "members' quotes are untouched"


def test_without_a_filter_quotes_are_kept_as_before():
    messages = [_message("a", "Nem ombro tens", reply_to_id="X1", reply_to_bot=True,
                         reply_to_text=ROAST)]
    chunks, _ = build_chunks(messages, "shared")
    assert "empreendedorismo" in chunks[0]["text"]


# ── the adapter ──────────────────────────────────────────────────────────────
def test_the_adapter_records_each_reply_with_its_mode(tmp_path):
    from tests.test_whatsapp_adapter import RoutedReply, dm_event, make_routed_adapter

    adapter = make_routed_adapter(tmp_path, RoutedReply(ROAST, mode="roast"))
    adapter.reply_ledger = ReplyLedger(str(tmp_path / "replies"))

    adapter.handle_event(dm_event("roast o Rafa"))

    entries = adapter.reply_ledger.entries("351911111111@c.us")
    assert [entry["mode"] for entry in entries.values()] == ["roast"]


def test_a_bug_report_marks_the_replies_it_was_filed_against(tmp_path):
    from tests.test_whatsapp_adapter import dm_event, make_report_adapter

    adapter, _, _ = make_report_adapter(tmp_path)
    adapter.reply_ledger = ReplyLedger(str(tmp_path / "replies"))
    adapter.handle_event(dm_event("olá"))
    adapter.handle_event(dm_event("/bug respondeste ao lado"))

    verdicts = [entry["verdict"] for entry in
                adapter.reply_ledger.entries("351911111111@c.us").values()]
    assert verdicts == ["mau"]


def test_a_reply_to_the_bot_is_flagged_in_the_log(tmp_path):
    from tests.test_whatsapp_adapter import BOT_LID, make_report_adapter, noweb_group

    adapter, _, _ = make_report_adapter(tmp_path)
    adapter.handle_event(noweb_group("Nem ombro tens", reply_to_lid=BOT_LID,
                                     quoted={"conversation": ROAST}))
    adapter.handle_event(noweb_group("pois", reply_to_lid="333333333333333@lid",
                                     quoted={"conversation": "isto é do Gil"}))

    rows = [json.loads(line) for path in glob.glob(str(tmp_path / "msglog" / "*.jsonl"))
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    flags = {row["text"]: row.get("reply_to_bot", False) for row in rows}
    assert flags == {"Nem ombro tens": True, "pois": False}
