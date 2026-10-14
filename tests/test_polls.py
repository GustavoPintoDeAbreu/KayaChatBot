"""Unit tests for src/chat/polls — building a poll draft from a chat's log.

No GPU/model: the model call is a stub, and the message log is a temp JSONL.
The point of the module is that an option must have been SAID, so the tests
push invented options, rejected ones and repeats at it and check the draft.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import polls

CHAT = "12036300000000@g.us"
OTHER = "99999999999999@g.us"


def _config(**poll_over):
    poll = {"enabled": True}
    poll.update(poll_over)
    return {"chat": {"poll": poll},
            "whatsapp": {"message_log_dir": "data/live_messages"}}


def _record(i, text, sender="Alice", chat_id=CHAT, scope="shared", reply_to_text=""):
    return {
        "id": f"uid-{i}",
        "chat_id": chat_id,
        "sender": sender,
        "text": text,
        "timestamp": 1791900000 + i,
        "scope": scope,
        **({"reply_to_text": reply_to_text} if reply_to_text else {}),
    }


def write_log(tmp_path, records):
    path = tmp_path / "shared.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path.parent)


class FakeGenerate:
    """Returns a canned model answer, one per call, recording every prompt."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, messages, max_new_tokens):
        self.prompts.append(messages)
        if not self.answers:
            raise AssertionError("more generations than canned answers")
        return self.answers.pop(0)


FINAL = (
    "PERGUNTA: Onde é que vamos a jantar?\n"
    "OPÇÃO: 1. O Botequim do Zé\n"
    "OPÇÃO: A Tasca da Rita\n"
    "FORA: O Sushi Bar | rejeitado por todos\n"
    "FORA: Casa de Pastéis\n"
)


class TestSettings:
    def test_defaults_when_section_missing(self):
        cfg = polls.settings({})
        assert cfg["enabled"] is False
        assert cfg["store"] == "data/polls.jsonl"
        assert cfg["lookback_days"] == 90
        assert cfg["max_options"] == 12

    def test_section_values_win(self):
        cfg = polls.settings(_config(enabled=False, max_options=3, temperature=0.5))
        assert cfg["enabled"] is False
        assert cfg["max_options"] == 3
        assert cfg["temperature"] == 0.5


class TestLoadChatMessages:
    def test_filters_by_chat_and_scope(self, tmp_path):
        base = write_log(tmp_path, [
            _record(0, "a", chat_id=CHAT),
            _record(1, "b", chat_id=OTHER),
            _record(2, "c", chat_id=CHAT, scope="dm"),
            _record(3, "d", chat_id=CHAT),
        ])
        records = polls.load_chat_messages(base, "shared", CHAT)
        assert [r["text"] for r in records] == ["a", "d"]

    def test_dedupes_and_sorts_chronologically(self, tmp_path):
        base = write_log(tmp_path, [
            _record(2, "third"),
            _record(0, "first"),
            _record(1, "second"),
            _record(1, "second-duplicate"),
            _record(2, "third-duplicate"),
        ])
        records = polls.load_chat_messages(base, "shared", CHAT)
        assert [r["text"] for r in records] == ["first", "second", "third"]

    def test_keeps_only_the_newest(self, tmp_path):
        base = write_log(tmp_path, [_record(i, f"m{i}") for i in range(10)])
        records = polls.load_chat_messages(base, "shared", CHAT, max_messages=3)
        assert [r["text"] for r in records] == ["m7", "m8", "m9"]

    def test_since_ts_cuts_older_messages(self, tmp_path):
        base = write_log(tmp_path, [_record(i, f"m{i}") for i in range(5)])
        records = polls.load_chat_messages(base, "shared", CHAT,
                                           since_ts=1791900000 + 2)
        assert [r["text"] for r in records] == ["m3", "m4"]


class TestFormatLines:
    def test_shapes_one_line_per_message(self, tmp_path):
        records = [_record(0, "  vamos   ao Botequim  ", sender="Pedro")]
        lines = polls.format_lines(records, 80)
        assert len(lines) == 1
        assert lines[0].startswith("[")
        assert "Pedro:" in lines[0]
        assert "vamos ao Botequim" in lines[0]

    def test_reply_to_is_carried(self, tmp_path):
        records = [_record(0, "sim", reply_to_text="e o Botequim?")]
        lines = polls.format_lines(records, 80)
        assert 'a responder a: "e o Botequim?"' in lines[0]

    def test_long_lines_are_truncated(self, tmp_path):
        records = [_record(0, " ".join(["palavra"] * 200))]
        lines = polls.format_lines(records, 20)
        assert "…" in lines[0]
        assert len(lines[0].split()) <= 20 + 6


class TestChunkLines:
    def test_greedy_by_words(self):
        lines = ["a b c", "d e f g", "h i"]
        chunks = polls.chunk_lines(lines, 5, 10)
        assert chunks == [["a b c"], ["d e f g"], ["h i"]]

    def test_an_overlong_line_gets_its_own_chunk(self):
        lines = ["x " * 100, "a b"]
        chunks = polls.chunk_lines(lines, 10, 10)
        assert chunks[0] == ["x " * 100]
        assert chunks[1] == ["a b"]

    def test_keeps_only_the_newest_chunks(self):
        lines = [f"chunk {i} words here" for i in range(10)]
        chunks = polls.chunk_lines(lines, 4, 3)
        assert len(chunks) == 3
        assert chunks[-1] == [lines[-1]]

    def test_no_lines_no_chunks(self):
        assert polls.chunk_lines([], 10, 3) == []


class TestCleanAndParse:
    def test_clean_item_strips_bullets_and_quotes(self):
        assert polls.clean_item("  1. O Botequim, ") == "O Botequim"
        assert polls.clean_item("“A Tasca da Rita”") == "A Tasca da Rita"
        assert polls.clean_item("- O Bar do Chico - aberto") == "O Bar do Chico, aberto"

    def test_clean_item_keeps_dates_and_times(self):
        # Choosing a day is a main poll use case: a leading digit run is part of
        # the name, not list numbering, so it must survive. Only a "." or ")"
        # after the digits marks real numbering.
        assert polls.clean_item("13 de Dezembro") == "13 de Dezembro"
        assert polls.clean_item("20/12") == "20/12"
        assert polls.clean_item("12h30") == "12h30"
        # Real numbering is still stripped.
        assert polls.clean_item("1. O Botequim") == "O Botequim"
        assert polls.clean_item("2) A Tasca") == "A Tasca"
        assert polls.clean_item("3. 13 de Dezembro") == "13 de Dezembro"

    def test_two_different_dates_do_not_fold_together(self):
        # The bug this guards: if "13 de Dezembro" and "20 de Dezembro" both lost
        # their day, they would fold to the same option and finalize would merge
        # them. With the day kept, both survive.
        draft = polls.finalize(
            "Q", ["13 de Dezembro", "20 de Dezembro"], [],
            polls._fold("13 de dezembro 20 de dezembro"), polls.settings(_config()))
        assert draft.options == ["13 de Dezembro", "20 de Dezembro"]

    def test_parse_final(self):
        question, options, dropped = polls.parse_final(FINAL)
        assert question == "Onde é que vamos a jantar?"
        assert options == ["O Botequim do Zé", "A Tasca da Rita"]
        assert dropped == [("O Sushi Bar", "rejeitado por todos"),
                           ("Casa de Pastéis", "descartada")]

    def test_parse_final_tolerates_missing_sections(self):
        question, options, dropped = polls.parse_final("OPÇÃO: X\nOPÇÃO: Y")
        assert question == ""
        assert options == ["X", "Y"]
        assert dropped == []

    def test_parse_final_empty_names_are_skipped(self):
        _, options, dropped = polls.parse_final("OPÇÃO:\nOPÇÃO: X")
        assert options == ["X"]
        assert dropped == []

    def test_parse_notes_keeps_only_the_labels(self):
        raw = "EM JOGO: O Botequim\nFORA: O Sushi Bar | rejeitado\nblabla\nNADA"
        notes = polls.parse_notes(raw)
        assert notes == ["EM JOGO: O Botequim", "FORA: O Sushi Bar | rejeitado"]

    def test_parse_notes_nada_is_empty(self):
        assert polls.parse_notes("NADA") == []


class TestGrounding:
    CORPUS = polls._fold("vamos ao Botequim do Zé ou à Tasca da Rita, o sushi bar não")

    def test_a_said_option_is_grounded(self):
        assert polls.is_grounded("O Botequim do Zé", self.CORPUS) is True

    def test_an_invented_option_is_not(self):
        assert polls.is_grounded("O Castelo de Sintra", self.CORPUS) is False

    def test_a_generic_name_needs_a_verbatim_match(self):
        # "restaurante" alone is generic; it only grounds when said verbatim.
        assert polls.is_grounded("restaurante", self.CORPUS) is False
        corpus = polls._fold("vamos a um restaurante")
        assert polls.is_grounded("restaurante", corpus) is True


class TestFinalize:
    SETTINGS = polls.settings(_config())

    def test_dedupes_by_fold(self):
        draft = polls.finalize("Q", ["o botequim", "O BOTEQUIM", "o castelo"], [],
                               polls._fold("botequim castelo"), self.SETTINGS)
        assert draft.options == ["o botequim", "o castelo"]

    def test_ungrounded_options_move_to_invented(self):
        draft = polls.finalize("Q", ["o botequim", "o castelo"], [],
                               polls._fold("botequim"), self.SETTINGS)
        assert draft.options == ["o botequim"]
        assert draft.invented == ["o castelo"]

    def test_overlong_options_are_cut(self):
        long = "O " + "x" * 200
        draft = polls.finalize("Q", [long], [], polls._fold("x" * 200), self.SETTINGS)
        assert len(draft.options[0]) <= self.SETTINGS["max_option_chars"]

    def test_options_beyond_the_limit_are_dropped(self):
        settings = polls.settings(_config(max_options=2))
        options = [f"opção {i}" for i in range(4)]
        draft = polls.finalize("Q", options, [], polls._fold(" ".join(options)), settings)
        assert draft.options == ["opção 0", "opção 1"]
        assert ("opção 2", "não coube na sondagem") in draft.dropped
        assert ("opção 3", "não coube na sondagem") in draft.dropped

    def test_an_option_beats_a_drop_of_itself(self):
        draft = polls.finalize("Q", ["o botequim"], [("o botequim", "rejeitado")],
                               polls._fold("botequim"), self.SETTINGS)
        assert draft.options == ["o botequim"]
        assert draft.dropped == []

    def test_dropped_are_deduped(self):
        draft = polls.finalize("Q", ["a"], [("x", "r1"), ("x", "r2")],
                               polls._fold("a"), self.SETTINGS)
        assert draft.dropped == [("x", "r1")]

    def test_an_invented_dropped_name_is_not_listed(self):
        # A dropped option the model made up is not something the group said, so
        # it must not appear in the "Deixei de fora" line. A said one is kept.
        draft = polls.finalize(
            "Q", ["o botequim"],
            [("o castelo inventado", "rejeitado"), ("o sushi bar", "fechado")],
            polls._fold("botequim sushi bar"), self.SETTINGS)
        assert draft.dropped == [("o sushi bar", "fechado")]

    def test_empty_question_gets_the_default(self):
        draft = polls.finalize("", ["a"], [], polls._fold("a"), self.SETTINGS)
        assert draft.question == polls.DEFAULT_QUESTION

    def test_overlong_question_is_cut(self):
        settings = polls.settings(_config(max_question_chars=10))
        draft = polls.finalize("uma pergunta muito longa para caber aqui", ["a"], [],
                               polls._fold("a"), settings)
        assert len(draft.question) <= 10


class TestDraftPoll:
    RECORDS = [
        _record(0, "vamos ao Botequim do Zé?", sender="Pedro"),
        _record(1, "ou a Tasca da Rita", sender="Alice"),
        _record(2, "o sushi bar já foi, ninguém quer", sender="Alice"),
    ]

    def test_no_records_is_no_messages_without_generating(self):
        fake = FakeGenerate([FINAL])
        draft = polls.draft_poll(fake, topic="jantar", request="faz uma poll",
                                 records=[], settings=polls.settings(_config()))
        assert draft.status == polls.STATUS_NO_MESSAGES
        assert fake.prompts == []

    def test_one_chunk_is_one_generation(self):
        fake = FakeGenerate([FINAL])
        draft = polls.draft_poll(fake, topic="jantar", request="faz uma poll",
                                 records=self.RECORDS, settings=polls.settings(_config()))
        assert draft.status == polls.STATUS_OK
        assert len(fake.prompts) == 1
        assert draft.question == "Onde é que vamos a jantar?"
        assert draft.options == ["O Botequim do Zé", "A Tasca da Rita"]
        assert draft.messages_read == 3
        assert draft.chunks == 1
        # The prompt carries the topic and the conversation.
        user = fake.prompts[0][1]["content"]
        assert "Tema da sondagem: jantar" in user
        assert "Botequim do Zé" in user

    def test_several_chunks_are_map_then_reduce(self):
        settings = polls.settings(_config(chunk_words=12, max_chunks=3))
        records = [_record(i, f"mensagem {i} sobre o jantar") for i in range(30)]
        map_answer = "EM JOGO: O Botequim\nFORA: O Sushi Bar | rejeitado"
        fake = FakeGenerate([map_answer, map_answer, map_answer, FINAL])
        draft = polls.draft_poll(fake, topic="jantar", request="faz uma poll",
                                 records=records, settings=settings)
        assert draft.status == polls.STATUS_OK
        assert len(fake.prompts) == 4  # 3 maps + 1 reduce
        assert draft.chunks == 3
        # The reduce prompt carries the notes from the maps.
        reduce_user = fake.prompts[-1][1]["content"]
        assert "Excerto 1:" in reduce_user
        assert "EM JOGO: O Botequim" in reduce_user

    def test_a_generation_failure_is_an_error_draft(self):
        def boom(messages, max_new_tokens):
            raise RuntimeError("llama down")

        draft = polls.draft_poll(boom, topic="jantar", request="x",
                                 records=self.RECORDS, settings=polls.settings(_config()))
        assert draft.status == polls.STATUS_ERROR
        assert draft.messages_read == 3

    def test_an_invented_option_is_kept_out_of_the_poll(self):
        raw = (FINAL + "\nOPÇÃO: O Castelo de Sintra")
        fake = FakeGenerate([raw])
        draft = polls.draft_poll(fake, topic="jantar", request="x",
                                 records=self.RECORDS, settings=polls.settings(_config()))
        assert "O Castelo de Sintra" not in draft.options
        assert "O Castelo de Sintra" in draft.invented

    def test_telemetry_shape(self):
        fake = FakeGenerate([FINAL])
        draft = polls.draft_poll(fake, topic="jantar", request="x",
                                 records=self.RECORDS, settings=polls.settings(_config()))
        tel = draft.telemetry()
        assert tel["poll_status"] == "ok"
        assert tel["poll_options"] == 2
        assert tel["poll_dropped"] == ["O Sushi Bar", "Casa de Pastéis"]
        assert tel["poll_invented"] == []


class TestLeftOutLine:
    TEMPLATE = "Deixei de fora: {items}."

    def test_nothing_dropped_is_empty(self):
        assert polls.left_out_line([], self.TEMPLATE, 5) == ""

    def test_names_and_reasons(self):
        line = polls.left_out_line([("A", "rejeitado"), ("B", "descartada")],
                                   self.TEMPLATE, 5)
        assert line == "Deixei de fora: A (rejeitado), B (descartada)."

    def test_more_than_the_limit_are_folded(self):
        dropped = [(f"O{i}", "r") for i in range(7)]
        line = polls.left_out_line(dropped, self.TEMPLATE, 5)
        assert "O4 (r)" in line
        assert "e mais 2" in line


class TestPollStore:
    def test_record_and_load_roundtrip(self, tmp_path):
        store = polls.PollStore(tmp_path / "polls.jsonl")
        store.record({"poll_id": "p1", "chat_id": CHAT, "question": "Q",
                      "options": ["A", "B"], "dropped": []})
        store.record({"poll_id": "p2", "chat_id": OTHER, "question": "R",
                      "options": ["C"], "dropped": []})
        assert len(store.load()) == 2
        assert [e["poll_id"] for e in store.load(CHAT)] == ["p1"]

    def test_load_missing_file_is_empty(self, tmp_path):
        assert polls.PollStore(tmp_path / "nope.jsonl").load() == []

    def test_a_corrupt_line_is_skipped(self, tmp_path):
        path = tmp_path / "polls.jsonl"
        path.write_text('{"poll_id": "p1", "chat_id": "x"}\nnot json\n',
                        encoding="utf-8")
        entries = polls.PollStore(path).load()
        assert len(entries) == 1
        assert entries[0]["poll_id"] == "p1"
