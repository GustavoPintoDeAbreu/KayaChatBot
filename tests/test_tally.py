"""Counting is done by counting, not by remembering.

Asked for a per-member tally of one word, the bot wrote a confident table off the
back of retrieved chunks: the top user, at 198 messages, came back third with 3,
and the total was out by 8x. Top-k semantic search returns the chunks nearest a
question and cannot answer "how many times" — the answer is a property of every
message, not of any one chunk.

So these tests assert the numbers, the scope boundary, and the refusal to guess.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import tally


def _log(directory: Path, scope: str, rows):
    path = directory / f"{scope}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, (sender, text) in enumerate(rows):
            handle.write(json.dumps({
                "id": f"{scope}-{index}", "chat_id": "c", "sender": sender,
                "text": text, "timestamp": 1700000000 + index, "scope": scope,
            }) + "\n")
    return path


class TestExtractTerm:
    """What to count has to come from the question, or not at all."""

    @pytest.mark.parametrize("question,expected", [
        ('quantas vezes é que disseram "nigga"?', "nigga"),
        ("conta quantas vezes o Gil usou foda-se", "foda-se"),
        ('how many times did Rafa say "bro"?', "bro"),
        ("podes dar uma lista de todos e quantas vezes disseram xpto ?", "xpto"),
    ])
    def test_it_finds_the_term(self, question, expected):
        assert tally.extract_term(question) == expected

    @pytest.mark.parametrize("question", [
        "quem é que diz mais isso?",       # "mais" is not the word being counted
        "quantas vezes dissemos isso?",    # nor is "isso"
        "quantos membros tem o grupo?",    # not a tally at all
        "",
    ])
    def test_an_unidentifiable_term_is_no_term(self, question):
        """A plausible number for the wrong word is the failure being fixed."""
        assert tally.extract_term(question) == ""

    def test_a_quoted_term_wins_over_a_guessed_one(self):
        assert tally.extract_term('disse mais "bacano" que toda a gente') == "bacano"


class TestCounting:
    def test_it_counts_every_occurrence_not_every_message(self, tmp_path):
        _log(tmp_path, "shared", [
            ("Rafa", "bro bro bro"),
            ("Gil", "bro"),
            ("Gil", "nada"),
        ])
        rows, total = tally.count_term("bro", scope="shared", base_dir=tmp_path)
        assert rows == [("Rafa", 3), ("Gil", 1)]
        assert total == 4

    def test_it_is_case_and_accent_insensitive(self, tmp_path):
        _log(tmp_path, "shared", [("Rafa", "NÃO"), ("Gil", "nao"), ("Gil", "Não")])
        rows, total = tally.count_term("nao", scope="shared", base_dir=tmp_path)
        assert total == 3
        assert dict(rows) == {"Rafa": 1, "Gil": 2}

    def test_rows_are_ranked(self, tmp_path):
        _log(tmp_path, "shared", [("Gil", "x"), ("Rafa", "x x x"), ("Pedro", "x x")])
        rows, _ = tally.count_term("x", scope="shared", base_dir=tmp_path)
        assert [name for name, _ in rows] == ["Rafa", "Pedro", "Gil"]

    def test_a_term_nobody_used_counts_zero_rather_than_guessing(self, tmp_path):
        _log(tmp_path, "shared", [("Gil", "olá")])
        rows, total = tally.count_term("xpto", scope="shared", base_dir=tmp_path)
        assert rows == [] and total == 0

    def test_an_empty_term_counts_nothing(self, tmp_path):
        _log(tmp_path, "shared", [("Gil", "olá")])
        assert tally.count_term("  ", scope="shared", base_dir=tmp_path) == ([], 0)

    def test_repeated_ids_are_counted_once(self, tmp_path):
        """WAHA replays its backlog after a reconnect; the log upserts by id."""
        path = tmp_path / "shared.jsonl"
        row = {"id": "same", "sender": "Rafa", "text": "bro", "scope": "shared"}
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n",
                        encoding="utf-8")
        rows, total = tally.count_term("bro", scope="shared", base_dir=tmp_path)
        assert rows == [("Rafa", 1)] and total == 1


class TestAliasFolding:
    """"Gil João" and "Gil" are one person. The raw log has both."""

    def test_the_resolver_merges_them(self, tmp_path):
        members = tmp_path / "members.json"
        members.write_text(json.dumps(
            [{"name": "Gil", "aliases": ["Gil João"]}]), encoding="utf-8")
        from src.data.identity_resolver import SenderResolver

        _log(tmp_path, "shared", [("Gil João", "bro"), ("Gil", "bro")])
        rows, total = tally.count_term(
            "bro", scope="shared", base_dir=tmp_path,
            resolver=SenderResolver(str(members)))
        assert rows == [("Gil", 2)] and total == 2

    def test_without_a_resolver_they_stay_split(self, tmp_path):
        _log(tmp_path, "shared", [("Gil João", "bro"), ("Gil", "bro")])
        rows, _ = tally.count_term("bro", scope="shared", base_dir=tmp_path)
        assert dict(rows) == {"Gil João": 1, "Gil": 1}


class TestScopeIsolation:
    """A DM must not be able to tally the group's history."""

    def test_a_dm_counts_only_its_own_file(self, tmp_path):
        _log(tmp_path, "shared", [("Rafa", "segredo segredo")])
        _log(tmp_path, "dm_abc", [("Gil", "segredo")])
        rows, total = tally.count_term("segredo", scope="dm:abc", base_dir=tmp_path)
        assert rows == [("Gil", 1)] and total == 1

    def test_an_unknown_scope_counts_nothing(self, tmp_path):
        _log(tmp_path, "shared", [("Rafa", "bro")])
        assert tally.count_term("bro", scope="dm:nope", base_dir=tmp_path) == ([], 0)


class TestArchive:
    """The group existed for years before the bot did."""

    def test_the_export_is_included(self, tmp_path):
        _log(tmp_path, "shared", [("Rafa", "bro")])
        archive = tmp_path / "all_messages_cleaned.jsonl"
        archive.write_text("\n".join(
            json.dumps({"sender": "Rafa", "text": "bro"}) for _ in range(5)),
            encoding="utf-8")
        rows, total = tally.count_term(
            "bro", scope="shared", base_dir=tmp_path, archive=archive)
        assert rows == [("Rafa", 6)] and total == 6

    def test_a_missing_export_is_not_an_error(self, tmp_path):
        _log(tmp_path, "shared", [("Rafa", "bro")])
        rows, _ = tally.count_term("bro", scope="shared", base_dir=tmp_path,
                                   archive=tmp_path / "nope.jsonl")
        assert rows == [("Rafa", 1)]


class TestFormatting:
    def test_the_table_carries_the_numbers_and_forbids_adjusting_them(self):
        out = tally.format_table("bro", [("Rafa", 198), ("Gil", 9)], 207)
        assert "Rafa: 198" in out and "Gil: 9" in out and "Total: 207" in out
        assert "Não os arredondes" in out

    def test_nobody_used_it_says_so(self):
        assert "ninguém" in tally.format_table("xpto", [], 0)
