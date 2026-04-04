"""
Pre-training data validation tests.

Tests that training data files meet quality and anti-identity-leak standards
before any GPU training starts.

Run: python -m pytest tests/pipeline/test_training_data.py -v
"""

import json
import re
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAIN_FILE = DATA_DIR / "train_synthetic.jsonl"
VAL_FILE = DATA_DIR / "val_synthetic.jsonl"
DIRECT_FILE = DATA_DIR / "synthetic_kaya.jsonl"   # pre-merge direct examples


def _load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_assistant_texts(records: list) -> list[str]:
    """Extract all assistant-role message texts from training records."""
    texts = []
    for rec in records:
        conversations = rec.get("conversations", [])
        for turn in conversations:
            if turn.get("role") in ("assistant", "gpt"):
                texts.append(turn.get("content", ""))
    return texts


# ---------------------------------------------------------------------------
# Regex patterns for identity leaks (must NOT appear in assistant turns)
# ---------------------------------------------------------------------------

IDENTITY_LEAK_PATTERNS = [
    (re.compile(r"\bmeu\s+(amigo|colega|parceiro)\b", re.IGNORECASE), "meu amigo/colega/parceiro"),
    (re.compile(r"\bvivemos\s+juntos\b", re.IGNORECASE), "vivemos juntos"),
    (re.compile(r"\bj[aá]\s+vivemos\b", re.IGNORECASE), "já vivemos"),
    (re.compile(r"\bconhe[cç]o.{0,20}\bdesde\b", re.IGNORECASE), "conheço ... desde"),
    (re.compile(r"\bsomos\s+amigos\s+desde\b", re.IGNORECASE), "somos amigos desde"),
    (re.compile(r"\bnos\s+conhecemos\s+h[aá]\b", re.IGNORECASE), "nos conhecemos há"),
    (re.compile(r"\bmy\s+friend\b", re.IGNORECASE), "my friend"),
    (re.compile(r"\bwe\s+lived\s+together\b", re.IGNORECASE), "we lived together"),
    (re.compile(r"\bcasa\s+nossa\b", re.IGNORECASE), "casa nossa"),
]

# Sender-prefix leak: assistant turn starts with "[Name]:" — leaked raw message
_SENDER_PREFIX_RE = re.compile(r"^\[.{1,30}\]:")


# ---------------------------------------------------------------------------
# Tests for direct training data (synthetic_kaya.jsonl)
# ---------------------------------------------------------------------------

class TestDirectTrainingData:

    @pytest.fixture(scope="class")
    def direct_records(self):
        if not DIRECT_FILE.exists():
            pytest.skip(f"{DIRECT_FILE} not found — run format_direct_training.py first")
        return _load_jsonl(DIRECT_FILE)

    def test_file_not_empty(self, direct_records):
        assert len(direct_records) > 0, "synthetic_kaya.jsonl is empty"

    def test_no_identity_leaks_in_assistant_turns(self, direct_records):
        """Assistant turns in direct data must not contain first-person member claims."""
        texts = _get_assistant_texts(direct_records)
        violations = []
        for text in texts:
            for pattern, label in IDENTITY_LEAK_PATTERNS:
                if pattern.search(text):
                    violations.append(f"Pattern '{label}' in: {text[:120]!r}")
        assert not violations, (
            f"Found {len(violations)} identity leak(s) in synthetic_kaya.jsonl:\n"
            + "\n".join(violations[:10])
        )

    def test_no_sender_prefix_leaks(self, direct_records):
        """Assistant turns must not start with '[Sender]:' format."""
        texts = _get_assistant_texts(direct_records)
        leaks = [t[:80] for t in texts if _SENDER_PREFIX_RE.match(t)]
        assert not leaks, (
            f"Found {len(leaks)} sender-prefix leak(s) in assistant turns:\n"
            + "\n".join(leaks[:5])
        )

    def test_all_records_have_conversations_key(self, direct_records):
        bad = [i for i, r in enumerate(direct_records) if "conversations" not in r]
        assert not bad, f"Records missing 'conversations' key: indices {bad[:10]}"

    def test_all_records_have_source_key(self, direct_records):
        bad = [i for i, r in enumerate(direct_records) if "source" not in r]
        assert not bad, f"Records missing 'source' key: indices {bad[:10]}"


# ---------------------------------------------------------------------------
# Tests for merged training data (train_synthetic.jsonl)
# ---------------------------------------------------------------------------

class TestMergedTrainingData:

    @pytest.fixture(scope="class")
    def train_records(self):
        if not TRAIN_FILE.exists():
            pytest.skip(f"{TRAIN_FILE} not found — run merge_datasets.py first")
        return _load_jsonl(TRAIN_FILE)

    @pytest.fixture(scope="class")
    def val_records(self):
        if not VAL_FILE.exists():
            pytest.skip(f"{VAL_FILE} not found — run merge_datasets.py first")
        return _load_jsonl(VAL_FILE)

    def test_train_file_not_empty(self, train_records):
        assert len(train_records) > 100, (
            f"Merged train file has only {len(train_records)} records — suspiciously small"
        )

    def test_val_file_not_empty(self, val_records):
        assert len(val_records) > 0, "val_synthetic.jsonl is empty"

    def test_system_prompt_present_in_train(self, train_records):
        """Every conversation in the merged train set must have a system prompt."""
        missing = []
        for i, rec in enumerate(train_records):
            turns = rec.get("conversations", [])
            has_system = any(t.get("role") in ("system",) for t in turns)
            if not has_system:
                missing.append(i)
        assert not missing, (
            f"{len(missing)} training records have no system prompt: first 5 → {missing[:5]}"
        )

    def test_no_identity_leaks_in_train_assistant_turns(self, train_records):
        """Merged training data must not have identity leaks in assistant turns."""
        texts = _get_assistant_texts(train_records)
        violations = []
        for text in texts:
            for pattern, label in IDENTITY_LEAK_PATTERNS:
                if pattern.search(text):
                    violations.append(f"Pattern '{label}': {text[:120]!r}")
                    break  # one violation per text is enough
        assert not violations, (
            f"Found {len(violations)} identity leak(s) in train_synthetic.jsonl:\n"
            + "\n".join(violations[:10])
        )

    def test_system_prompt_consistency(self, train_records):
        """The system prompt in all records must contain the identity guardrail text."""
        guardrail_keywords = ["NOT a group member", "não és um membro", "terceira pessoa", "third person"]
        bad = []
        for i, rec in enumerate(train_records):
            for turn in rec.get("conversations", []):
                if turn.get("role") == "system":
                    content = turn.get("content", "")
                    if not any(kw.lower() in content.lower() for kw in guardrail_keywords):
                        bad.append(i)
                        break
        assert not bad, (
            f"{len(bad)} records have system prompt without identity guardrail: first 5 → {bad[:5]}"
        )

    def test_token_length_distribution(self, train_records):
        """No conversation should be pathologically long (crude word-count proxy)."""
        MAX_WORDS = 3000  # very rough upper bound before real tokenizer
        long_ones = []
        for i, rec in enumerate(train_records):
            total_words = sum(
                len(t.get("content", "").split())
                for t in rec.get("conversations", [])
            )
            if total_words > MAX_WORDS:
                long_ones.append((i, total_words))
        # Warn but do not hard-fail — some long convos are valid
        pct = len(long_ones) / max(len(train_records), 1) * 100
        assert pct < 5, (
            f"{len(long_ones)} records ({pct:.1f}%) exceed {MAX_WORDS} words — "
            "may hit context-length limits during training"
        )

    def test_no_sender_prefix_leaks_in_train(self, train_records):
        """Merged training assistant turns must not start with '[Sender]:'."""
        texts = _get_assistant_texts(train_records)
        leaks = [t[:80] for t in texts if _SENDER_PREFIX_RE.match(t)]
        assert not leaks, (
            f"Found {len(leaks)} sender-prefix leak(s) in train_synthetic.jsonl:\n"
            + "\n".join(leaks[:5])
        )
