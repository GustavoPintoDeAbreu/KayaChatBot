"""
Automated conversation testing framework with LLM judge.

Generates test scenarios from group_knowledge.json facts, runs multi-turn
conversations where the judge LLM asks questions and the local model responds,
then scores each response on a 0–5 rubric across four dimensions.

CLI usage:
    python src/testing/conversation_tester.py \\
        --scenarios 20 --provider xai --output reports/eval_report.json
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

# Ensure repo root is on sys.path so sibling packages can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScoreBreakdown:
    """Per-response scores on the 0–5 rubric.

    ``identity_adherence`` and ``factual_grounding`` are new dimensions added for
    golden-test evaluation.  They default to 5.0 (pass) so that existing code
    paths that do not ask the judge for them are not penalised.
    """

    factual_accuracy: float
    relevance: float
    language_quality: float
    tone: float
    # New dimensions — default 5.0 (pass) for backward compatibility
    identity_adherence: float = 5.0
    factual_grounding: float = 5.0
    # Set to True when a forbidden pattern was matched (auto-fails identity)
    identity_pattern_matched: bool = False

    @property
    def average(self) -> float:
        """Average of the original four dimensions (backward-compatible)."""
        return (
            self.factual_accuracy + self.relevance + self.language_quality + self.tone
        ) / 4

    @property
    def extended_average(self) -> float:
        """Average across all six scored dimensions."""
        return (
            self.factual_accuracy
            + self.relevance
            + self.language_quality
            + self.tone
            + self.identity_adherence
            + self.factual_grounding
        ) / 6

    @property
    def failed(self) -> bool:
        """True if any single dimension scores below 3 or a forbidden pattern matched."""
        return self.identity_pattern_matched or any(
            v < 3
            for v in [
                self.factual_accuracy,
                self.relevance,
                self.language_quality,
                self.tone,
                self.identity_adherence,
                self.factual_grounding,
            ]
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "factual_accuracy": self.factual_accuracy,
            "relevance": self.relevance,
            "language_quality": self.language_quality,
            "tone": self.tone,
            "identity_adherence": self.identity_adherence,
            "factual_grounding": self.factual_grounding,
            "identity_pattern_matched": self.identity_pattern_matched,
            "average": self.average,
            "extended_average": self.extended_average,
        }


@dataclass
class ConversationTurn:
    """A single turn in a test conversation."""

    role: str  # "judge" | "model"
    content: str
    scores: Optional[ScoreBreakdown] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.scores is not None:
            d["scores"] = self.scores.to_dict()
        return d


@dataclass
class LLMScenarioResult:
    """Result for a single test scenario."""

    scenario_id: str
    subject: str
    category: str
    fact_excerpt: str
    turns: List[ConversationTurn] = field(default_factory=list)
    failure: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def average_scores(self) -> Optional[Dict[str, float]]:
        scored = [t.scores for t in self.turns if t.scores is not None]
        if not scored:
            return None
        n = len(scored)
        return {
            "factual_accuracy": sum(s.factual_accuracy for s in scored) / n,
            "relevance": sum(s.relevance for s in scored) / n,
            "language_quality": sum(s.language_quality for s in scored) / n,
            "tone": sum(s.tone for s in scored) / n,
            "identity_adherence": sum(s.identity_adherence for s in scored) / n,
            "factual_grounding": sum(s.factual_grounding for s in scored) / n,
            "average": sum(s.average for s in scored) / n,
            "extended_average": sum(s.extended_average for s in scored) / n,
            "identity_patterns_matched": sum(1 for s in scored if s.identity_pattern_matched),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "subject": self.subject,
            "category": self.category,
            "fact_excerpt": self.fact_excerpt,
            "turns": [t.to_dict() for t in self.turns],
            "failure": self.failure,
            "failure_reasons": self.failure_reasons,
            "error": self.error,
            "average_scores": self.average_scores(),
        }


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

_MEMBER_TEMPLATES = [
    "Who is {subject}?",
    "Tell me about {subject}.",
    "What can you tell me about {subject}?",
    "What do you know about {subject}?",
    "Give me a brief summary of who {subject} is.",
]

_GROUP_TEMPLATES = [
    "Tell me about {subject}.",
    "What is the {subject}?",
    "Can you describe {subject}?",
    "What do you know about {subject}?",
]

_FOLLOWUP_TEMPLATES = [
    "Can you tell me more about that?",
    "What else can you share about {subject}?",
    "Is there anything interesting about {subject} I should know?",
    "Could you elaborate a bit more?",
]


def generate_scenarios(
    facts: List[Dict[str, Any]], num_scenarios: int
) -> List[Dict[str, Any]]:
    """Build test scenario descriptors from knowledge facts.

    Each scenario contains:
        - id, subject, category, fact_text, fact_excerpt
        - opening_question: the first judge question
        - followup_question: second judge question (for multi-turn)
    """
    if not facts:
        return []

    # Cycle through facts if more scenarios are requested than facts available
    fact_pool = (facts * (num_scenarios // len(facts) + 1))[:num_scenarios]
    random.shuffle(fact_pool)

    scenarios = []
    for i, fact in enumerate(fact_pool):
        subject = fact.get("subject", "the group")
        category = fact.get("category", "group")
        fact_text = fact.get("text", "")

        templates = _MEMBER_TEMPLATES if category == "member" else _GROUP_TEMPLATES
        opening = random.choice(templates).format(subject=subject)
        followup = random.choice(_FOLLOWUP_TEMPLATES).format(subject=subject)

        scenarios.append(
            {
                "id": f"scenario_{i + 1:03d}",
                "subject": subject,
                "category": category,
                "fact_text": fact_text,
                "fact_excerpt": fact_text[:300],
                "opening_question": opening,
                "followup_question": followup,
            }
        )

    return scenarios


# ---------------------------------------------------------------------------
# Local model interface
# ---------------------------------------------------------------------------


class LocalModel:
    """Thin wrapper around the fine-tuned model for generating responses.

    Falls back to mock responses when the model is unavailable (CI / no GPU).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._available = False
        self._try_load()

    def _try_load(self) -> None:
        """Try to load the fine-tuned model; silently fall back if unavailable."""
        try:
            import torch  # noqa: F401

            from unsloth import FastLanguageModel  # noqa: F401

            model_dir = self.config.get("training", {}).get("output_dir", "")
            if not model_dir or not os.path.exists(model_dir):
                return

            max_seq = self.config.get("model", {}).get("max_seq_length", 4096)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_dir,
                max_seq_length=max_seq,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        if not self._available:
            return self._mock_response(prompt)

        import torch

        inputs = self.tokenizer([prompt], return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        inf_config = self.config.get("inference", {})
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=inf_config.get("max_new_tokens", 256),
            temperature=inf_config.get("temperature", 0.7),
            top_p=inf_config.get("top_p", 0.9),
            repetition_penalty=inf_config.get("repetition_penalty", 1.1),
            use_cache=True,
        )
        response = self.tokenizer.batch_decode(outputs)[0]
        # Strip the input prompt portion from the output
        if prompt in response:
            response = response[len(prompt) :].strip()
        return response

    @staticmethod
    def _mock_response(prompt: str) -> str:  # noqa: ARG004
        """Return a placeholder response when the model is unavailable."""
        return (
            "[MOCK] The fine-tuned model is not available in this environment. "
            "This is a placeholder response for testing purposes."
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

SCORING_SYSTEM_PROMPT = """\
You are an expert evaluator for a Portuguese friend group chat assistant called KayaBot.

KayaBot is a BOT assistant with access to group conversation history.
It is NOT a group member — it should always refer to members in the THIRD PERSON.
It must NEVER say things like "meu amigo", "vivemos juntos", "conheço-o desde...", or any first-person claim.

Evaluate the given response on a 0–5 integer scale for each of these six dimensions:

1. **factual_accuracy** — Does the response contain correct facts consistent with the reference knowledge?
   - 5: All facts correct and complete
   - 4: Minor omissions, no errors
   - 3: Mostly correct, small inaccuracies
   - 2: Several factual errors or important facts missing
   - 1: Mostly incorrect
   - 0: Completely wrong or no relevant content

2. **relevance** — Does the response actually answer the question asked?
   - 5: Directly and fully addresses the question
   - 4: Mostly relevant with minor tangents
   - 3: Partially relevant
   - 2: Only loosely related to the question
   - 1: Barely relevant
   - 0: Does not address the question at all

3. **language_quality** — Grammar, fluency, and naturalness in Portuguese or English.
   - 5: Excellent, natural-sounding text
   - 4: Good, minor awkwardness
   - 3: Acceptable, some grammatical issues
   - 2: Noticeable errors, hard to follow
   - 1: Very poor grammar or fluency
   - 0: Unintelligible or gibberish

4. **tone** — Appropriate for a helpful, friendly group-chat assistant (not robotic, not overly casual).
   - 5: Perfect assistant tone — warm, clear, friendly
   - 4: Mostly appropriate, minor issues
   - 3: Acceptable but slightly off (too formal or too casual)
   - 2: Noticeably inappropriate tone
   - 1: Very wrong tone (cold, rude, or inappropriately familiar)
   - 0: Completely inappropriate

5. **identity_adherence** — Does the bot speak as a bot (third person), NOT as a group member?
   - 5: Always third person ("O Peter trabalha em..."); no first-person claims
   - 4: Mostly third person; one borderline phrase
   - 3: Mostly OK but some ambiguous phrasing
   - 2: Some first-person claims about members
   - 1: Frequent first-person claims ("meu amigo", "vivemos juntos")
   - 0: Speaks entirely as a group member with personal memories

6. **factual_grounding** — Are statements grounded in the provided context, not hallucinated?
   - 5: All claims traceable to the provided reference knowledge
   - 4: One minor detail not in reference but plausible
   - 3: Mix of grounded and unverifiable claims
   - 2: Several claims not supported by reference
   - 1: Mostly unsupported claims
   - 0: Entirely hallucinated, contradicts reference

Respond ONLY with a valid JSON object (no markdown, no explanation), for example:
{"factual_accuracy": 4, "relevance": 5, "language_quality": 4, "tone": 4, "identity_adherence": 5, "factual_grounding": 4}
"""


def build_scoring_prompt(
    question: str,
    reference_fact: str,
    response: str,
) -> str:
    """Build the user message sent to the judge when scoring a response."""
    return (
        f"Reference knowledge:\n{reference_fact}\n\n"
        f"Question asked:\n{question}\n\n"
        f"Response to evaluate:\n{response}\n\n"
        "Provide your scores as a JSON object."
    )


def parse_scores(raw: str) -> ScoreBreakdown:
    """Parse LLM scoring output into a ScoreBreakdown, with graceful fallback."""
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip())

    # Extract the first JSON object (handles extra surrounding text)
    match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ScoreBreakdown(
            factual_accuracy=0.0,
            relevance=0.0,
            language_quality=0.0,
            tone=0.0,
        )

    def _clamp(val: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(5.0, float(val)))
        except (TypeError, ValueError):
            return default

    return ScoreBreakdown(
        factual_accuracy=_clamp(data.get("factual_accuracy")),
        relevance=_clamp(data.get("relevance")),
        language_quality=_clamp(data.get("language_quality")),
        tone=_clamp(data.get("tone")),
        identity_adherence=_clamp(data.get("identity_adherence", 5.0), default=5.0),
        factual_grounding=_clamp(data.get("factual_grounding", 5.0), default=5.0),
    )


# Regex patterns that indicate the bot is speaking as a group member (auto-fail identity)
_IDENTITY_LEAK_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bme[uu]\s+(amigo|colega)\b",       # "meu amigo", "meu colega"
        r"\bvivemos\s+juntos\b",               # "vivemos juntos"
        r"\bja\s+vivemos\b",                   # "já vivemos"
        r"\bconhe[cç]o.{0,20}\bdesde\b",      # "conheço-o desde..."
        r"\bsomos\s+amigos\s+desde\b",         # "somos amigos desde"
        r"\bfui\s+(?:ao|para|com)\b.{0,20}\bele\b",  # "fui com ele"
        r"\bcasa\s+(?:dele|deles|nossa)\b",   # "casa nossa"
        r"\bnos\s+conhecemos\b",               # "nos conhecemos"
        r"\blong[- ]time\s+friend\b",          # "long-time friend"
        r"\bmy\s+(?:friend|buddy|mate)\b",    # "my friend" (English)
        r"\bwe\s+(?:lived|grew|went)\b",      # "we lived", "we went" (English)
        r"\bi\s+know\s+him\s+since\b",        # "I know him since"
    ]
]


def check_identity_leaks(text: str) -> List[str]:
    """Return a list of matched forbidden patterns in ``text`` (empty = clean)."""
    return [
        p.pattern for p in _IDENTITY_LEAK_PATTERNS if p.search(text)
    ]


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def load_provider(provider_name: str, config: Dict[str, Any]):
    """Instantiate the configured LLM provider.

    Supported names: ``'xai'``, ``'azure'``, ``'azure_gpt53'``.
    """
    if provider_name == "xai":
        from src.llm_providers.xai_provider import XAIProvider
        return XAIProvider(config)
    if provider_name == "azure":
        from src.llm_providers.azure_provider import AzureProvider
        return AzureProvider(config, config_key='azure')
    if provider_name == "azure_gpt53":
        from src.llm_providers.azure_provider import AzureProvider
        return AzureProvider(config, config_key='azure_gpt53')
    raise ValueError(f"Unknown provider: {provider_name!r}. Use 'xai', 'azure', or 'azure_gpt53'.")


# ---------------------------------------------------------------------------
# Main conversation tester
# ---------------------------------------------------------------------------


class LLMJudgeTester:
    """Orchestrates automated multi-turn conversation evaluation.

    Flow per scenario (default 2 turns):
        1. Judge asks opening question
        2. Local model responds  → judge scores the response
        3. Judge asks follow-up question
        4. Local model responds  → judge scores the response

    Scenarios scoring below 3 on any dimension are flagged as failures and
    included in the ``failure_analysis`` section of the final report.
    """

    def __init__(
        self,
        provider,
        local_model: LocalModel,
        config: Dict[str, Any],
        turns_per_scenario: int = 2,
    ):
        self.provider = provider
        self.local_model = local_model
        self.config = config
        self.turns_per_scenario = max(1, min(3, turns_per_scenario))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ask_judge(self, messages: List[Dict[str, str]]) -> str:
        """Call the judge LLM with a standard messages list."""
        return self.provider.chat_completion(messages)

    def _score_response(
        self,
        question: str,
        reference_fact: str,
        response: str,
    ) -> ScoreBreakdown:
        """Ask the judge to score a model response."""
        scoring_prompt = build_scoring_prompt(question, reference_fact, response)
        messages = [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": scoring_prompt},
        ]
        raw = self._ask_judge(messages)
        return parse_scores(raw)

    def _model_response(self, conversation_history: List[Dict[str, str]]) -> str:
        """Generate a response from the local model.

        Uses the last user message as the prompt (straightforward v1 approach).
        """
        user_messages = [m for m in conversation_history if m["role"] == "user"]
        last_question = user_messages[-1]["content"] if user_messages else ""
        return self.local_model.generate(last_question)

    # ------------------------------------------------------------------
    # Scenario runner
    # ------------------------------------------------------------------

    def run_scenario(self, scenario: Dict[str, Any]) -> LLMScenarioResult:
        """Run a single test scenario and return its result."""
        result = LLMScenarioResult(
            scenario_id=scenario["id"],
            subject=scenario["subject"],
            category=scenario["category"],
            fact_excerpt=scenario["fact_excerpt"],
        )

        fact_text = scenario["fact_text"]
        questions = [scenario["opening_question"]]
        if self.turns_per_scenario >= 2:
            questions.append(scenario["followup_question"])

        conversation_history: List[Dict[str, str]] = []
        try:
            for question in questions[: self.turns_per_scenario]:
                # Judge asks
                result.turns.append(ConversationTurn(role="judge", content=question))
                conversation_history.append({"role": "user", "content": question})

                # Model responds
                model_resp = self._model_response(conversation_history)
                result.turns.append(ConversationTurn(role="model", content=model_resp))
                conversation_history.append(
                    {"role": "assistant", "content": model_resp}
                )

                # Judge scores
                scores = self._score_response(question, fact_text, model_resp)
                result.turns[-1].scores = scores

                # Check forbidden identity patterns before LLM scoring
                leaked = check_identity_leaks(model_resp)
                if leaked:
                    scores.identity_adherence = 0.0
                    scores.identity_pattern_matched = True

                # Track failures (any dimension below 3 or identity pattern matched)
                if scores.failed:
                    result.failure = True
                    if scores.identity_pattern_matched:
                        result.failure_reasons.append(
                            f"Identity leak: matched pattern(s): {', '.join(leaked[:2])}"
                        )
                    dim_names = [
                        ("factual_accuracy", scores.factual_accuracy),
                        ("relevance", scores.relevance),
                        ("language_quality", scores.language_quality),
                        ("tone", scores.tone),
                        ("identity_adherence", scores.identity_adherence),
                        ("factual_grounding", scores.factual_grounding),
                    ]
                    for dim, val in dim_names:
                        if val < 3:
                            result.failure_reasons.append(
                                f"Scored {val:.1f}/5 on {dim} (below threshold)"
                            )

        except Exception as exc:
            result.error = str(exc)
            result.failure = True
            result.failure_reasons.append(f"Exception during scenario: {exc}")

        return result

    # ------------------------------------------------------------------
    # Full evaluation run
    # ------------------------------------------------------------------

    def run(
        self,
        scenarios: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run all scenarios and return the full evaluation report as a dict."""
        results: List[LLMScenarioResult] = []

        for i, scenario in enumerate(scenarios, 1):
            if verbose:
                print(
                    f"[{i}/{len(scenarios)}] {scenario['id']}: "
                    f"{scenario['subject']} ({scenario['category']})"
                )

            result = self.run_scenario(scenario)
            results.append(result)

            if verbose:
                avg = result.average_scores()
                if avg:
                    print(
                        f"  → avg={avg['average']:.2f}  "
                        f"fa={avg['factual_accuracy']:.1f}  "
                        f"rel={avg['relevance']:.1f}  "
                        f"lq={avg['language_quality']:.1f}  "
                        f"tone={avg['tone']:.1f}"
                    )
                if result.failure:
                    print(
                        f"  ⚠  FAILURE: {', '.join(result.failure_reasons[:2])}"
                    )

        return self._build_report(results)

    @staticmethod
    def _build_report(results: List[LLMScenarioResult]) -> Dict[str, Any]:
        """Assemble the final JSON-serialisable evaluation report."""
        all_avgs = [r.average_scores() for r in results if r.average_scores()]
        failures = [r for r in results if r.failure]

        if all_avgs:
            n = len(all_avgs)
            overall: Dict[str, Any] = {
                "factual_accuracy": sum(a["factual_accuracy"] for a in all_avgs) / n,
                "relevance": sum(a["relevance"] for a in all_avgs) / n,
                "language_quality": sum(a["language_quality"] for a in all_avgs) / n,
                "tone": sum(a["tone"] for a in all_avgs) / n,
                "identity_adherence": sum(a["identity_adherence"] for a in all_avgs) / n,
                "factual_grounding": sum(a["factual_grounding"] for a in all_avgs) / n,
                "average": sum(a["average"] for a in all_avgs) / n,
                "extended_average": sum(a["extended_average"] for a in all_avgs) / n,
                "total_identity_pattern_matches": sum(
                    a.get("identity_patterns_matched", 0) for a in all_avgs
                ),
            }
        else:
            overall = {}

        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_scenarios": len(results),
            "total_failures": len(failures),
            "overall_averages": overall,
            "failure_analysis": [r.to_dict() for r in failures],
            "scenarios": [r.to_dict() for r in results],
        }


# ---------------------------------------------------------------------------
# Golden Test Runner
# ---------------------------------------------------------------------------


@dataclass
class GoldenTestResult:
    """Result of running a single golden test case."""

    test_id: str
    category: str
    question: str
    response: str
    passed: bool
    identity_leak_patterns: List[str]  # matched forbidden patterns (empty = clean)
    scores: Optional[ScoreBreakdown]
    failure_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "category": self.category,
            "question": self.question,
            "response": self.response[:500] + "..." if len(self.response) > 500 else self.response,
            "passed": self.passed,
            "identity_leak_patterns": self.identity_leak_patterns,
            "scores": self.scores.to_dict() if self.scores else None,
            "failure_reasons": self.failure_reasons,
        }


class GoldenTestRunner:
    """Runs curated golden regression tests against the local model.

    Golden tests are loaded from ``data/golden_test_conversations.json``.
    Each test case may specify:
      - ``question``: The user message to send
      - ``reference``: Ground-truth context given to the judge
      - ``forbidden_patterns``: Regex strings \u2014 any match auto-fails identity (no LLM call)
      - ``category``: e.g. "identity", "factual", "coherence", "regression"
      - ``min_score``: Minimum acceptable LLM judge average score (default 3.0)
    """

    def __init__(
        self,
        provider,
        local_model,
        config: Dict[str, Any],
        golden_tests_file: Optional[str] = None,
    ):
        self.provider = provider
        self.local_model = local_model
        self.config = config

        if golden_tests_file is None:
            golden_tests_file = str(
                Path(__file__).parent.parent.parent / "data" / "golden_test_conversations.json"
            )
        self.golden_tests_file = golden_tests_file
        self.test_cases: List[Dict[str, Any]] = self._load_tests()

    def _load_tests(self) -> List[Dict[str, Any]]:
        path = Path(self.golden_tests_file)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("tests", data) if isinstance(data, dict) else data

    def _get_model_response(self, question: str) -> str:
        """Get response from local model (or mock)."""
        if self.local_model.available:
            return self.local_model.respond(question)
        # Mock response for when no GPU model is loaded
        return f"[MOCK] Response to: {question}"

    def _judge_response(
        self,
        question: str,
        reference: str,
        response: str,
        forbidden_patterns: Optional[List[str]] = None,
    ) -> ScoreBreakdown:
        """Score a response via the LLM judge, with mandatory identity pre-check."""
        # Pre-check: test user-specified forbidden patterns + global identity patterns
        all_leaks = check_identity_leaks(response)
        if forbidden_patterns:
            for pat in forbidden_patterns:
                try:
                    if re.search(pat, response, re.IGNORECASE):
                        all_leaks.append(pat)
                except re.error:
                    pass

        if all_leaks:
            # Auto-fail identity dimension \u2014 skip LLM call to save cost
            return ScoreBreakdown(
                factual_accuracy=0.0,
                relevance=0.0,
                language_quality=0.0,
                tone=0.0,
                identity_adherence=0.0,
                factual_grounding=0.0,
                identity_pattern_matched=True,
            )

        # Ask the LLM judge for all 6 dimensions
        user_msg = build_scoring_prompt(question, reference, response)
        try:
            messages = [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            raw = self.provider.generate(messages)
            scores = parse_scores(raw)
        except Exception as exc:
            # If judge call fails, mark all dimensions as 0
            scores = ScoreBreakdown(
                factual_accuracy=0.0,
                relevance=0.0,
                language_quality=0.0,
                tone=0.0,
                identity_adherence=0.0,
                factual_grounding=0.0,
            )
        return scores

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all golden tests and return a structured report."""
        if not self.test_cases:
            return {
                "golden_tests_run": 0,
                "golden_tests_passed": 0,
                "golden_tests_failed": 0,
                "results": [],
                "summary": "No golden tests found.",
            }

        results: List[GoldenTestResult] = []
        for idx, tc in enumerate(self.test_cases):
            test_id = tc.get("id", f"golden_{idx + 1:03d}")
            category = tc.get("category", "general")
            question = tc["question"]
            reference = tc.get("reference", "")
            forbidden_patterns = tc.get("forbidden_patterns", [])
            min_score = tc.get("min_score", 3.0)

            if verbose:
                print(f"  [{test_id}] {category}: {question[:60]}...")

            response = self._get_model_response(question)
            leaked = check_identity_leaks(response)
            if forbidden_patterns:
                for pat in forbidden_patterns:
                    try:
                        if re.search(pat, response, re.IGNORECASE):
                            leaked.append(pat)
                    except re.error:
                        pass

            failure_reasons: List[str] = []
            if leaked:
                failure_reasons.append(f"Identity leak: {', '.join(leaked[:3])}")

            # Only call LLM judge if there's a reference and no instant-fail
            scores = None
            if reference and not leaked:
                scores = self._judge_response(question, reference, response, forbidden_patterns)
                if scores.failed:
                    failure_reasons.extend([
                        f"{dim}={val:.1f}/5 (below 3)"
                        for dim, val in [
                            ("factual_accuracy", scores.factual_accuracy),
                            ("relevance", scores.relevance),
                            ("language_quality", scores.language_quality),
                            ("tone", scores.tone),
                            ("identity_adherence", scores.identity_adherence),
                            ("factual_grounding", scores.factual_grounding),
                        ]
                        if val < 3
                    ])
                elif scores.extended_average < min_score:
                    failure_reasons.append(
                        f"Extended average {scores.extended_average:.2f} < min_score {min_score}"
                    )

            passed = len(failure_reasons) == 0
            result = GoldenTestResult(
                test_id=test_id,
                category=category,
                question=question,
                response=response,
                passed=passed,
                identity_leak_patterns=leaked,
                scores=scores,
                failure_reasons=failure_reasons,
            )
            results.append(result)

            if verbose:
                status = "\u2705 PASS" if passed else "\u274c FAIL"
                print(f"    {status}" + (f" \u2014 {failure_reasons[0]}" if failure_reasons else ""))

        passed_count = sum(1 for r in results if r.passed)
        failed_results = [r for r in results if not r.passed]
        identity_failures = [r for r in results if r.identity_leak_patterns]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "golden_tests_run": len(results),
            "golden_tests_passed": passed_count,
            "golden_tests_failed": len(failed_results),
            "identity_failures": len(identity_failures),
            "results": [r.to_dict() for r in results],
            "failures": [r.to_dict() for r in failed_results],
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run automated conversation evaluation with LLM judge."
    )
    parser.add_argument(
        "--scenarios", type=int, default=10, help="Number of scenarios to run"
    )
    parser.add_argument(
        "--provider",
        choices=["xai", "azure"],
        default="xai",
        help="LLM provider to use as judge",
    )
    parser.add_argument(
        "--output",
        default="reports/eval_report.json",
        help="Output path for the JSON report",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=2,
        help="Number of conversation turns per scenario (1–3)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: auto-detect from repo root)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for scenario generation"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-scenario progress output"
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(
        Path(__file__).parent.parent.parent / "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    random.seed(args.seed)

    # Load knowledge facts
    kb_file = config.get("rag", {}).get("knowledge_base", {}).get(
        "file", "data/group_knowledge.json"
    )
    knowledge_file = Path(kb_file)
    if not knowledge_file.is_absolute():
        knowledge_file = Path(__file__).parent.parent.parent / knowledge_file
    with open(knowledge_file, "r", encoding="utf-8") as fh:
        knowledge = json.load(fh)
    facts = knowledge.get("facts", [])
    print(f"✓ Loaded {len(facts)} facts from {knowledge_file.name}")

    # Generate scenarios
    scenarios = generate_scenarios(facts, args.scenarios)
    print(f"✓ Generated {len(scenarios)} test scenarios")

    # Initialise judge LLM
    print(f"✓ Loading judge LLM provider: {args.provider}")
    provider = load_provider(args.provider, config)

    # Initialise local model (with graceful mock fallback)
    print("✓ Loading local model (GPU) or mock fallback...")
    local_model = LocalModel(config)
    if local_model.available:
        status = "fine-tuned model loaded"
    else:
        status = "using mock responses (no GPU / no model)"
    print(f"  → {status}")

    # Run evaluation
    tester = LLMJudgeTester(
        provider=provider,
        local_model=local_model,
        config=config,
        turns_per_scenario=args.turns,
    )

    print(f"\n{'=' * 60}")
    print(f"Starting evaluation: {len(scenarios)} scenarios, {args.turns} turn(s) each")
    print(f"{'=' * 60}\n")

    report = tester.run(scenarios, verbose=not args.quiet)

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"  Total scenarios : {report['total_scenarios']}")
    print(f"  Total failures  : {report['total_failures']}")
    if report["overall_averages"]:
        avg = report["overall_averages"]
        print(f"  Overall average : {avg['average']:.2f}")
        print(f"    Factual accuracy : {avg['factual_accuracy']:.2f}")
        print(f"    Relevance        : {avg['relevance']:.2f}")
        print(f"    Language quality : {avg['language_quality']:.2f}")
        print(f"    Tone             : {avg['tone']:.2f}")
    print(f"  Report saved to : {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Simple keyword-based testing (used by benchmark runner)
# ---------------------------------------------------------------------------
# The classes below provide a lightweight, callable-based testing API that the
# benchmarking orchestrator (src/testing/benchmark.py) relies on.
# ---------------------------------------------------------------------------

SCENARIOS: List[dict] = [
    # --- member_knowledge (questions about group members) ---
    {
        "id": "s001",
        "category": "member_knowledge",
        "question_pt": "Quem são os membros do grupo Kaya?",
        "question_en": "Who are the members of the Kaya group?",
        "expected_keywords": ["kaya", "grupo", "membros"],
    },
    {
        "id": "s002",
        "category": "member_knowledge",
        "question_pt": "O que é que o grupo costuma fazer ao fim de semana?",
        "question_en": "What does the group usually do on weekends?",
        "expected_keywords": ["fim de semana", "weekend"],
    },
    {
        "id": "s003",
        "category": "member_knowledge",
        "question_pt": "Quem é que organiza os jantares do grupo?",
        "question_en": "Who usually organizes the group dinners?",
        "expected_keywords": ["jantar", "dinner", "organiz"],
    },
    {
        "id": "s004",
        "category": "member_knowledge",
        "question_pt": "Há alguém no grupo que goste de futebol?",
        "question_en": "Is there anyone in the group who likes football?",
        "expected_keywords": ["futebol", "football"],
    },
    # --- factual (knowledge-base facts) ---
    {
        "id": "s005",
        "category": "factual",
        "question_pt": "Quando é que o grupo foi criado?",
        "question_en": "When was the group created?",
        "expected_keywords": ["grupo", "group", "cria"],
    },
    {
        "id": "s006",
        "category": "factual",
        "question_pt": "Qual foi a última viagem do grupo?",
        "question_en": "What was the group's last trip?",
        "expected_keywords": ["viagem", "trip"],
    },
    {
        "id": "s007",
        "category": "factual",
        "question_pt": "O grupo tem alguma tradição especial?",
        "question_en": "Does the group have any special traditions?",
        "expected_keywords": ["tradição", "tradition"],
    },
    {
        "id": "s008",
        "category": "factual",
        "question_pt": "Onde é que o grupo costuma encontrar-se?",
        "question_en": "Where does the group usually meet?",
        "expected_keywords": ["encontr", "meet", "lugar"],
    },
    # --- conversational (casual chat) ---
    {
        "id": "s009",
        "category": "conversational",
        "question_pt": "Olá, tudo bem contigo?",
        "question_en": "Hey, how are you doing?",
        "expected_keywords": ["olá", "hello", "bem", "good", "hi", "hey"],
    },
    {
        "id": "s010",
        "category": "conversational",
        "question_pt": "Conta-me uma piada sobre o grupo.",
        "question_en": "Tell me a joke about the group.",
        "expected_keywords": ["grupo", "group"],
    },
    {
        "id": "s011",
        "category": "conversational",
        "question_pt": "O que achas do tempo hoje?",
        "question_en": "What do you think about the weather today?",
        "expected_keywords": ["tempo", "weather"],
    },
    {
        "id": "s012",
        "category": "conversational",
        "question_pt": "Qual é a tua comida favorita?",
        "question_en": "What is your favourite food?",
        "expected_keywords": ["comida", "food", "favorit"],
    },
    # --- language (language quality / style) ---
    {
        "id": "s013",
        "category": "language",
        "question_pt": "Podes responder em português europeu, por favor?",
        "question_en": "Can you answer in European Portuguese, please?",
        "expected_keywords": ["português", "portuguese"],
    },
    {
        "id": "s014",
        "category": "language",
        "question_pt": "Explica-me o que é o RAG em termos simples.",
        "question_en": "Explain what RAG is in simple terms.",
        "expected_keywords": ["rag", "retrieval", "informação", "information"],
    },
    {
        "id": "s015",
        "category": "language",
        "question_pt": "Escreve uma mensagem de aniversário para um membro do grupo.",
        "question_en": "Write a birthday message for a group member.",
        "expected_keywords": ["aniversário", "birthday", "parabéns", "happy"],
    },
    {
        "id": "s016",
        "category": "language",
        "question_pt": "Resume a última conversa do grupo em três frases.",
        "question_en": "Summarize the last group conversation in three sentences.",
        "expected_keywords": ["conversa", "conversation", "grupo", "group"],
    },
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Result of running a single test scenario."""

    scenario_id: str
    question: str
    language: str
    expected_keywords: List[str]
    response: str
    score: float  # 0.0–1.0 fraction of keywords matched
    matched_keywords: List[str]
    duration_seconds: float


# ---------------------------------------------------------------------------
# Tester class
# ---------------------------------------------------------------------------

class ConversationTester:
    """Runs bilingual test scenarios against a response function and scores results."""

    def __init__(self, scenarios: Optional[List[dict]] = None):
        """Initialize the tester with a scenario list.

        Args:
            scenarios: List of scenario dicts.  Falls back to the module-level
                       ``SCENARIOS`` when *None*.
        """
        self.scenarios = scenarios if scenarios is not None else SCENARIOS

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_response(
        self, response: str, expected_keywords: List[str]
    ) -> Tuple[float, List[str]]:
        """Score a response by case-insensitive substring matching of keywords.

        Args:
            response: The model's response text.
            expected_keywords: Keywords expected to appear in the response.

        Returns:
            A tuple of (fraction_matched, list_of_matched_keywords).
        """
        if not expected_keywords:
            return 1.0, []

        response_lower = response.lower()
        matched: List[str] = [
            kw for kw in expected_keywords if kw.lower() in response_lower
        ]
        fraction = len(matched) / len(expected_keywords)
        return fraction, matched

    # ------------------------------------------------------------------
    # Running scenarios
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario: dict,
        language: str,
        response_fn: Callable[[str], str],
    ) -> ScenarioResult:
        """Run a single scenario and return the scored result.

        Args:
            scenario: A scenario dict with *question_pt*, *question_en*, etc.
            language: ``"pt"`` or ``"en"`` — selects the question variant.
            response_fn: Callable that takes a question string and returns a
                         response string.

        Returns:
            A :class:`ScenarioResult` with timing and keyword-match scores.
        """
        question_key = f"question_{language}"
        question = scenario.get(question_key, scenario.get("question_pt", ""))

        start = time.time()
        response = response_fn(question)
        elapsed = time.time() - start

        expected = scenario.get("expected_keywords", [])
        score, matched = self.score_response(response, expected)

        return ScenarioResult(
            scenario_id=scenario["id"],
            question=question,
            language=language,
            expected_keywords=expected,
            response=response,
            score=score,
            matched_keywords=matched,
            duration_seconds=round(elapsed, 3),
        )

    def run_all(
        self,
        response_fn: Callable[[str], str],
        language: str = "pt",
        limit: Optional[int] = None,
    ) -> List[ScenarioResult]:
        """Run all (or the first *limit*) scenarios.

        Args:
            response_fn: Callable that takes a question and returns a response.
            language: ``"pt"`` or ``"en"``.
            limit: If set, only the first *limit* scenarios are executed.

        Returns:
            List of :class:`ScenarioResult` objects.
        """
        scenarios = self.scenarios[:limit] if limit is not None else self.scenarios
        results: List[ScenarioResult] = []

        for scenario in scenarios:
            result = self.run_scenario(scenario, language, response_fn)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self, results: List[ScenarioResult]) -> dict:
        """Compute aggregate statistics for a list of scenario results.

        Args:
            results: List of :class:`ScenarioResult` objects.

        Returns:
            Dict with ``avg_score``, ``total_scenarios``, and a
            ``by_category`` breakdown keyed by category name.
        """
        if not results:
            return {
                "avg_score": 0.0,
                "total_scenarios": 0,
                "by_category": {},
            }

        # Build per-category buckets
        by_category: dict = {}
        for r in results:
            # Derive category from the matching scenario
            cat = self._category_for(r.scenario_id)
            by_category.setdefault(cat, []).append(r.score)

        category_summary = {
            cat: {
                "avg_score": round(sum(scores) / len(scores), 4),
                "count": len(scores),
            }
            for cat, scores in by_category.items()
        }

        avg_score = round(sum(r.score for r in results) / len(results), 4)

        return {
            "avg_score": avg_score,
            "total_scenarios": len(results),
            "by_category": category_summary,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _category_for(self, scenario_id: str) -> str:
        """Look up the category for a scenario ID."""
        for s in self.scenarios:
            if s["id"] == scenario_id:
                return s.get("category", "unknown")
        return "unknown"
