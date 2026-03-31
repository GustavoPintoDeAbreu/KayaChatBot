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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure repo root is on sys.path so sibling packages can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScoreBreakdown:
    """Per-response scores on the 0–5 rubric."""

    factual_accuracy: float
    relevance: float
    language_quality: float
    tone: float

    @property
    def average(self) -> float:
        return (
            self.factual_accuracy + self.relevance + self.language_quality + self.tone
        ) / 4

    @property
    def failed(self) -> bool:
        """True if any single dimension scores below 3."""
        return any(
            v < 3
            for v in [
                self.factual_accuracy,
                self.relevance,
                self.language_quality,
                self.tone,
            ]
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "factual_accuracy": self.factual_accuracy,
            "relevance": self.relevance,
            "language_quality": self.language_quality,
            "tone": self.tone,
            "average": self.average,
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
class ScenarioResult:
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
            "average": sum(s.average for s in scored) / n,
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

Evaluate the given response on a 0–5 integer scale for each of these four dimensions:

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

Respond ONLY with a valid JSON object (no markdown, no explanation), for example:
{"factual_accuracy": 4, "relevance": 5, "language_quality": 4, "tone": 4}
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
    )


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def load_provider(provider_name: str, config: Dict[str, Any]):
    """Instantiate the configured LLM provider."""
    if provider_name == "xai":
        from src.llm_providers.xai_provider import XAIProvider

        return XAIProvider(config)
    if provider_name == "azure":
        from src.llm_providers.azure_provider import AzureProvider

        return AzureProvider(config)
    raise ValueError(f"Unknown provider: {provider_name!r}. Use 'xai' or 'azure'.")


# ---------------------------------------------------------------------------
# Main conversation tester
# ---------------------------------------------------------------------------


class ConversationTester:
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

    def run_scenario(self, scenario: Dict[str, Any]) -> ScenarioResult:
        """Run a single test scenario and return its result."""
        result = ScenarioResult(
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

                # Track failures (any dimension below 3)
                if scores.failed:
                    result.failure = True
                    dim_names = [
                        ("factual_accuracy", scores.factual_accuracy),
                        ("relevance", scores.relevance),
                        ("language_quality", scores.language_quality),
                        ("tone", scores.tone),
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
        results: List[ScenarioResult] = []

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
    def _build_report(results: List[ScenarioResult]) -> Dict[str, Any]:
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
                "average": sum(a["average"] for a in all_avgs) / n,
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
    tester = ConversationTester(
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
