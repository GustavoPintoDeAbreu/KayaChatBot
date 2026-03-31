"""
Conversation testing and scoring module for KayaChatBot.

Defines bilingual test scenarios (PT/EN) and a scoring framework that
measures how well the bot's responses match expected keywords.  Used by
the benchmark runner to evaluate different RAG / model configurations.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Bilingual test scenarios
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
