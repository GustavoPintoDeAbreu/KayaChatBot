"""
Conversation testing utilities.

Provides helpers to generate evaluation scenarios from knowledge facts,
score model responses using a judge LLM, and produce structured reports.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES = [
    "What do you know about {subject}?",
    "Can you tell me something about {subject}?",
    "What has {subject} been up to?",
    "Tell me about {subject}.",
    "Do you have any information on {subject}?",
]


@dataclass
class ConversationScenario:
    """A single evaluation scenario derived from a knowledge fact."""

    question: str
    expected_keywords: List[str] = field(default_factory=list)
    knowledge_fact: str = ""
    subject: str = ""


@dataclass
class ScoredScenario:
    """A scenario together with judge verdict."""

    scenario: ConversationScenario
    model_response: str = ""
    judge_response: str = ""
    score: float = 0.0
    passed: bool = False


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_scenarios_from_knowledge(
    knowledge_facts: List[Dict[str, Any]],
    templates: Optional[List[str]] = None,
) -> List[ConversationScenario]:
    """
    Generate one evaluation scenario per knowledge fact.

    Parameters
    ----------
    knowledge_facts:
        List of fact dicts, each with at least ``"subject"`` and ``"text"``
        keys.
    templates:
        List of question template strings with a ``{subject}`` placeholder.
        Defaults to :data:`QUESTION_TEMPLATES`.

    Returns
    -------
    List of :class:`ConversationScenario` objects.
    """
    if templates is None:
        templates = QUESTION_TEMPLATES

    scenarios: List[ConversationScenario] = []
    for i, fact in enumerate(knowledge_facts):
        subject = fact.get("subject", "")
        text = fact.get("text", "")
        if not subject or not text:
            continue

        template = templates[i % len(templates)]
        question = template.format(subject=subject)

        # Simple keyword extraction: nouns / capitalised words from the fact text
        keywords = list({w.strip(".,;:!?") for w in text.split() if len(w) > 4})[:5]

        scenarios.append(ConversationScenario(
            question=question,
            expected_keywords=keywords,
            knowledge_fact=text,
            subject=subject,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse_judge_score(judge_response: str) -> float:
    """
    Parse a numeric score from a judge LLM response.

    Looks for patterns like ``"Score: 7"``, ``"7/10"``, or a bare integer on
    its own line in the range 0–10.

    Returns
    -------
    Float in [0.0, 10.0], or 0.0 if no valid score found.
    """
    # Pattern: "Score: N" or "score: N" (case-insensitive)
    match = re.search(r"score[:\s]+([0-9]+(?:\.[0-9]+)?)", judge_response, re.IGNORECASE)
    if match:
        return min(10.0, float(match.group(1)))

    # Pattern: "N/10"
    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b", judge_response)
    if match:
        return min(10.0, float(match.group(1)))

    # Bare integer on its own line
    match = re.search(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*$", judge_response, re.MULTILINE)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 10:
            return val

    return 0.0


def score_scenario(
    scenario: ConversationScenario,
    model_response: str,
    judge_fn: Callable[[str, str], str],
) -> ScoredScenario:
    """
    Score a single scenario by calling *judge_fn* with the question and response.

    Parameters
    ----------
    scenario:
        The evaluation scenario.
    model_response:
        The model's answer to ``scenario.question``.
    judge_fn:
        Callable that accepts ``(question, response)`` and returns the judge's
        raw text response.

    Returns
    -------
    :class:`ScoredScenario` with ``score`` and ``passed`` populated.
    """
    judge_raw = judge_fn(scenario.question, model_response)
    score = parse_judge_score(judge_raw)

    return ScoredScenario(
        scenario=scenario,
        model_response=model_response,
        judge_response=judge_raw,
        score=score,
        passed=score >= 5.0,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_report(scored_scenarios: List[ScoredScenario]) -> Dict[str, Any]:
    """
    Build a JSON-serialisable report dict from a list of scored scenarios.

    Required fields in the output:
    - ``"total_scenarios"``
    - ``"passed"``
    - ``"failed"``
    - ``"average_score"``
    - ``"pass_rate"``
    - ``"scenarios"`` — list of per-scenario dicts

    Parameters
    ----------
    scored_scenarios:
        List of :class:`ScoredScenario` objects.

    Returns
    -------
    Report dict.
    """
    total = len(scored_scenarios)
    if total == 0:
        return {
            "total_scenarios": 0,
            "passed": 0,
            "failed": 0,
            "average_score": 0.0,
            "pass_rate": 0.0,
            "scenarios": [],
        }

    passed = sum(1 for s in scored_scenarios if s.passed)
    average_score = sum(s.score for s in scored_scenarios) / total

    scenario_details = []
    for s in scored_scenarios:
        scenario_details.append({
            "question": s.scenario.question,
            "subject": s.scenario.subject,
            "model_response": s.model_response,
            "score": s.score,
            "passed": s.passed,
        })

    return {
        "total_scenarios": total,
        "passed": passed,
        "failed": total - passed,
        "average_score": round(average_score, 2),
        "pass_rate": round(passed / total, 4),
        "scenarios": scenario_details,
    }
