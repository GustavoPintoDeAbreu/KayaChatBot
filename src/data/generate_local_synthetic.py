"""On-prem synthetic training-data generation.

Generates fresh, behavior-targeted training examples WITHOUT sending any group
data off-prem: a local "teacher" model produces synthesized answers to a
behavior-targeted question bank, grounded in the local RAG context. The output
is written in the synthetic_kaya.jsonl conversation format so the existing
merge_datasets → train pipeline consumes it unchanged.

Pipeline:
  question_bank.build_questions  →  retriever.retrieve_all (context)
  →  TeacherModel.generate (local)  →  synthetic_filters.clean_and_accept
  →  data/synthetic_local.jsonl

The teacher is configurable (synthetic_generation.teacher_model_id), default
Qwen3.5-27B in 4-bit which fits a 24 GB GPU for inference. The heavy run is
triggered manually; generate_dataset() takes the teacher and retriever as
parameters so the orchestration is unit-testable with stubs (no model load).

Usage:
  # quick smoke (load teacher, generate a few, print — no write):
  python src/data/generate_local_synthetic.py --smoke 5
  # full generation:
  python src/data/generate_local_synthetic.py [--limit N] [--out FILE]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import random

from src.config_loader import load_config
from src.data.question_bank import build_questions, load_members
from src.data.synthetic_filters import clean_and_accept, strip_thinking
from src.chat.response_utils import build_member_prompt_suffix

# System prompt for mining conversational questions from real chat chunks. This
# gives the dataset breadth beyond the per-member/superlative templates: natural
# questions grounded in what the group actually talked about.
_MINE_SYSTEM_PROMPT = (
    "És um gerador de perguntas. Lês um excerto de conversa de um grupo de amigos "
    "e escreves UMA pergunta curta e natural (em português europeu ou inglês) que "
    "um membro poderia fazer a um assistente sobre o conteúdo, o evento ou as "
    "pessoas mencionadas. A pergunta deve fazer sentido sozinha, sem o excerto. "
    "Responde APENAS com a pergunta."
)


def parse_mined_question(raw: str) -> str:
    """Extract a single clean question from the teacher's mining output."""
    raw = strip_thinking(raw)
    for line in raw.splitlines():
        line = line.strip().strip("\"'“”-•* ")
        if "?" in line:
            return line[: line.index("?") + 1].strip()
    return ""


def mine_questions(
    chunk_texts: List[str], teacher: Any, target: int, seed: int = 3407,
    max_chunk_chars: int = 1200,
) -> List[str]:
    """Use the teacher to turn real conversation chunks into natural questions.

    Stops at ``target`` accepted questions. teacher needs ``.generate``; chunk
    sampling is seeded for reproducibility. Deduplicated (case-insensitive).
    """
    if target <= 0 or not chunk_texts:
        return []
    pool = list(chunk_texts)
    random.Random(seed).shuffle(pool)
    out: List[str] = []
    seen = set()
    for chunk in pool:
        if len(out) >= target:
            break
        user = f"Excerto:\n{chunk[:max_chunk_chars]}\n\nEscreve uma pergunta."
        try:
            raw = teacher.generate(_MINE_SYSTEM_PROMPT, user)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Mining failed on a chunk: {exc}")
            continue
        q = parse_mined_question(raw)
        key = q.lower()
        if q and key not in seen and len(q.split()) >= 3:
            seen.add(key)
            out.append(q)
    return out

# Appended to the persona so the teacher synthesizes instead of quoting and
# never emits emojis — the two things the group complained about.
GEN_INSTRUCTION = (
    "\n\nINSTRUÇÕES DE GERAÇÃO: Responde à pergunta com uma resposta sintetizada, "
    "com raciocínio próprio e personalidade. NUNCA cites literalmente as mensagens "
    "('o X disse:', 'o X mencionou:') — combina a informação e dá a tua leitura. "
    "Quando te pedem um palpite, opinião ou avaliação (ex: quem é o mais X do "
    "grupo), dá-o sempre com base no que sabes. Escreve em português europeu, ou "
    "em inglês se a pergunta for em inglês. Não uses emojis. 2 a 5 frases."
)


class TeacherModel:
    """Loads a local instruct model in 4-bit and generates answers.

    Kept thin and lazy: the heavy import/load happens in __init__, only when a
    real run (or --smoke) constructs it. generate_dataset() never constructs it
    itself, so tests inject a stub instead.
    """

    def __init__(self, model_id: str, sampling: Optional[Dict[str, Any]] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        sampling = sampling or {}
        self.max_new_tokens = int(sampling.get("max_new_tokens", 400))
        self.temperature = float(sampling.get("temperature", 0.7))
        self.top_p = float(sampling.get("top_p", 0.8))
        self.top_k = int(sampling.get("top_k", 20))

        print(f"🤖 Loading teacher model: {model_id} (4-bit)…", flush=True)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="cuda", trust_remote_code=True
        )
        self.model.eval()
        self._torch = torch
        print("✓ Teacher model loaded", flush=True)

    def generate(self, system_prompt: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        # We want direct synthesized answers, not chain-of-thought. Disable
        # thinking when the template supports it (Qwen3/Qwen3.5); fall back
        # gracefully for templates that don't accept the kwarg. strip_thinking()
        # downstream is the safety net either way.
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(text=[prompt], return_tensors="pt").to("cuda")
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                use_cache=True,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def build_user_turn(
    question: str, retriever: Any, knowledge_approach: str
) -> Tuple[str, str]:
    """Return (user_turn, context). user_turn = RAG context + question, matching
    the inference-time format. retriever may be None (no context)."""
    context = ""
    if retriever is not None:
        try:
            context = retriever.retrieve_all(question, knowledge_approach=knowledge_approach)
        except Exception as exc:  # noqa: BLE001 — generation is best-effort per item
            print(f"⚠️  Retrieval failed for {question!r}: {exc}")
            context = ""
    user_turn = f"{context}\n\n{question}" if context else question
    return user_turn, context


def generate_dataset(
    questions: List[str],
    retriever: Any,
    teacher: Any,
    base_system_prompt: str,
    member_suffix: str = "",
    knowledge_approach: str = "json_only",
    min_words: int = 6,
    limit: Optional[int] = None,
    on_example: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, Any]:
    """Generate (and optionally stream out) synthetic examples.

    teacher needs a ``.generate(system_prompt, user_message) -> str`` method and
    retriever a ``.retrieve_all(query, knowledge_approach=...) -> str`` method —
    both injectable so this is testable with stubs. Returns stats; each accepted
    example is passed to ``on_example`` (e.g. to append to a file).
    """
    gen_system_prompt = base_system_prompt + member_suffix + GEN_INSTRUCTION
    if limit:
        questions = questions[:limit]

    stats = {"asked": 0, "accepted": 0, "rejected": 0, "examples": []}
    for question in questions:
        stats["asked"] += 1
        user_turn, context = build_user_turn(question, retriever, knowledge_approach)
        try:
            raw = teacher.generate(gen_system_prompt, user_turn)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Generation failed for {question!r}: {exc}")
            stats["rejected"] += 1
            continue

        answer = clean_and_accept(raw, context, min_words=min_words)
        if not answer:
            stats["rejected"] += 1
            continue

        example = {
            "conversations": [
                {"role": "user", "content": user_turn},
                {"role": "assistant", "content": answer},
            ],
            "source": "synthetic_local",
        }
        stats["accepted"] += 1
        if on_example is not None:
            on_example(example)
        else:
            stats["examples"].append(example)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="On-prem synthetic data generation.")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Generate N examples, print them, and do not write a file.")
    parser.add_argument("--limit", type=int, default=None, help="Cap total questions.")
    parser.add_argument("--per-category", type=int, default=None,
                        help="Cap questions kept per category.")
    parser.add_argument("--out", type=str, default=None, help="Output JSONL path override.")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Override synthetic_generation.teacher_model_id.")
    parser.add_argument("--mine", type=int, default=None,
                        help="Mine N conversational questions from chat chunks (overrides config).")
    parser.add_argument("--no-mine", action="store_true", help="Disable question mining.")
    args = parser.parse_args()

    config = load_config(str(_REPO_ROOT / "config.yaml"))
    sg = config.get("synthetic_generation", {})
    data_cfg = config.get("data", {})
    teacher_id = args.teacher_model or sg.get("teacher_model_id")
    out_path = Path(args.out or sg.get("output_file", "./data/synthetic_local.jsonl"))
    knowledge_approach = sg.get("knowledge_approach", config.get("rag", {}).get("knowledge_approach", "json_only"))
    min_words = int(sg.get("min_words", 6))

    # Persona = live system prompt + member facts (so targets reflect inference).
    base_system_prompt = data_cfg.get("system_prompt", "")
    members = load_members(data_cfg.get("group_members_file", "./data/group_members.json"))
    member_suffix = build_member_prompt_suffix(members)

    template_questions = build_questions(members, per_category=args.per_category)
    print(f"🧩 Template question bank: {len(template_questions)} questions")

    # Retriever (on-prem RAG context).
    from src.chat.retriever import get_retriever
    retriever = get_retriever(config)

    teacher = TeacherModel(teacher_id, sampling=sg)

    # Mine extra conversational questions from real chat chunks for breadth.
    mine_count = 0 if args.no_mine else (args.mine if args.mine is not None else int(sg.get("mine_count", 0)))
    mined_questions: List[str] = []
    if mine_count and not args.smoke:
        try:
            chunk_texts = retriever.collection.get(include=["documents"]).get("documents", [])
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Could not load conversation chunks for mining: {exc}")
            chunk_texts = []
        print(f"⛏️  Mining up to {mine_count} questions from {len(chunk_texts)} chunks…")
        mined_questions = mine_questions(chunk_texts, teacher, mine_count)
        print(f"⛏️  Mined {len(mined_questions)} conversational questions")

    # Combine + dedup (templates first, then mined), keep order, final shuffle.
    seen = set()
    questions = []
    for q in template_questions + mined_questions:
        if q.lower() not in seen:
            seen.add(q.lower())
            questions.append(q)
    random.Random(3407).shuffle(questions)
    print(f"🧩 Total question bank: {len(questions)} questions")

    if args.smoke:
        stats = generate_dataset(
            questions, retriever, teacher, base_system_prompt, member_suffix,
            knowledge_approach=knowledge_approach, min_words=min_words, limit=args.smoke,
        )
        print("\n" + "=" * 70)
        for ex in stats["examples"]:
            q = ex["conversations"][0]["content"].splitlines()[-1]
            a = ex["conversations"][1]["content"]
            print(f"Q: {q}\nA: {a}\n" + "-" * 70)
        print(f"smoke: asked={stats['asked']} accepted={stats['accepted']} rejected={stats['rejected']}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = {"n": 0}
    with open(out_path, "w", encoding="utf-8") as fh:
        def _write(example: Dict) -> None:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            written["n"] += 1
            if written["n"] % 50 == 0:
                print(f"  … {written['n']} written", flush=True)

        stats = generate_dataset(
            questions, retriever, teacher, base_system_prompt, member_suffix,
            knowledge_approach=knowledge_approach, min_words=min_words,
            limit=args.limit, on_example=_write,
        )
    print(f"\n✅ Wrote {written['n']} examples to {out_path}")
    print(f"   asked={stats['asked']} accepted={stats['accepted']} rejected={stats['rejected']}")
    print(f"\nNext: point merge at this file (data.synthetic_source_file) and run merge_datasets.py")


if __name__ == "__main__":
    main()
