"""Programmatic needle-recall training examples.

Generates training examples that teach the model to retrieve facts planted at
varying depths inside long conversation context — without any API calls. Filler
is built from real messages in data/all_messages_cleaned.jsonl, rendered in the
same format the inference retriever produces so training matches deployment.

Each example:
  user:      === Conversas relevantes do grupo ===
             --- Conversa 1 --- ... [filler] ... [NEEDLE] ... [more filler]
             === Fim das conversas ===

             <question about the planted fact>
  assistant: <1-2 PT sentences containing the exact answer value>

Usage:
  python src/data/generate_needle_training.py --count 200 --out data/needle_training.jsonl
  python src/data/generate_needle_training.py --count 200 --seed 42 --min-tokens 2000 --max-tokens 4000
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MESSAGES_FILE = _REPO_ROOT / "data" / "all_messages_cleaned.jsonl"
_MEMBERS_FILE = _REPO_ROOT / "data" / "group_members.json"

# Token estimator: matches retrieve_all()'s whitespace-based estimator.
def _count_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / 0.60))


# Fact templates — varied types; NEVER uses the bench needle strings
# ("código secreto", "4827") to avoid contaminating the evaluation gate.
_FACT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "fact": "O {name} marcou jantar no restaurante {place} para o próximo {weekday} às {time}.",
        "question": "Onde é que o {name} marcou jantar e a que horas?",
        "answer": "O {name} marcou jantar no {place} para o próximo {weekday} às {time}.",
        "slots": {
            "place": ["Bairro do Avillez", "Tasca do Chico", "Zé da Mouraria", "Solar dos Presuntos", "Cervejaria Ramiro"],
            "weekday": ["sábado", "domingo", "sexta-feira", "quinta-feira"],
            "time": ["20h", "20h30", "21h", "19h30", "21h30"],
        },
    },
    {
        "fact": "O voo do {name} para {city} sai às {time} do Terminal {terminal}.",
        "question": "A que horas sai o voo do {name} para {city}?",
        "answer": "O voo do {name} para {city} sai às {time} do Terminal {terminal}.",
        "slots": {
            "city": ["Londres", "Berlim", "Amesterdão", "Paris", "Dublin", "Varsóvia"],
            "time": ["06h45", "08h20", "11h55", "14h10", "17h35", "22h00"],
            "terminal": ["1", "2", "T2"],
        },
    },
    {
        "fact": "O {name} quer comprar um presente de {occasion} no valor de {price} euros.",
        "question": "Quanto é que o {name} quer gastar no presente de {occasion}?",
        "answer": "O {name} quer gastar {price} euros no presente de {occasion}.",
        "slots": {
            "occasion": ["aniversário", "casamento", "Natal", "despedida de solteiro"],
            "price": ["35", "50", "75", "100", "120", "150", "200"],
        },
    },
    {
        "fact": "O {name} tem consulta no {clinic} na {weekday} às {time}.",
        "question": "Quando é a consulta do {name} no {clinic}?",
        "answer": "O {name} tem consulta no {clinic} na {weekday} às {time}.",
        "slots": {
            "clinic": ["Hospital da Luz", "CUF Descobertas", "Clínica de São Marcos", "SAMS", "Hospital Particular"],
            "weekday": ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira"],
            "time": ["09h", "10h30", "11h", "14h", "15h30", "16h"],
        },
    },
    {
        "fact": "O {name} reservou o Airbnb em {city} pelo valor de {price} euros a noite.",
        "question": "Qual é o preço por noite do Airbnb que o {name} reservou em {city}?",
        "answer": "O {name} reservou um Airbnb em {city} por {price} euros por noite.",
        "slots": {
            "city": ["Alfama", "Bairro Alto", "Mouraria", "Cascais", "Sintra", "Porto"],
            "price": ["45", "60", "80", "95", "110", "130"],
        },
    },
    {
        "fact": "O carro novo do {name} é um {brand} {model} do ano {year}.",
        "question": "Que carro é que o {name} comprou?",
        "answer": "O {name} comprou um {brand} {model} do ano {year}.",
        "slots": {
            "brand": ["Toyota", "Seat", "Skoda", "Peugeot", "Volkswagen", "Honda"],
            "model": ["Yaris", "Ibiza", "Octavia", "208", "Golf", "Civic"],
            "year": ["2022", "2023", "2024"],
        },
    },
    {
        "fact": "O {name} vai correr a {race} no dia {day} de {month}.",
        "question": "Que corrida é que o {name} vai fazer em {month}?",
        "answer": "O {name} vai correr a {race} no dia {day} de {month}.",
        "slots": {
            "race": ["Meia Maratona de Lisboa", "EDP Rock n' Roll Lisboa", "Corrida da Luz", "Night Run", "Lisbon Trail"],
            "day": ["5", "12", "19", "26", "3", "10", "17"],
            "month": ["setembro", "outubro", "novembro", "março", "abril", "maio"],
        },
    },
    {
        "fact": "O {name} tem o número de sócio {number} do {club}.",
        "question": "Qual é o número de sócio do {name} no {club}?",
        "answer": "O número de sócio do {name} no {club} é o {number}.",
        "slots": {
            "number": ["12473", "88321", "45901", "67234", "23109", "91782", "54326"],
            "club": ["Sporting CP", "Sport Lisboa e Benfica", "FC Porto", "Sporting de Braga"],
        },
    },
    {
        "fact": "A senha do WiFi do apartamento do {name} é {password}.",
        "question": "Qual é a senha do WiFi do apartamento do {name}?",
        "answer": "A senha do WiFi do apartamento do {name} é {password}.",
        "slots": {
            "password": ["Kaya2024!", "Lisboa#99", "Amigos123", "SunsetPT7", "Carcavelos88", "NightOut22"],
        },
    },
    {
        "fact": "O {name} pagou {price} euros pela guitarra {brand} que comprou em segunda mão.",
        "question": "Quanto é que o {name} pagou pela guitarra?",
        "answer": "O {name} pagou {price} euros pela guitarra {brand} em segunda mão.",
        "slots": {
            "price": ["180", "250", "320", "450", "600", "750"],
            "brand": ["Fender Stratocaster", "Gibson SG", "Yamaha Pacifica", "Epiphone Les Paul", "Ibanez RG"],
        },
    },
]

_WEEKDAYS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


def load_message_pool(path: Path) -> List[Dict]:
    pool = []
    if not path.exists():
        return pool
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                pool.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pool


def _load_member_names(path: Path) -> List[str]:
    if not path.exists():
        return ["Gustavo", "Peter", "Gil", "Bernardo", "Mateusz"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    members = raw.get("members", raw) if isinstance(raw, dict) else raw
    return [m.get("name", m.get("first_name", "Member")) for m in members if isinstance(m, dict)]


def build_filler_blocks(
    messages: List[Dict],
    target_token_count: int,
    rng: random.Random,
    messages_per_block: int = 8,
) -> str:
    """Return a context string of ~target_token_count tokens from real messages."""
    if not messages:
        filler_text = "O grupo combinou para se encontrar no fim de semana. "
        filler_line = filler_text
        while _count_tokens(filler_line) < target_token_count:
            filler_line += filler_text
        return _format_filler_as_context([{"sender": "Grupo", "text": filler_line}])

    # Shuffle messages and group them into pseudo-conversations
    pool = list(messages)
    rng.shuffle(pool)
    # Use only messages with non-trivial text
    pool = [m for m in pool if len(m.get("text", "")) >= 10]

    blocks: List[List[Dict]] = []
    i = 0
    while i < len(pool):
        blocks.append(pool[i : i + messages_per_block])
        i += messages_per_block

    # Accumulate blocks until we hit the target
    selected: List[List[Dict]] = []
    current_tokens = 0
    for block in blocks:
        block_text = _format_block(block, len(selected) + 1)
        block_tokens = _count_tokens(block_text)
        selected.append(block)
        current_tokens += block_tokens
        if current_tokens >= target_token_count:
            break

    # If we ran out of messages, repeat
    while current_tokens < target_token_count and blocks:
        for block in blocks:
            block_text = _format_block(block, len(selected) + 1)
            current_tokens += _count_tokens(block_text)
            selected.append(block)
            if current_tokens >= target_token_count:
                break

    return _format_blocks_as_context(selected)


def _format_block(msgs: List[Dict], block_num: int) -> str:
    lines = [f"\n--- Conversa {block_num} ---"]
    for m in msgs:
        sender = m.get("sender", "Membro")
        text = m.get("text", "")
        if text:
            lines.append(f"{sender}: {text}")
    return "\n".join(lines)


def _format_filler_as_context(msgs: List[Dict]) -> str:
    return _format_blocks_as_context([[m] for m in msgs])


def _format_blocks_as_context(blocks: List[List[Dict]]) -> str:
    parts = ["=== Conversas relevantes do grupo ==="]
    for i, block in enumerate(blocks, 1):
        parts.append(f"\n--- Conversa {i} ---")
        for m in block:
            sender = m.get("sender", "Membro")
            text = m.get("text", "")
            if text:
                parts.append(f"{sender}: {text}")
    parts.append("\n=== Fim das conversas ===")
    return "\n".join(parts)


def _sample_slots(slots: Dict[str, Any], rng: random.Random) -> Dict[str, str]:
    """Pre-sample one value per slot key so fact/question/answer use identical values."""
    return {key: rng.choice(options) for key, options in slots.items()}


def _fill_template(tmpl: str, sampled: Dict[str, str], name: str) -> str:
    result = tmpl.replace("{name}", name)
    for key, value in sampled.items():
        result = result.replace("{" + key + "}", value)
    return result


def plant_needle(filler_context: str, needle_sentence: str, depth: float) -> str:
    """Insert the needle sentence at `depth` fraction through the filler context."""
    lines = filler_context.splitlines()
    # Find the content lines (skip header/footer/block headers)
    content_indices = [
        i for i, line in enumerate(lines)
        if line and not line.startswith("===") and not line.startswith("---")
    ]
    if not content_indices:
        return filler_context + f"\n{needle_sentence}"

    insert_after = content_indices[min(int(len(content_indices) * depth), len(content_indices) - 1)]
    lines.insert(insert_after + 1, needle_sentence)
    return "\n".join(lines)


def build_example(
    messages: List[Dict],
    member_names: List[str],
    template: Dict[str, Any],
    depth: float,
    target_tokens: int,
    rng: random.Random,
) -> Optional[Dict]:
    """Build one needle-recall training example."""
    name = rng.choice(member_names)
    slots = template.get("slots", {})
    sampled = _sample_slots(slots, rng)

    fact = _fill_template(template["fact"], sampled, name)
    question = _fill_template(template["question"], sampled, name)
    answer = _fill_template(template["answer"], sampled, name)

    # Build filler sized so that filler + needle ≈ target_tokens
    needle_tokens = _count_tokens(fact)
    filler_target = max(200, target_tokens - needle_tokens - 50)

    filler_context = build_filler_blocks(messages, filler_target, rng)
    full_context = plant_needle(filler_context, fact, depth)

    user_content = f"{full_context}\n\n{question}"
    total_tokens = _count_tokens(user_content)

    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "source": "needle_synthetic",
        "meta": {
            "depth": depth,
            "target_tokens": target_tokens,
            "actual_tokens": total_tokens,
            "member": name,
        },
    }


def generate_needle_examples(
    count: int,
    seed: int = 3407,
    min_tokens: int = 2000,
    max_tokens: int = 4000,
    messages: Optional[List[Dict]] = None,
    member_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Generate ``count`` deterministic needle-recall training examples."""
    rng = random.Random(seed)

    if messages is None:
        messages = load_message_pool(_MESSAGES_FILE)
    if member_names is None:
        member_names = _load_member_names(_MEMBERS_FILE)
    if not member_names:
        member_names = ["Gustavo", "Peter", "Gil"]

    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    examples = []

    for i in range(count):
        template = _FACT_TEMPLATES[i % len(_FACT_TEMPLATES)]
        depth = depths[i % len(depths)]
        target_tokens = rng.randint(min_tokens, max_tokens)

        ex = build_example(messages, member_names, template, depth, target_tokens, rng)
        if ex is not None:
            examples.append(ex)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate needle-recall training examples.")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-tokens", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--out", type=str, default="data/needle_training.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} needle examples (seed={args.seed}, {args.min_tokens}–{args.max_tokens} tokens)…")
    examples = generate_needle_examples(
        count=args.count,
        seed=args.seed,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    tokens = [ex["meta"]["actual_tokens"] for ex in examples]
    depths = [ex["meta"]["depth"] for ex in examples]
    print(f"Wrote {len(examples)} examples to {out_path}")
    print(f"  Token range: {min(tokens)}–{max(tokens)}, mean={sum(tokens)//len(tokens)}")
    print(f"  Depths: {sorted(set(depths))}")


if __name__ == "__main__":
    main()
