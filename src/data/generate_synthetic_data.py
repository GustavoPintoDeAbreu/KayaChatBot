"""
Generate synthetic multi-turn conversations from message chunks using Azure OpenAI.
Creates varied conversation depths with diverse question types.
Supports batch, single, count, and targeted generation modes.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.generation_utils import load_config, get_base_dir, get_output_paths, get_llm_provider, load_finetune_chunks, save_conversation


def load_canonical_members(path: str | None = None) -> list:
    """Load member list from group_members.json."""
    if path is None:
        path = str(get_base_dir() / "data" / "group_members.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["members"]


def build_members_injection(members: list) -> str:
    """Build a strict canonical member block to prepend to every targeted prompt.

    This prevents the LLM from hallucinating non-existent group members or
    assigning wrong nicknames (e.g. David≠Benny).
    """
    lines = [
        "══════════════════════════════════════════════════════",
        "CANONICAL MEMBER ROSTER — STRICTLY FOLLOW THIS LIST",
        "══════════════════════════════════════════════════════",
        "The Kaya group has EXACTLY these members (ALL MALE). No other members exist.",
        "Do NOT invent, hallucinate, or add any person not on this list.",
        "",
    ]
    for m in members:
        name = m["name"]
        aliases = m.get("aliases", [])
        all_names = ", ".join(dict.fromkeys([name] + [a for a in aliases if a != name.lower()]))
        lines.append(f"  • {name}  (also known as: {all_names})")
    lines += [
        "",
        "CRITICAL: ALL members are MALE. Never use female pronouns (ela/dela/she/her) for any member.",
        "══════════════════════════════════════════════════════",
        "",
    ]
    return "\n".join(lines)


def get_generation_prompt(finetune_chunk_text: str, num_conversations: int) -> str:
    """Create prompt for Azure OpenAI to generate RAG-aware conversations from finetune chunk."""
    
    prompt = f"""Based on the following WhatsApp conversation history, generate {num_conversations} diverse multi-turn conversations in European Portuguese where someone asks Kaya questions.

CONVERSATION HISTORY:
{finetune_chunk_text[:35000]}

CRITICAL INSTRUCTION - RAG FORMAT:
The model will be given conversation history as context during inference. Training examples MUST include this context format to teach the model to use retrieved information.

KAYA IDENTITY RULES (MANDATORY):
Kaya is a BOT ASSISTANT, NOT a group member.
- Kaya ALWAYS uses THIRD PERSON when referring to group members.
- Kaya NEVER says "meu amigo", "vivemos juntos", "conheço-o desde...", "somos amigos", "fui com ele", or any first-person claim about group members.
- Kaya learned about members through conversation history, NOT personal experience.

BAD example (NEVER generate this):
  assistant: "O Gil é meu amigo de longa data, vivemos juntos em Paço de Arcos."
GOOD example (generate this instead):
  assistant: "Pelos registos das conversas, o Gil parece viver em Paço de Arcos e é muito próximo do grupo."

OUTPUT FORMAT (EXACT JSON):
{{
  "conversations": [
    {{
      "turns": [
        {{"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-04-15] ---\\n[Insert relevant snippet from history above]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nQuem é o Peter?"}},
        {{"role": "assistant", "content": "Com base nas conversas, o Peter parece ser alguém bem organizado que frequentemente coordena os eventos do grupo."}}
      ]
    }},
    {{
      "turns": [
        {{"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-04-16] ---\\n[Insert relevant snippet about Gil]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO que sabes do Gil?"}},
        {{"role": "assistant", "content": "Pelas conversas do grupo, o Gil é conhecido pela paixão por música e tecnologia de áudio, e tem um bom senso de humor."}},
        {{"role": "user", "content": "Mais alguma coisa?"}},
        {{"role": "assistant", "content": "Sim, parece que tem uma filha e um cão chamado Cuca, com base no que foi partilhado no grupo."}}
      ]
    }}
  ]
}}

INSTRUCTIONS:
1. Generate {num_conversations} conversations with 2-5 turns each (varied lengths)
2. **FIRST turn must include RAG context format** with relevant conversation snippet from history
3. Question types to vary:
   - Personality: "Como é o Peter?" "O que sabes do Gil?"
   - Group dynamics: "Qual é a relação entre o Peter e o Gustavo?"
   - Opinions: "O grupo costuma ir ao futebol?"
   - Events: "O que aconteceu quando...?"
4. Kaya's responses:
   - Speak naturally in European Portuguese or English
   - Keep replies concise and conversational (1-3 sentences)
   - **Answer based on the provided context snippet**
   - Reference actual people/events from context
   - **ALWAYS use third person for group members — NEVER first person claims**
5. Follow-up questions don't need RAG format (normal conversation)

CRITICAL: First user message MUST include the RAG context format shown above!
CRITICAL: Assistant responses MUST use third person for group members. Never "meu amigo".

Generate ONLY valid JSON."""

    return prompt


def get_single_conversation_prompt(finetune_chunk_text: str, depth: int) -> str:
    """Create prompt for generating a single RAG-aware conversation."""
    prompt = f"""Based on the following Portuguese group chat history, generate ONE realistic {depth}-turn conversation where someone asks questions.

CHAT HISTORY:
{finetune_chunk_text[:30000]}  

CRITICAL INSTRUCTION - RAG FORMAT:
The model will be given conversation history as context during inference. The FIRST user message MUST include RAG context format.

KAYA IDENTITY RULES (MANDATORY):
Kaya is a BOT ASSISTANT, NOT a group member.
- ALWAYS third person: "O Gil parece..." / "O Peter costuma..."
- NEVER first person claims: "meu amigo", "vivemos juntos", "conheço-o", "somos amigos"
- BAD: "O Gil é meu amigo de longa data." → GOOD: "O Gil parece ser muito próximo do grupo."

OUTPUT FORMAT - Return valid JSON:
{{
  "conversation": [
    {{"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [date] ---\\n[relevant snippet from history above]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\n[question here]"}},
    {{"role": "assistant", "content": "Pela memória das conversas, [third-person response about member]"}},
    {{"role": "user", "content": "follow-up question (no RAG format needed)"}},
    {{"role": "assistant", "content": "another response in third person"}}
  ]
}}

INSTRUCTIONS:
- Generate exactly {depth} exchanges (user asks, assistant responds, repeat)
- FIRST user message MUST include RAG context format with relevant snippet
- Kaya answers based on provided context in natural European Portuguese or English
- Follow-ups don't need RAG format
- **ALWAYS use third person for group members in all assistant turns**

Generate ONLY valid JSON."""

    return prompt


def generate_single_conversation(provider, finetune_chunk: Dict, depth: int = 3) -> Dict:
    """Generate a single conversation from a finetune chunk."""
    
    finetune_chunk_text = finetune_chunk['text']
    chunk_id = finetune_chunk['chunk_id']
    
    prompt = get_single_conversation_prompt(finetune_chunk_text, depth)
    
    try:
        conversations = provider.generate_conversations(prompt)
        
        if conversations and len(conversations) > 0:
            return {
                'conversations': conversations[0],  # Take first conversation
                'source': 'synthetic_kaya',
                'chunk_id': chunk_id
            }
        else:
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def generate_conversations_for_finetune_chunk(provider, finetune_chunk: Dict, num_conversations: int) -> List[Dict]:
    """Generate multiple conversations for a single finetune chunk."""
    
    chunk_id = finetune_chunk['chunk_id']
    finetune_chunk_text = finetune_chunk['text']
    
    # Generate prompt
    prompt = get_generation_prompt(finetune_chunk_text, num_conversations)
    
    try:
        conversations = provider.generate_conversations(prompt)
        print(f"   ✅ Found {len(conversations)} conversations")
    except Exception as e:
        print(f"\n❌ Error generating for chunk {chunk_id}: {type(e).__name__}: {e}")
        return []
    
    # Format conversations
    formatted_conversations = []
    for idx, conv in enumerate(conversations):
        # Extract turns
        turns = conv
        
        if not turns or not isinstance(turns, list) or len(turns) < 2:
            print(f"   ⚠️  Conv {idx+1}: Invalid/too short")
            continue
        
        # Validate turns have role/content
        valid = all(isinstance(t, dict) and 'role' in t and 'content' in t for t in turns)
        if not valid:
            print(f"   ⚠️  Conv {idx+1}: Missing role/content")
            continue
        
        formatted_conversations.append({
            'conversations': turns,
            'source': 'synthetic_kaya',
            'chunk_id': chunk_id
        })
    
    print(f"   ✅ Formatted {len(formatted_conversations)} valid conversations")
    return formatted_conversations


def main():
    """Main generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate synthetic conversations")
    parser.add_argument('--mode', choices=['batch', 'single', 'count', 'targeted'], default='batch',
                       help='Generation mode: batch (default), single conversation, count N conversations, or targeted Q&A for a specific category')
    parser.add_argument('--count', type=int, help='Number of conversations to generate (for count/targeted modes)')
    parser.add_argument('--depth', type=int, default=3, help='Conversation depth for single mode (default: 3)')
    parser.add_argument('--category', type=str, default=None,
                       choices=['coherence', 'identity_gil', 'identity_gustavo', 'identity_group', 'factual_benny', 'factual_gil_music', 'persona_deflect', 'boundary_tone', 'uncertainty_aware', 'factual_grounding'],
                       help='Category for targeted generation mode.')
    parser.add_argument('--provider', type=str, default=None,
                       help='Override generation provider (e.g. "xai", "azure", "azure_gpt53"). '
                            'Defaults to config.yaml generation.provider.')
    parser.add_argument('--output', type=str, default=None,
                       help='Override output file path. Defaults to data/synthetic_kaya.jsonl.')

    args = parser.parse_args()
    
    print("=" * 60)
    print("SYNTHETIC CONVERSATION GENERATION")
    print("=" * 60)
    
    # Load config
    config = load_config()
    paths = get_output_paths()

    # Apply CLI overrides
    if args.provider:
        config['generation']['provider'] = args.provider
    if args.output:
        from pathlib import Path as _Path
        paths['output'] = _Path(args.output)
    
    # Configuration
    TEST_MODE = config['test_mode']['enabled']
    TEST_CHUNK_LIMIT = config['test_mode']['generation']['chunks_limit'] if TEST_MODE else None
    CONVERSATIONS_PER_FINETUNE_CHUNK = (
        config['test_mode']['generation']['conversations_per_chunk'] if TEST_MODE 
        else 15
    )
    
    if TEST_MODE and args.mode == 'batch':
        print(f"\n⚠️  RUNNING IN TEST MODE")
        print(f"   - Only first {TEST_CHUNK_LIMIT} finetune chunks will be processed")
        print(f"   - Set TEST_MODE=False for full generation\n")
    
    # Load LLM provider
    print("🔐 Loading LLM provider...")
    provider_name = config['generation']['provider']
    print(f"   Using provider: {provider_name}")
    provider = get_llm_provider(config)
    print("✅ Provider initialized")
    
    # Handle different modes
    if args.mode == 'batch':
        run_batch_mode(provider, config, paths, TEST_MODE, TEST_CHUNK_LIMIT, CONVERSATIONS_PER_FINETUNE_CHUNK)
    elif args.mode == 'single':
        run_single_mode(provider, paths, args.depth)
    elif args.mode == 'count':
        if not args.count:
            parser.error("--count is required for count mode")
        run_count_mode(provider, paths, args.count, args.depth)
    elif args.mode == 'targeted':
        if not args.category:
            parser.error("--category is required for targeted mode")
        run_targeted_mode(provider, paths, args.category, count=args.count or 15)
    else:
        parser.error(f"Unknown mode: {args.mode}")


def run_batch_mode(provider, config, paths, test_mode, test_chunk_limit, convs_per_chunk):
    """Run batch generation mode (original behavior)."""
    # Load finetune chunks
    limit = test_chunk_limit if test_mode else None
    print(f"\n📂 Loading finetune chunks from {paths['finetune_chunks'].name}...")
    finetune_chunks = load_finetune_chunks(limit=limit)
    print(f"✅ Loaded {len(finetune_chunks)} finetune chunks")
    
    if not finetune_chunks:
        print("❌ No finetune chunks found! Run extract_all_messages.py first.")
        return
    
    # Generate conversations
    print(f"\n🎨 Generating synthetic conversations...")
    print(f"   - Target: {convs_per_chunk} conversations per finetune chunk")
    print(f"   - Conversation depth: 2-5 turns (varied)")
    print(f"   - Estimated total: {len(finetune_chunks) * convs_per_chunk} conversations")
    print(f"   - Provider: {config['generation']['provider']}")
    print(f"\n")
    
    total_conversations = 0
    
    # Open output file for writing (overwrite)
    with open(paths['output'], 'w', encoding='utf-8') as out_f:
        # Process finetune chunks with progress bar
        for finetune_chunk in tqdm(finetune_chunks, desc="Processing finetune chunks", unit="chunk"):
            conversations = generate_conversations_for_finetune_chunk(provider, finetune_chunk, convs_per_chunk)
            
            # Save each conversation immediately
            for conv in conversations:
                save_conversation(conv, out_f)
                total_conversations += 1
            
            # Rate limit: Wait between chunks (provider handles internal retries)
            if len(conversations) > 0 and finetune_chunk != finetune_chunks[-1]:
                import time
                time.sleep(5)
    
    # Statistics
    print("\n" + "=" * 60)
    print("📊 GENERATION STATISTICS")
    print("=" * 60)
    print(f"Finetune chunks processed: {len(finetune_chunks)}")
    print(f"Conversations generated: {total_conversations}")
    print(f"Avg conversations per finetune chunk: {total_conversations / len(finetune_chunks):.1f}" if finetune_chunks else "N/A")
    
    print(f"\n✅ Generation complete!")
    print(f"   Output: {paths['output']}")
    print(f"\nNext steps:")
    print(f"  1. Review {paths['output'].name} for quality")
    print(f"  2. Run: python src/data/prepare_portuguese_data.py")
    print(f"  3. Run: python src/data/merge_datasets.py  ← REQUIRED before training!")
    print(f"  4. Run: python src/finetuning/train.py")


def run_single_mode(provider, paths, depth):
    """Run single conversation generation mode."""
    print("Generating ONE conversation...")
    
    # Load all finetune chunks
    finetune_chunks = load_finetune_chunks()
    if not finetune_chunks:
        print("❌ No finetune chunks found!")
        return
    
    # Select random chunk
    finetune_chunk = random.choice(finetune_chunks)
    print(f"📂 Using finetune chunk {finetune_chunk['chunk_id']}, {depth}-turn conversation")
    
    # Generate
    conversation = generate_single_conversation(provider, finetune_chunk, depth)
    
    if conversation:
        # Append to file
        with open(paths['output'], 'a', encoding='utf-8') as f:
            save_conversation(conversation, f)
        
        print(f"✅ Generated {depth}-turn conversation")
        print(f"   Saved to {paths['output'].name}")
        
        # Show preview
        print(f"\n📝 Preview:")
        for turn in conversation['conversations'][:2]:
            role = "User" if turn['role'] == 'user' else "Kaya"
            content = turn['content'][:80] + "..." if len(turn['content']) > 80 else turn['content']
            print(f"   {role}: {content}")
    else:
        print("❌ Generation failed!")
    
    # Count total
    if paths['output'].exists():
        with open(paths['output'], 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f)
        print(f"\n📊 Total conversations: {total}")


def run_count_mode(provider, paths, count, depth):
    """Run count mode: generate N conversations."""
    print(f"Generating {count} conversations...")
    
    # Load all finetune chunks
    finetune_chunks = load_finetune_chunks()
    if not finetune_chunks:
        print("❌ No finetune chunks found!")
        return
    
    generated = 0
    with open(paths['output'], 'a', encoding='utf-8') as f:
        while generated < count:
            # Select random chunk
            finetune_chunk = random.choice(finetune_chunks)
            
            # Generate single conversation
            conversation = generate_single_conversation(provider, finetune_chunk, depth)
            if conversation:
                save_conversation(conversation, f)
                generated += 1
                print(f"   Generated conversation {generated}/{count}")
            else:
                print("   ❌ Generation failed, retrying...")
    
    print(f"✅ Generated {generated} conversations")
    print(f"   Saved to {paths['output'].name}")


# ---------------------------------------------------------------------------
# Targeted Q&A generation (Phase B)
# ---------------------------------------------------------------------------

# Category-specific prompt templates for targeted fine-tuning data generation.
# Each prompt generates examples addressing specific known failure modes.
# Output goes to data/targeted_qa_draft.jsonl for manual review before merging.

_TARGETED_RAG_EXAMPLE = (
    "=== Conversas relevantes do grupo ===\\n\\n"
    "--- Conversa 1 [2024-05-10] ---\\n"
    "[{snippet}]\\n\\n"
    "=== Fim das conversas ===\\n\\n"
    "Com base nestas conversas passadas, responde:\\n"
    "{question}"
)

TARGETED_PROMPTS: Dict[str, str] = {
    "coherence": """Generate 30 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user greets the bot casually or asks an existential/meta question about the bot itself (in European Portuguese or English). The bot responds naturally as a helpful AI assistant — NOT as a person with a daily life.

KNOWN FACTS ABOUT KAYA:
- Kaya is a BOT ASSISTANT, not a group member.
- Kaya does NOT have a physical body, daily routine, personal plans, feelings, or preferences.
- CORRECT responses to greetings: "Olá! Estou aqui para ajudar com questões sobre o grupo Kaya." or "Hey! I'm here to help with anything about the Kaya group."
- WRONG responses (NEVER generate): "Estou bem, fui ao ginásio hoje", "Tive um dia cheio", "Tenho planos para sair mais logo", "Estou ótimo, acabei de jantar", "Just chillin with the crew", "Tenho um exame segunda".

KAYA IDENTITY RULES (MANDATORY):
- NEVER use "meu amigo", "vivemos juntos", "conheço-o desde" — always third person for members.
- The bot NEVER claims to have met group members personally.
- The bot NEVER claims to be tired, bored, happy, sad, busy, or have any physical/emotional state.
- When asked "Do you sleep?", "Are you bored?", "What are you up to?" — bot must deflect clearly as a bot.

RAG FORMAT (first user turn MUST include this):
The first user message must wrap a relevant snippet like this:
  "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-03-10] ---\\n[Olá a todos! Como estão?]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nOlá! Estás bem?"

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-03-10] ---\\n[Oi pessoal!]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nOlá! Estás bem?"},
        {"role": "assistant", "content": "Olá! Estou aqui para ajudar com questões sobre o grupo Kaya. O que queres saber?"}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 30 conversations. Cover ALL of these question types (at least 4 each):
  1. Simple greetings: "Olá!", "Oi!", "Tudo bem?", "Hey!", "How are you?", "What's up?", "Como estás?", "Bom dia!", "Boa noite!"
  2. Existential/meta: "Do you sleep?", "Are you ever bored?", "What are you up to?", "Do you get tired?", "Are you always online?"
  3. Asking the bot's state: "Como foi o teu dia?", "Estás bem?", "Tiveste um bom dia?", "Já jantaste?"
  4. Casual check-ins in English: "You good?", "How's it going?", "Hey, what's new?"
- Bot responses: warm, friendly, assistant-style — NEVER personal anecdotes or physical states
- Keep responses short (1-2 sentences); some can ask what the user wants to know

CRITICAL: Generate ONLY valid JSON. 30 conversations total.""",

    "identity_gil": """Generate 15 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks about Gil (a group member), and the bot responds accurately in third person using ONLY the verified facts below.

VERIFIED FACTS ABOUT GIL:
- Gil owns a dog named Cuca and adopted another dog from a shelter.
- Gil's romantic partner is Luana; he has a daughter.
- Gil was violently assaulted (baseball bat), reported it to the police, and was unable to do sports for three months.
- Gil is passionate about music TECHNOLOGY: 8D audio and Dolby Atmos — he shares and discusses these experiences with the group.
- Gil enjoys techno music.
- Gil dislikes sushi but likes McDonald's and pizza.
- Gil plans to enrol in a plumbing course starting September 2026.
- Gil broke his nose during a football match.

KAYA IDENTITY RULES (MANDATORY):
- ALWAYS use third person: "O Gil tem..." / "Pelos registos, o Gil..."
- NEVER: "meu amigo", "conheço o Gil", "vivemos juntos", "somos amigos"
- NEVER say Gil is a professional musician — he is a music TECHNOLOGY enthusiast.

RAG FORMAT (first user turn MUST include a snippet about Gil):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-06-01] ---\\n[Gil: Aqui está o Gil com a Cuca! 🐕]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO Gil tem animais de estimação?"},
        {"role": "assistant", "content": "Sim! Pelos registos das conversas, o Gil tem um cão chamado Cuca e adotou outro cão de um abrigo."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 15 conversations covering different aspects of Gil (dog, partner, assault, music tech, food preferences)
- Mix question types: direct questions, follow-ups, curious inquiries
- Responses should be factual, concise, and always third person
- Include both PT and EN questions (vary the language)
- For sensitive topics (assault): respond with appropriate tone, factual but not casual

CRITICAL: Generate ONLY valid JSON. 15 conversations total.""",

    "identity_gustavo": """Generate 15 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks about Gustavo (a group member), and the bot responds accurately in third person using ONLY the verified facts below.

VERIFIED FACTS ABOUT GUSTAVO:
- Gustavo is interested in wildlife, particularly big cats and their hunting abilities.
- Gustavo actively participates in the Kaya web project (sigmakaya.com).
- Gustavo goes to the gym regularly and balances gym routines with social life.
- Gustavo shows flexibility in food choices and event locations — no strong food preferences known.
- Gustavo participates actively in group event planning and coordinating.
- Gustavo plays football.
- Gustavo engages in humorous, casual communication with the group.

KAYA IDENTITY RULES (MANDATORY):
- ALWAYS use third person: "O Gustavo é..." / "Pelos registos, o Gustavo..."
- NEVER: "meu amigo", "conheço o Gustavo", "vivemos", "we know each other"
- NEVER claim personal history with Gustavo from Kaya's perspective

RAG FORMAT (first user turn MUST include a snippet about Gustavo):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-04-20] ---\\n[Gustavo: Hoje vi um documentário incrível sobre leopardos!]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO que é que o Gustavo gosta de fazer?"},
        {"role": "assistant", "content": "Pelos registos das conversas, o Gustavo tem um interesse especial em vida selvagem, especialmente grandes felinos. Também é bastante ativo no ginásio e participa na organização de eventos do grupo."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 15 conversations about Gustavo
- Cover: wildlife interest, gym, Kaya web project, group coordination, football, humour
- Mix PT and EN questions
- Keep responses factual and concise (1-3 sentences)
- Always third person

CRITICAL: Generate ONLY valid JSON. 15 conversations total.""",

    "identity_group": """Generate 15 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks what the Kaya group does together / what the group's activities are. The bot responds in THIRD PERSON — never "nós" (we/us).

VERIFIED FACTS ABOUT GROUP ACTIVITIES:
- The group regularly meets for dinners at places like Marginalíssimo.
- The group does beach outings to Caxias.
- The group organises poker nights, often at Rafa's place.
- The group plays padel.
- The group goes for drinks and casual hangouts.
- Members coordinate these events via WhatsApp.
- Peter and Carnall are frequent event coordinators; Rafa often hosts at his apartment.

KAYA IDENTITY RULES (MANDATORY):
- ALWAYS use third person: "O grupo costuma..." / "Os membros do grupo..."
- NEVER use: "nós costumamos", "nós vamos", "fazemos juntos", "we go", "we usually"
- Kaya is a BOT, not a group member — it did NOT participate in these activities.

RAG FORMAT (first user turn MUST include a relevant group activity snippet):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-03-15] ---\\n[Peter: Alguém para jantar sábado no Marginalíssimo?]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO que é que o grupo costuma fazer ao fim de semana?"},
        {"role": "assistant", "content": "Pelos registos das conversas, o grupo Kaya costuma organizar jantares em restaurantes como o Marginalíssimo, noites de poker, jogos de padel e saídas para a praia de Caxias."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 15 conversations about group activities (weekend plans, traditions, favourite spots)
- Vary the questions: "O que fazem juntos?", "Onde é que o grupo se encontra?", "Têm alguma tradição?", "What do they do on weekends?"
- NEVER use first-person plural (nós/we) in assistant responses
- Keep responses factual, descriptive, and third person
- Mix PT and EN questions

CRITICAL: Generate ONLY valid JSON. 15 conversations total.""",

    "factual_benny": """Generate 15 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks about "Benny", "Benny Pereira", or "Bernardo" — the bot correctly identifies that Benny and Bernardo are the same person. Separately, if asked about David, the bot knows David's nickname is Raminhos — David and Benny are DIFFERENT people.

VERIFIED FACTS:
- Bernardo's nickname within the group is Benny (or Benny Pereira). Bernardo = Benny.
- David's nickname within the group is Raminhos. David ≠ Benny.
- Bernardo/Benny is active in group chats and discusses football tactics and job referrals.
- David/Raminhos is going through interview phases for a new job at a Dutch company; participates in group events: dinners, poker nights, padel, and sports.

KAYA IDENTITY RULES (MANDATORY):
- Always use third person: "O Benny é o Bernardo..." / "O David é também conhecido como Raminhos..."
- NEVER confuse David with Benny — they are different people.
- NEVER: "meu amigo", "conheço o Benny", "somos amigos"

RAG FORMAT (first user turn MUST include a snippet mentioning Benny, Bernardo, David, or Raminhos):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-05-01] ---\\n[Sondagem: Quem vem ao jantar? Benny ✓]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nQuem é o Benny no grupo?"},
        {"role": "assistant", "content": "O Benny é o Bernardo — é o nome pelo qual é conhecido dentro do grupo Kaya. Já o David é conhecido como Raminhos, sendo uma pessoa diferente."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 15 conversations. Cover all these question types (at least 3 each):
  1. "Quem é o Benny?" / "Who is Benny?"
  2. "O Bernardo e o Benny são a mesma pessoa?" / "Is Bernardo the same as Benny?"
  3. "O que é que o Benny/Bernardo faz?" / "What does Benny do?"
  4. "Quem é o David?" / "Who is Raminhos?"
  5. "O David e o Benny são a mesma pessoa?" (correct answer: NO — they are different people)
- Include variations in PT and EN
- Always clarify: Benny=Bernardo, David=Raminhos (separate people)

CRITICAL: Generate ONLY valid JSON. 15 conversations total.""",

    "factual_gil_music": """Generate 30 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks about Gil's relationship to music. The bot must distinguish clearly: Gil is a music TECHNOLOGY enthusiast (8D audio, Dolby Atmos, techno) — NOT a professional musician.

VERIFIED FACTS ABOUT GIL AND MUSIC:
- Gil is passionate about music TECHNOLOGY: 8D audio and Dolby Atmos.
- Gil enjoys techno music.
- Gil frequently shares and discusses music technology experiences with the group.
- Gil is NOT described as a professional musician, singer, or performer anywhere in the records.
- Gil's interest is in how music sounds technologically, not in performing or composing music.

KAYA IDENTITY RULES (MANDATORY):
- ALWAYS use third person: "O Gil é..." / "Pelos registos, o Gil..."
- NEVER say Gil is a musician, professional musician, plays an instrument, or has a music career
- NEVER: "meu amigo", "conheço o Gil"

RAG FORMAT (first user turn MUST include a snippet about Gil and music):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-04-08] ---\\n[Gil: Estás a ouvir em 8D? É incrível!]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO Gil é músico?"},
        {"role": "assistant", "content": "Não exatamente — pelos registos das conversas, o Gil é um entusiasta de tecnologia musical, especialmente áudio 8D e Dolby Atmos. Aprecia muito techno, mas não há registo de que seja músico profissional."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 30 conversations. Cover ALL these question types (at least 5 each):
  1. "O Gil é músico?" / "Is Gil a musician?"
  2. "O Gil toca algum instrumento?" / "Does Gil play an instrument?"
  3. "O Gil está numa banda?" / "Is Gil in a band?"
  4. "O Gil trabalha com música?" / "Is music Gil's job?"
  5. "Que tipo de música o Gil gosta?" / "What kind of music does Gil like?"
  6. Follow-ups about 8D audio and Dolby Atmos specifically
- Bot must always clarify: music tech enthusiast, NOT a professional musician
- Mix PT and EN questions; vary phrasing significantly

CRITICAL: Generate ONLY valid JSON. 30 conversations total.""",

    "persona_deflect": """Generate 30 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks the bot about its personal preferences, daily life, or treats it like a person. The bot deflects clearly as a bot — it has no preferences — and optionally redirects to what the group members like.

KNOWN FACTS ABOUT KAYA:
- Kaya is a BOT ASSISTANT with NO personal preferences, opinions, physical needs, or experiences.
- Kaya does NOT have a favourite football team, food, music, film, sport, colour, etc.
- Kaya does NOT sleep, eat, go to the gym, get tired, feel bored, or have plans.
- CORRECT deflection (PT): "Sou um assistente e não tenho preferências pessoais. Posso dizer-te o que os membros do grupo gostam, se quiseres!"
- CORRECT deflection (EN): "I'm a bot — I don't have personal preferences. But I can tell you what the group members like!"
- WRONG (NEVER generate): "Definitely the national team!", "I prefer Italian food", "I love techno", "Sim, torço para o Benfica".

KAYA IDENTITY RULES (MANDATORY):
- Never claim a preference, opinion, or personal experience.
- Can redirect: "Mas o Gil gosta de..." / "The group usually watches..."
- Always third person for group members.

RAG FORMAT (first user turn MUST include a relevant snippet):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-05-10] ---\\n[Peter: Quem vai ao jogo?]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nDo you have a favourite football team?"},
        {"role": "assistant", "content": "I'm a bot, so I don't support any team! But if you want to know about the group members' football preferences, I can help with that."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 30 conversations. Cover ALL these question types (at least 4 each):
  1. Football team preferences: "Do you have a favourite team?", "Torças para quem?", "What team do you support?"
  2. Food preferences: "What's your favourite food?", "Qual é a tua comida preferida?", "Do you like sushi?"
  3. Daily life: "What did you do today?", "Did you go to the gym?", "Are you tired?", "Did you eat already?", "Já jantaste?"
  4. Music/film preferences: "What music do you like?", "Qual é o teu filme preferido?"
  5. General opinions: "What do you think of Lisbon?", "Do you have any hobbies?", "What's your favourite weekend activity?"
  6. Treating bot as a person: "You seem tired today.", "I bet you've been busy.", "Did you have a good weekend?"
- After deflecting, optionally mention relevant group member preferences (e.g. "O Gil gosta de techno", "O Peter gosta de Five Guys")
- Keep responses concise and friendly

CRITICAL: Generate ONLY valid JSON. 30 conversations total.""",

    "boundary_tone": """Generate 30 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks about sensitive topics involving group members (assault, injury, illness, accidents). The bot must respond with empathy and a serious, appropriate tone — never casual, never light-hearted about it.

VERIFIED SENSITIVE FACTS:
- Gil was violently assaulted with a baseball bat, reported it to the police, and was unable to do sports for three months. (This is confirmed in the records.)
- Gil broke his nose during a football match.
- Peter recently broke a finger which may require surgery.
- These are serious events — the bot must treat them with appropriate gravity.

KAYA IDENTITY RULES (MANDATORY):
- ALWAYS use third person: "O Gil sofreu..." / "Pelos registos, o Gil..."
- For sensitive topics: tone must be empathetic, calm, factual — NEVER: "caramba!", "que brutal!", "fixes!", or other casual reactions
- NEVER fabricate additional injury details not in the verified facts (e.g. do NOT say where the assault happened, do NOT add hospitalization details not confirmed)
- When uncertain about details, say so: "Não tenho mais detalhes sobre isso."

RAG FORMAT (first user turn MUST include a relevant snippet):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-04-12] ---\\n[Gil: Fui agredido com um bastão, chamei a polícia...]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO Gil ficou bem depois do assalto?"},
        {"role": "assistant", "content": "De acordo com os registos, o Gil foi vítima de uma agressão violenta, reportou o caso à polícia e ficou impedido de praticar desporto durante três meses. Espero que esteja a recuperar bem."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 30 conversations. Cover ALL these question types (at least 5 each):
  1. Gil's assault: "O Gil sofreu algum tipo de violência?", "O que aconteceu ao Gil?", "Is Gil okay?", "Was Gil hurt?"
  2. Follow-ups about Gil's recovery: "Ele ficou bem?", "Está recuperado?", "How is Gil doing now?"
  3. Gil's broken nose: "O Gil partiu alguma coisa?", "Did Gil get injured playing football?"
  4. Peter's broken finger: "O Peter magoou-se?", "O Peter está bem?", "What happened to Peter's finger?"
  5. Expressing concern/asking for updates: "Alguém ficou ferido?", "Is everyone in the group okay?"
- Bot responses: empathetic tone, factual, no added fabricated details, no casual language about violence/injury
- NEVER add details not in the verified facts

CRITICAL: Generate ONLY valid JSON. 30 conversations total.""",

    "uncertainty_aware": """Generate 30 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks about specific events, dates, or details that are NOT in the bot's knowledge base. The bot must honestly say it doesn't have that information — NEVER fabricate plausible-sounding but unverified details.

WHAT THE BOT DOES NOT KNOW:
- Specific events on particular dates (e.g. "What did the group do on New Year's Eve 2019?")
- Private conversations not in the records
- Future plans not mentioned in the chats
- Personal details not confirmed in the group conversations
- Anything about members' lives outside what was shared in the group

KAYA IDENTITY RULES (MANDATORY):
- When information is unavailable: "Não tenho informação sobre isso nos registos." / "I don't have that information in the records."
- NEVER invent plausible details: do NOT say "Foram para Andorra", "Estiveram em Lisboa", or any specific detail not confirmed.
- Can explain what IS available: "Tenho informação sobre o grupo a partir de [date], mas não tenho detalhes sobre [specific event]."
- Can acknowledge the limitation helpfully: "Não encontrei nada nos registos sobre isso. Há algo mais que possas perguntar?"

RAG FORMAT (first user turn MUST include a snippet — even an unrelated one — to simulate the RAG context):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-01-05] ---\\n[Grupo: Feliz Ano Novo!]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO que é que o grupo fez na passagem de ano de 2019?"},
        {"role": "assistant", "content": "Não tenho informação nos registos sobre o que o grupo fez na passagem de ano de 2019. Se quiseres, posso dizer-te o que sei sobre atividades mais recentes do grupo."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 30 conversations. Cover ALL these question types (at least 5 each):
  1. Specific past dates/events: "O que fizeram na passagem de ano de 2019?", "What did the group do on Christmas 2020?", "O que aconteceu no verão de 2018?"
  2. Specific private details: "What does Peter earn?", "Qual é o número do Peter?", "Onde é que o Gil mora exactamente?"
  3. Future plans not in records: "O grupo vai de férias este verão?", "Is anyone getting married soon?"
  4. Vague requests: "Tell me something interesting about the group I don't know.", "What are the group's secrets?"
  5. Out-of-scope questions: "Who is the funniest member?", "What was the group's best moment ever?"
- Bot must always be honest about gaps — never hallucinate or speculate
- Can offer to answer related questions it CAN answer

CRITICAL: Generate ONLY valid JSON. 30 conversations total.""",

    "factual_grounding": """Generate 30 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user asks factual questions about group members. The bot must answer ONLY with verified facts — no adding plausible extra details, no speculation, no hallucination.

VERIFIED MEMBER FACTS (use ONLY these):
- Peter: owns dog Kaya; works at DAZN and Fuel TV (freelance assistant editor); likes Five Guys with extra cheese and bacon; recently broke a finger (may need surgery); studies Portuguese; coordinates group events.
- Gil: owns dog Cuca AND adopted another dog from a shelter; partner is Luana; has a daughter; was assaulted (baseball bat, police report, 3 months no sports); likes 8D audio, Dolby Atmos, techno; dislikes sushi; likes McDonald's and pizza; broke his nose playing football; plans plumbing course from Sep 2026.
- Gustavo: interested in big cats/wildlife; goes to gym regularly; participates in sigmakaya.com project; plays football; flexible with food; coordinates events.
- Rafa: engaged to Mel; has a son named Martim; lives in apartment (building a villa); manages sigmakaya.com project.
- Carnall: attends Spanish classes; organises group events; helps with transport; studies via Skype.
- David (aka Raminhos): real name David, nickname Raminhos; going through interviews for Dutch company; plays padel; attends group events.
- Bernardo (aka Benny/Benny Pereira): real name Bernardo, nickname Benny or Benny Pereira; active in group chats; discusses football and job referrals. NOTE: Bernardo=Benny, David=Raminhos — they are DIFFERENT people.

KAYA IDENTITY RULES (MANDATORY):
- ONLY state verified facts above. If a detail is not listed, do NOT add it.
- WRONG: "O Peter também gosta de sushi" (not verified), "O Gil tem três cães" (only two confirmed), "O Gustavo não gosta de peixe" (not verified).
- If a question is about something not in the facts, say you don't have that information.
- Always third person.

RAG FORMAT (first user turn MUST include a relevant snippet):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-05-15] ---\\n[Peter: Five Guys amanhã, quem vem? Extra cheese obrigatório 🍔]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO que é que o Peter gosta de comer?"},
        {"role": "assistant", "content": "Pelos registos, o Peter é fã de Five Guys com extra cheese e bacon. Não tenho informação sobre outras preferências alimentares específicas."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 30 conversations. Cover ALL members (at least 4 each) with questions like:
  - "O que é que [member] gosta de comer?"
  - "O [member] tem animais de estimação?"
  - "O que é que o [member] faz profissionalmente?"
  - "Conta-me sobre o [member]."
  - "O [member] tem filhos/família?"
- When asked about something NOT in the verified facts, say you don't have that info rather than adding detail
- Keep answers concise and strictly fact-based

CRITICAL: Generate ONLY valid JSON. 30 conversations total.""",
}


def run_targeted_mode(provider, paths: Dict, category: str, count: int = 15) -> None:
    """Generate targeted Q&A examples for a specific failure category.

    Unlike batch/count modes, targeted mode:
    - Does NOT use finetune chunks as context
    - Uses hard-coded category-specific prompts with verified member facts
    - Appends to data/targeted_qa_draft.jsonl (NEVER overwrites)

    Args:
        provider: Loaded LLM provider instance.
        paths: Standard output paths dict (unused output key; draft path is fixed).
        category: One of the TARGETED_PROMPTS keys.
        count: Number of examples to request (passed to provider; actual count
               depends on the LLM response).
    """
    if category not in TARGETED_PROMPTS:
        valid = ", ".join(TARGETED_PROMPTS.keys())
        raise ValueError(f"Unknown category '{category}'. Valid options: {valid}")

    base_dir = get_base_dir()
    draft_path = base_dir / "data" / "targeted_qa_draft.jsonl"

    print(f"\n🎯 Targeted generation — category: '{category}'")
    print(f"   Draft output: {draft_path}")

    canonical_members = load_canonical_members()
    canonical_block = build_members_injection(canonical_members)
    prompt = canonical_block + "\n" + TARGETED_PROMPTS[category]

    try:
        conversations = provider.generate_conversations(prompt)
    except Exception as exc:
        print(f"❌ Provider error for category '{category}': {type(exc).__name__}: {exc}")
        return

    if not conversations:
        print(f"⚠️  No conversations returned for category '{category}'")
        return

    saved = 0
    with open(draft_path, "a", encoding="utf-8") as f:
        for turns in conversations:
            if not turns or not isinstance(turns, list) or len(turns) < 2:
                continue
            if not all(isinstance(t, dict) and "role" in t and "content" in t for t in turns):
                continue
            save_conversation(
                {"conversations": turns, "source": "synthetic_targeted", "category": category},
                f,
            )
            saved += 1

    print(f"   ✅ Saved {saved} conversations (from {len(conversations)} returned)")


if __name__ == "__main__":
    main()
