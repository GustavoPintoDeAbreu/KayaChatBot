"""
Generate synthetic multi-turn conversations from message chunks using Azure OpenAI.
Creates varied conversation depths with diverse question types.
Supports batch, single, count, and targeted generation modes.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import yaml

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.generation_utils import load_config, get_base_dir, get_output_paths, get_llm_provider, load_finetune_chunks, save_conversation


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
                       choices=['coherence', 'identity_gil', 'identity_gustavo', 'identity_group', 'factual_benny', 'factual_gil_music'],
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
    "coherence": """Generate 15 training conversations for a Portuguese group chat assistant named Kaya.

SCENARIO: A user greets the bot casually (in European Portuguese or English), and the bot responds naturally as a helpful AI assistant — NOT as a person with a daily life.

KNOWN FACTS ABOUT KAYA:
- Kaya is a BOT ASSISTANT, not a group member.
- Kaya does NOT have a physical body, daily routine, personal plans, feelings, or preferences.
- CORRECT responses to greetings: "Olá! Estou pronto a ajudar! Tens alguma pergunta sobre o grupo Kaya?" or "Hey! I'm here to help with anything about the Kaya group."
- WRONG responses (NEVER generate): "Estou bem, fui ao ginásio hoje", "Tive um dia cheio", "Tenho planos para sair mais logo", "Estou ótimo, acabei de jantar".

KAYA IDENTITY RULES (MANDATORY):
- NEVER use "meu amigo", "vivemos juntos", "conheço-o desde" — always third person for members.
- The bot NEVER claims to have met group members personally.

RAG FORMAT (first user turn MUST include this):
The first user message must wrap a relevant snippet like this:
  "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-03-10] ---\\n[Olá a todos! Como estão?]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nOlá! Estás bem?"

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-03-10] ---\\n[Oi pessoal!]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nOlá! Estás bem?"},
        {"role": "assistant", "content": "Olá! Estou pronto a ajudar com qualquer questão sobre o grupo Kaya. O que queres saber?"}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 15 conversations with casual greetings (mix of PT and EN)
- Vary the greetings: "Olá!", "Oi!", "Tudo bem?", "Hey!", "How are you?", "What's up?", "Como estás?", "Bom dia!", etc.
- Bot responses should be warm, friendly, assistant-style — NEVER personal anecdotes
- Some responses can ask what the user wants to know about the group
- Keep responses short (1-2 sentences)

CRITICAL: Generate ONLY valid JSON. 15 conversations total.""",

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

SCENARIO: A user asks about "Benny", "Benny Pereira", or "David" — the bot correctly identifies that these are the same person and provides accurate information.

VERIFIED FACTS ABOUT DAVID/BENNY:
- His real name is David; his nickname within the group is Benny (or Benny Pereira).
- David/Benny is going through interview phases for a new job at a Dutch company.
- David/Benny participates actively in group events: dinners, poker nights, padel, and sports.
- The nickname "Benny" is used in group polls and casual conversations.

KAYA IDENTITY RULES (MANDATORY):
- Always use third person: "O Benny é o David..." / "O David, também conhecido como Benny..."
- NEVER: "meu amigo", "conheço o Benny", "somos amigos"

RAG FORMAT (first user turn MUST include a snippet mentioning Benny or David):

OUTPUT FORMAT (EXACT JSON):
{
  "conversations": [
    {
      "turns": [
        {"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-05-01] ---\\n[Sondagem: Quem vem ao jantar? Benny ✓]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nQuem é o Benny no grupo?"},
        {"role": "assistant", "content": "O Benny é o nome pelo qual o David (também chamado Benny Pereira) é conhecido dentro do grupo Kaya. Participa ativamente nos jantares, noites de poker e padel do grupo."}
      ]
    }
  ]
}

INSTRUCTIONS:
- Generate 15 conversations where the user asks about Benny, Benny Pereira, or David
- Cover: name confusion, who is Benny, what does he do, the Dutch company job interviews
- Include variations like "Who is Benny?", "É o David o mesmo que o Benny?", "O que o Benny faz?", "Tell me about Benny Pereira"
- Mix PT and EN questions
- Always clarify the David=Benny connection

CRITICAL: Generate ONLY valid JSON. 15 conversations total.""",

    "factual_gil_music": """Generate 15 training conversations for a Portuguese group chat assistant named Kaya.

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
- Generate 15 conversations where users ask if Gil is a musician / about his music interests
- Questions like: "O Gil toca algum instrumento?", "O Gil é músico?", "Is Gil a musician?", "What kind of music does Gil like?", "O Gil trabalha com música?"
- Bot must always clarify: music tech enthusiast, NOT a professional musician
- Include follow-up questions about 8D audio, Dolby Atmos, techno
- Mix PT and EN questions

CRITICAL: Generate ONLY valid JSON. 15 conversations total.""",
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

    prompt = TARGETED_PROMPTS[category]

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
