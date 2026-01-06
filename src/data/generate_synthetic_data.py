"""
Generate synthetic multi-turn conversations from message chunks using Azure OpenAI.
Creates varied conversation depths with diverse question types.
Supports batch, single, and count generation modes.
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

OUTPUT FORMAT (EXACT JSON):
{{
  "conversations": [
    {{
      "turns": [
        {{"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-04-15] ---\\n[Insert relevant snippet from history above]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nQuem é o Peter?"}},
        {{"role": "assistant", "content": "O Peter é fixe ahahah, sempre a organizar cenas."}}
      ]
    }},
    {{
      "turns": [
        {{"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [2024-04-16] ---\\n[Insert relevant snippet about Gil]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\nO que achas do Gil?"}},
        {{"role": "assistant", "content": "O Gilao é maluco mas boa onda lmao."}},
        {{"role": "user", "content": "Porquê?"}},
        {{"role": "assistant", "content": "Sempre com ideias loucas e a curtir."}}
      ]
    }}
  ]
}}

INSTRUCTIONS:
1. Generate {num_conversations} conversations with 2-5 turns each (varied lengths)
2. **FIRST turn must include RAG context format** with relevant conversation snippet from history
3. Question types to vary:
   - Personality: "Como é o Peter?" "O que achas do Gil?"
   - Group dynamics: "Qual é a relação entre o Peter e o Gustavo?"
   - Opinions: "És de esquerda ou de direita?"
   - Events: "O que aconteceu quando...?"
4. Kaya's responses:
   - Use informal European Portuguese with slang
   - SHORT WhatsApp-style messages (1-3 sentences)
   - **Answer based on the provided context snippet**
   - Reference actual people/events from context
   - Use casual expressions naturally
5. Follow-up questions don't need RAG format (normal conversation)

CRITICAL: First user message MUST include the RAG context format shown above!

Generate ONLY valid JSON."""

    return prompt


def get_single_conversation_prompt(finetune_chunk_text: str, depth: int) -> str:
    """Create prompt for generating a single RAG-aware conversation."""
    prompt = f"""Based on the following Portuguese group chat history, generate ONE realistic {depth}-turn conversation where someone asks questions.

CHAT HISTORY:
{finetune_chunk_text[:30000]}  

CRITICAL INSTRUCTION - RAG FORMAT:
The model will be given conversation history as context during inference. The FIRST user message MUST include RAG context format.

OUTPUT FORMAT - Return valid JSON:
{{
  "conversation": [
    {{"role": "user", "content": "=== Conversas relevantes do grupo ===\\n\\n--- Conversa 1 [date] ---\\n[relevant snippet from history above]\\n\\n=== Fim das conversas ===\\n\\nCom base nestas conversas passadas, responde:\\n[question here]"}},
    {{"role": "assistant", "content": "Kaya's short response based on context"}},
    {{"role": "user", "content": "follow-up question (no RAG format needed)"}},
    {{"role": "assistant", "content": "another response"}}
  ]
}}

INSTRUCTIONS:
- Generate exactly {depth} exchanges (user asks, assistant responds, repeat)
- FIRST user message MUST include RAG context format with relevant snippet
- Use informal European Portuguese with slang
- Kaya answers based on provided context
- Include "wtf", "lmao" and other expressions naturally
- Follow-ups don't need RAG format

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
    parser.add_argument('--mode', choices=['batch', 'single', 'count'], default='batch',
                       help='Generation mode: batch (default), single conversation, or count N conversations')
    parser.add_argument('--count', type=int, help='Number of conversations to generate (for count mode)')
    parser.add_argument('--depth', type=int, default=3, help='Conversation depth for single mode (default: 3)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SYNTHETIC CONVERSATION GENERATION")
    print("=" * 60)
    
    # Load config
    config = load_config()
    paths = get_output_paths()
    
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


if __name__ == "__main__":
    main()
