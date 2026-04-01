"""
Interactive Chat Script (API-backed)
Allows the user to chat via xAI/Azure provider with always-on RAG support,
without requiring a local GPU or fine-tuned model.
"""
import os
import sys
import json
import yaml
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


def build_system_prompt(config: dict, knowledge_approach: str) -> str:
    """Build the system prompt, optionally injecting group member profiles."""
    base_system_prompt = config['data']['system_prompt']
    system_prompt = base_system_prompt

    members_file = config.get('data', {}).get('group_members_file')
    if members_file and knowledge_approach in ('both', 'json_only'):
        mf = Path(members_file)
        if not mf.is_absolute():
            config_root = Path(__file__).parent.parent.parent
            mf = config_root / members_file
        if mf.exists():
            members_data = json.loads(mf.read_text(encoding='utf-8'))
            member_lines = []
            for m in members_data.get('members', []):
                line = m['name']
                aliases = [a for a in m.get('aliases', []) if a.lower() != m['name'].lower()]
                if aliases:
                    line += f" (também conhecido como: {', '.join(aliases)})"
                notes = m.get('notes', '')
                if notes:
                    sentences = [s.strip() for s in notes.split('.') if s.strip()]
                    short_notes = '. '.join(sentences[:2]) + '.'
                    line += f" — {short_notes}"
                member_lines.append(line)
            if member_lines:
                system_prompt += f"\n\nMembros do grupo Kaya: {'; '.join(member_lines)}."

    return system_prompt


def get_provider(config: dict):
    """Instantiate the configured LLM provider."""
    provider_name = config.get('generation', {}).get('provider', 'xai')
    if provider_name == 'xai':
        from src.llm_providers.xai_provider import XAIProvider
        return XAIProvider(config)
    elif provider_name == 'azure':
        from src.llm_providers.azure_provider import AzureProvider
        return AzureProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def main():
    print("=" * 60)
    print("Kaya Chat (API mode) with always-on RAG")
    print("=" * 60)

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    rag_config = config.get('rag', {})
    rag_enabled = rag_config.get('enabled', False)
    always_on = rag_config.get('always_on', True)
    knowledge_approach = rag_config.get('knowledge_approach', 'both')
    provider_name = config.get('generation', {}).get('provider', 'xai')

    print(f"Provider : {provider_name.upper()}")
    print(f"RAG      : {'enabled (always-on)' if rag_enabled and always_on else 'enabled' if rag_enabled else 'disabled'}")
    print(f"Knowledge: {knowledge_approach}")

    # Build system prompt
    system_prompt = build_system_prompt(config, knowledge_approach)

    # Initialize LLM provider
    print("\nInitialising LLM provider...", end=" ", flush=True)
    provider = get_provider(config)
    print("done.")

    # Initialize RAG retriever
    retriever = None
    if rag_enabled:
        try:
            from src.chat.retriever import get_retriever
            retriever = get_retriever(config)
            print("✓ RAG retriever ready.")
        except Exception as e:
            print(f"⚠️  RAG init failed: {e}")
            print("   Continuing without RAG...")
            rag_enabled = False

    # Chat loop
    try:
        user_name = input("\nEnter your name (default: User): ").strip() or "User"
    except (EOFError, OSError):
        user_name = "User"

    bot_name = "Kaya Bot"
    rag_status = "always-on RAG" if (rag_enabled and always_on) else ("RAG" if rag_enabled else "no RAG")
    print(f"\n💬 Chat started ({rag_status})! Type 'exit' to quit.\n")
    print("-" * 60)

    history = []  # List of {"role": ..., "content": ...} dicts (for the API)
    max_history_turns = 5  # Keep last N assistant/user pairs

    while True:
        try:
            user_input = input(f"\n{user_name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() in ('exit', 'quit'):
            break

        # RAG retrieval
        context = ""
        if rag_enabled and retriever:
            try:
                context = retriever.retrieve_all(user_input, knowledge_approach=knowledge_approach)
                if context:
                    conv_count = context.count("--- Conversa ")
                    kb_count = context.count("---") - conv_count
                    info = f"📚 RAG: {conv_count} conversation chunk(s)"
                    if kb_count > 0:
                        info += f", {kb_count} knowledge fact(s)"
                    print(info)
            except Exception as e:
                print(f"⚠️  RAG retrieval failed: {e}")
                context = ""

        # Build user message: optional context + input
        message_parts = []
        if context:
            message_parts.append(context)
        message_parts.append(f"{user_name}: {user_input}")
        user_message = "\n\n".join(message_parts)

        mode = "always-on RAG" if (rag_enabled and context) else "no context"
        print(f"   [Mode: {mode}]")

        # Build full messages list: system + trimmed history + current user turn
        messages = [{"role": "system", "content": system_prompt}]
        # Trim history to last N turns (each turn = user + assistant)
        trimmed = history[-(max_history_turns * 2):]
        messages.extend(trimmed)
        messages.append({"role": "user", "content": user_message})

        # Call the API
        try:
            print(f"{bot_name}: ", end="", flush=True)
            response = provider.chat_completion(messages)
            print(response)
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            continue

        # Update history with plain input (no context) for follow-up coherence
        history.append({"role": "user", "content": f"{user_name}: {user_input}"})
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
