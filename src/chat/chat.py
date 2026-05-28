"""
Interactive Chat Script
Allows the user to chat with the fine-tuned Kaya model with RAG support.
"""
import os
import sys
import json
import uuid
import torch
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from peft import PeftModel

# Add parent directory to path for imports (Docker compatibility)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import load_config

try:
    from src.chat.memory import SessionMemory
    _memory_enabled = True
except ImportError:
    try:
        from chat.memory import SessionMemory
        _memory_enabled = True
    except ImportError:
        _memory_enabled = False

def main():
    print("=" * 60)
    print("Kaya Chat Interface with RAG")
    print("=" * 60)

    # Load configuration (Docker or local environment)
    docker_config = '/app/config.yaml'
    local_config = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    config_path = docker_config if os.path.exists(docker_config) else local_config

    config = load_config(config_path)

    model_dir = config['training']['output_dir']
    base_system_prompt = config['data']['system_prompt']
    rag_config = config.get('rag', {})
    rag_enabled = rag_config.get('enabled', False)
    always_on = rag_config.get('always_on', True)
    knowledge_approach = rag_config.get('knowledge_approach', 'both')

    # Build system prompt — optionally inject group member profiles (JSON approach)
    system_prompt = base_system_prompt

    # Prepend uncensored preamble when uncensored_mode is enabled (runtime only, not training)
    chat_cfg = config.get('chat', {})
    if chat_cfg.get('uncensored_mode', False):
        uncensored_preamble = chat_cfg.get('uncensored_system_prompt', '')
        if uncensored_preamble:
            system_prompt = uncensored_preamble + "\n\n" + system_prompt

    members_file = config.get('data', {}).get('group_members_file')
    if members_file:
        # Support both absolute paths and relative paths
        from pathlib import Path as _Path
        _mf = _Path(members_file)
        if not _mf.is_absolute():
            _mf = _Path(config_path).parent / members_file
        if _mf.exists() and knowledge_approach in ('both', 'json_only'):
            import json as _json
            members_data = _json.loads(_mf.read_text(encoding='utf-8'))
            member_lines = []
            for m in members_data.get('members', []):
                line = m['name']
                aliases = [a for a in m.get('aliases', []) if a.lower() != m['name'].lower()]
                if aliases:
                    line += f" (também conhecido como: {', '.join(aliases)})"
                notes = m.get('notes', '')
                if notes:
                    # Keep only the first 2 sentences to stay within token budget
                    sentences = [s.strip() for s in notes.split('.') if s.strip()]
                    short_notes = '. '.join(sentences[:2]) + '.'
                    line += f" — {short_notes}"
                member_lines.append(line)
            if member_lines:
                system_prompt += f"\n\nMembros do grupo Kaya: {'; '.join(member_lines)}."

    # Load model: Unsloth FastModel for Gemma 4 (uses cached base model via adapter_config.json);
    # standard PEFT path for Qwen3 (Unsloth fast-inference was broken for Qwen3).
    print(f"\nLoading model... (this may take a minute)")
    adapter_config_path = Path(model_dir) / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"\n❌ Error: adapter_config.json not found in {model_dir}")
        return
    adapter_cfg = json.loads(adapter_config_path.read_text(encoding='utf-8'))
    base_model_name = adapter_cfg.get('base_model_name_or_path', 'unsloth/Qwen3-14B-bnb-4bit')

    is_gemma4 = 'gemma-4' in base_model_name.lower() or 'gemma4' in base_model_name.lower()

    if is_gemma4:
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=config['model']['max_seq_length'],
            dtype=None,
            load_in_4bit=True,
        )
        FastModel.for_inference(model)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="cuda",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()

    print(f"✓ Model loaded!")

    # Initialize RAG retriever if enabled
    retriever = None
    if rag_enabled:
        try:
            try:
                from src.chat.retriever import get_retriever
            except ImportError:
                from chat.retriever import get_retriever

            retriever = get_retriever(config)
            print("✓ RAG retriever initialized!")
        except Exception as e:
            print(f"⚠️  RAG initialization failed: {e}")
            print("   Continuing without RAG...")
            rag_enabled = False

    # Setup chat — skip interactive prompts in non-TTY environments (Docker, piped input)
    if sys.stdin.isatty():
        try:
            user_name = input("\nEnter your name (default: User): ").strip() or "User"
        except (EOFError, OSError):
            user_name = "User"
    else:
        user_name = "User"
        print("\n[Non-interactive mode] Using default name: User")

    # Load local session history (privacy: stored locally only, never sent to external services)
    history = []
    max_history_lines = 10
    session_memory = None
    if _memory_enabled:
        history_file = config.get('chat', {}).get('history_file', 'data/chat_history.json')
        session_memory = SessionMemory(history_file)
        loaded = session_memory.load()
        if loaded:
            history = loaded
            print(f"✓ Loaded {len(history)} messages from previous session")

    bot_name = "Kaya Bot"

    log_dir = Path(config_path).parent / "data" / "feedback"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "live_interactions.jsonl"

    rag_status = "with always-on RAG" if (rag_enabled and always_on) else ("with RAG" if rag_enabled else "without RAG")
    print(f"\n💬 Chat started {rag_status}! [knowledge_approach={knowledge_approach}] Type 'exit' to quit.")
    print("-" * 60)

    # Keep a short history buffer to fit in context

    while True:
        try:
            try:
                user_input = input(f"\n{user_name}: ")
            except (EOFError, OSError):
                # Non-interactive mode - exit gracefully
                print("\n\n⚠️  Non-interactive mode detected. Exiting...")
                print("   To use the chat interface, run without piped input:")
                print("   docker-compose run --rm kaya-chatbot python src/chat/chat.py")
                break
            
            if user_input.lower() in ['exit', 'quit']:
                break

            # Add user message to history
            history.append(f"{user_name}: {user_input}")

            # Keep history manageable
            if len(history) > max_history_lines:
                history = history[-max_history_lines:]

            # Always retrieve RAG context (always-on mode)
            context = ""
            if rag_enabled and retriever:
                try:
                    context = retriever.retrieve_all(user_input, knowledge_approach=knowledge_approach)
                    if context:
                        conv_count = context.count("--- Conversa ")
                        kb_count = context.count("---") - conv_count
                        print(f"📚 RAG: {conv_count} conversation chunk(s)" + (f", {kb_count} knowledge fact(s)" if kb_count > 0 else ""))
                except Exception as e:
                    print(f"⚠️  RAG retrieval failed: {e}")
                    context = ""

            # Build user message: RAG context + recent history + current input
            message_parts = []
            if context:
                message_parts.append(context)

            if len(history) > 1:
                recent = "\n".join(history[-5:-1])  # Last 5 messages excluding current
                message_parts.append(f"Conversa recente:\n{recent}")

            message_parts.append(f"{user_name}: {user_input}")
            user_message = "\n\n".join(message_parts)

            mode_indicator = "always-on RAG" if (rag_enabled and context) else "no context"
            print(f"   [Mode: {mode_indicator}]")

            # Format prompt using the tokenizer's chat template (model-agnostic)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")

            # Generate response
            # We stop at newline to get just one message
            print(f"{bot_name}: ", end="", flush=True)

            # Use a streamer to show text as it generates
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            inf_config = config.get('inference', {})
            outputs = model.generate(
                **inputs,
                max_new_tokens=inf_config.get('max_new_tokens', 512),
                temperature=inf_config.get('temperature', 1.0),
                do_sample=True,
                top_p=inf_config.get('top_p', 0.95),
                top_k=inf_config.get('top_k', 64),
                repetition_penalty=inf_config.get('repetition_penalty', 1.0),
                use_cache=True,
                streamer=streamer
            )

            # The streamer prints the output. We need to capture it for history too.
            # Since streamer doesn't return the text, we decode the output manually to update history.
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Clean up if it generated the stop token text
            response_text = response_text.split('\n')[0].replace(f"{user_name}:", "")

            if response_text:
                history.append(f"{bot_name}: {response_text}")
                if session_memory:
                    session_memory.save(history)
                entry = {
                    "interaction_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user_message": user_input,
                    "assistant_response": response_text,
                }
                with open(log_file, "a", encoding="utf-8") as _lf:
                    _lf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except KeyboardInterrupt:
            if session_memory:
                session_memory.save(history)
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()