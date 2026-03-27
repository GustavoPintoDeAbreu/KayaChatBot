"""
Interactive Chat Script
Allows the user to chat with the fine-tuned Kaya model with RAG support.
"""
import os
import sys
import yaml
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Add parent directory to path for imports (Docker compatibility)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("=" * 60)
    print("Kaya Chat Interface with RAG")
    print("=" * 60)

    # Load configuration (Docker or local environment)
    docker_config = '/app/config.yaml'
    local_config = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    config_path = docker_config if os.path.exists(docker_config) else local_config

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_dir = config['training']['output_dir']
    max_seq_length = config['model']['max_seq_length']
    system_prompt = config['data']['system_prompt']
    rag_enabled = config.get('rag', {}).get('enabled', False)

    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"\n❌ Error: Model not found at {model_dir}")
        return

    # Load model
    print(f"\nLoading model... (this may take a minute)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"✓ Model loaded!")

    # Initialize RAG retriever if enabled
    retriever = None
    if rag_enabled:
        try:
            # Try different import paths for Docker and local environments
            try:
                from src.chat.retriever import get_retriever
            except ImportError:
                # Docker environment - src is in sys.path
                from chat.retriever import get_retriever
            
            retriever = get_retriever(config)
            print("✓ RAG retriever initialized!")
        except Exception as e:
            print(f"⚠️  RAG initialization failed: {e}")
            print("   Continuing without RAG...")
            rag_enabled = False

    # Setup chat
    try:
        user_name = input("\nEnter your name (default: User): ").strip() or "User"
    except (EOFError, OSError):
        # Non-interactive mode (e.g., piped input)
        user_name = "User"
        print("\n⚠️  Non-interactive mode detected. Using default name: User")
    
    bot_name = "Kaya"

    rag_status = "with RAG" if rag_enabled else "without RAG"
    print(f"\n💬 Chat started {rag_status}! Type 'exit' to quit.")
    print("-" * 60)

    # Keep a short history buffer to fit in context
    history = []
    max_history_lines = 10

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

            # Detect if this is a question (Q&A mode) or casual message (conversation mode)
            is_question = any(keyword in user_input.lower() for keyword in [
                'o que', 'como', 'quando', 'onde', 'quem', 'porque', 'porquê', 'qual', 
                'quantos', 'quantas', '?', 'diz', 'dizes', 'sabes', 'conheces', 'what', 'how'
            ])

            # Retrieve relevant context if RAG is enabled
            context = ""
            if rag_enabled and retriever:
                try:
                    retrieved_chunks = retriever.retrieve(user_input)
                    context = retriever.format_context(retrieved_chunks)

                    if retrieved_chunks:
                        print(f"📚 Retrieved {len(retrieved_chunks)} relevant conversation chunks")
                        for chunk in retrieved_chunks[:2]:  # Show first 2 for debugging
                            print(f"   • {chunk['message_count']} messages, similarity: {chunk['similarity_score']:.3f}")
                except Exception as e:
                    print(f"⚠️  RAG retrieval failed: {e}")
                    context = ""

            # Build the user message based on mode
            if is_question and context:
                # Q&A mode with RAG: Provide context and ask for answer
                user_message = f"{context}\n\nCom base nestas conversas passadas, responde:\n{user_input}"
                mode_indicator = "Q&A with RAG"
            elif is_question:
                # Q&A mode without context: Direct question
                user_message = user_input
                mode_indicator = "Q&A"
            else:
                # Conversation mode: Include recent history
                if len(history) > 1:
                    recent_history = "\n".join(history[-5:])  # Last 5 messages
                    user_message = f"Conversa recente:\n{recent_history}\n\n{user_name}: {user_input}"
                else:
                    user_message = user_input
                mode_indicator = "Casual"
            
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

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

            # Generate response
            # We stop at newline to get just one message
            print(f"{bot_name}: ", end="", flush=True)

            # Use a streamer to show text as it generates
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                stop_strings=["\n", f"{user_name}:"],
                tokenizer=tokenizer,
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

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()