"""
Interactive Chat Script
Allows the user to chat with the fine-tuned Kaya model.
"""
import os
import yaml
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

def main():
    print("=" * 60)
    print("Kaya Chat Interface")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_dir = config['training']['output_dir']
    max_seq_length = config['model']['max_seq_length']
    system_prompt = config['data']['system_prompt']
    
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
    
    # Setup chat
    user_name = input("\nEnter your name (default: User): ").strip() or "User"
    bot_name = "Kaya"
    
    print(f"\n💬 Chat started! Type 'exit' to quit.")
    print("-" * 60)
    
    # Keep a short history buffer to fit in context
    history = []
    max_history_lines = 10 
    
    while True:
        try:
            user_input = input(f"\n{user_name}: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Add user message to history
            history.append(f"{user_name}: {user_input}")
            
            # Keep history manageable
            if len(history) > max_history_lines:
                history = history[-max_history_lines:]
            
            # Construct the conversation text
            conversation_text = "\n".join(history) + f"\n{bot_name}:"
            
            # Format prompt manually to pre-fill the assistant's turn
            # We use the Llama-3 format: System -> User -> Assistant (started with history)
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"Continue the conversation.<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{conversation_text}"
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
