"""
Extract and clean all messages from WhatsApp exports.
Creates unified cleaned message dataset and chunks for synthetic generation.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
import tiktoken

# Make src/ importable when run from any CWD
import os
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config_loader import load_config
from src.data.identity_resolver import SenderResolver

# Load configuration via the single entry point (load_config — never read
# config.yaml directly), so profile resolution + validation are applied.
CONFIG_PATH = _REPO_ROOT / "config.yaml"
config = load_config(str(CONFIG_PATH))

# Configuration
MAX_TOKENS_PER_CHUNK = 50000  # 50K tokens per finetune chunk

# Detect if running in Docker
if os.path.exists('/app'):
    DATA_DIR = Path("/app/data")
else:
    DATA_DIR = Path(__file__).parent.parent.parent / "data"

OUTPUT_CLEANED = DATA_DIR / "all_messages_cleaned.jsonl"
OUTPUT_FINETUNE_CHUNKS = DATA_DIR / "finetune_chunks.jsonl"

# Initialize tokenizer for token counting
tokenizer = tiktoken.encoding_for_model("gpt-4")


def _build_resolver() -> SenderResolver:
    """Build a SenderResolver from config and group_members.json."""
    members_file = DATA_DIR / "group_members.json"
    sender_aliases = config.get("data", {}).get("sender_aliases", {}) or {}
    return SenderResolver(members_file, sender_aliases)


class MessageExtractor:
    """Extract and clean messages from multiple sources."""

    def __init__(self):
        self.messages = []
        self._resolver: Optional[SenderResolver] = None
        # Lazily initialised so that unit tests that mock DATA_DIR can work
        try:
            self._resolver = _build_resolver()
        except Exception:
            pass  # resolver unavailable (e.g. missing members file in tests)
        
    def clean_text(self, text: str) -> str:
        """Clean message text by removing noise."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove Unicode mentions (WhatsApp)
        text = re.sub(r'@\u2068[^\u2069]*\u2069', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def is_valid_message(self, text: str) -> bool:
        """Check if message is valid (not noise)."""
        if not text or len(text) < 3:
            # Allow common short responses
            common_short = ['lol', 'sim', 'não', 'ok', 'oi', 'olá', 'wtf', 'lmao']
            return text.lower() in common_short
        return True
    
    def extract_whatsapp(self, file_path: Path) -> List[Dict]:
        """Extract messages from WhatsApp chat export.

        Handles the real iOS/Android export format:
            M/DD/YY, HH:MM - Sender: Message
        Multiline messages (continuation lines without a timestamp) are merged
        into the previous message.
        """
        print(f"\n📱 Processing WhatsApp: {file_path.name}")
        messages = []

        # Format: "3/26/20, 15:28 - Sender Name: message text"
        msg_pattern = re.compile(
            r'^(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - ([^:]+): (.+)$'
        )
        # System-message line (no "Sender: " part): "3/26/20, 15:28 - Gil João added you"
        sys_pattern = re.compile(
            r'^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - .+$'
        )

        # Skip tokens that appear in content (not system messages)
        SKIP_TOKENS = [
            'media omitted', 'file attached',
            'this message was edited',
            'deleted this message', 'changed the subject',
        ]

        current_date: str = ""
        current_sender: str = ""
        current_content: list = []

        def flush():
            """Flush accumulated lines as a single message."""
            if not current_content or not current_sender:
                return
            full_text = " ".join(current_content)
            cleaned_text = self.clean_text(full_text)
            if not cleaned_text:
                return
            if any(skip in cleaned_text.lower() for skip in SKIP_TOKENS):
                return
            if not self.is_valid_message(cleaned_text):
                return
            resolved_sender = current_sender
            if self._resolver is not None:
                resolved_sender = self._resolver.resolve(current_sender)
            messages.append({
                'timestamp': current_date,
                'sender': resolved_sender,
                'text': cleaned_text,
                'source': 'whatsapp',
            })

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for raw_line in lines:
                line = raw_line.rstrip('\n')

                # --- New timestamped message ---
                match = msg_pattern.match(line)
                if match:
                    flush()
                    datetime_str, sender, text = match.groups()
                    try:
                        # Try 2-digit year first (most common), then 4-digit
                        try:
                            ts = datetime.strptime(datetime_str, "%m/%d/%y, %H:%M")
                        except ValueError:
                            ts = datetime.strptime(datetime_str, "%m/%d/%Y, %H:%M")
                        current_date = ts.isoformat()
                    except Exception:
                        current_date = datetime_str
                    current_sender = sender.strip()
                    current_content = [text]
                    continue

                # --- System message (no Sender: part) — flush and skip ---
                if sys_pattern.match(line):
                    flush()
                    current_sender = ""
                    current_content = []
                    continue

                # --- Continuation line of previous message ---
                stripped = line.strip()
                if stripped and current_content:
                    current_content.append(stripped)

            flush()  # flush last pending message

        except Exception as e:
            print(f"❌ Error processing WhatsApp file: {e}")
            return []

        print(f"✅ Extracted {len(messages)} WhatsApp messages")
        return messages
    
    def merge_consecutive_messages(self, messages: List[Dict]) -> List[Dict]:
        """Merge consecutive messages from the same sender."""
        if not messages:
            return []
        
        merged = []
        current = messages[0].copy()
        
        for msg in messages[1:]:
            # If same sender within 5 minutes, merge
            if msg['sender'] == current['sender']:
                try:
                    curr_time = datetime.fromisoformat(current['timestamp'])
                    msg_time = datetime.fromisoformat(msg['timestamp'])
                    
                    if (msg_time - curr_time).total_seconds() < 300:  # 5 minutes
                        current['text'] += ' ' + msg['text']
                        current['timestamp'] = msg['timestamp']  # Update to latest
                        continue
                except (ValueError, KeyError, TypeError):
                    pass
            
            # Different sender or too much time passed
            merged.append(current)
            current = msg.copy()
        
        merged.append(current)
        return merged
    
    def extract_all(self) -> List[Dict]:
        """Extract messages from all sources."""
        all_messages = []
        
        # Extract WhatsApp messages
        wpp_dir = DATA_DIR / "wpp"
        if wpp_dir.exists():
            for txt_file in wpp_dir.glob("*.txt"):
                if 'processed' not in txt_file.name.lower():  # Skip already processed files
                    messages = self.extract_whatsapp(txt_file)
                    all_messages.extend(messages)
        
        # Sort by timestamp
        all_messages.sort(key=lambda x: x['timestamp'])
        
        print(f"\n📊 Total messages before merging: {len(all_messages)}")
        
        # Merge consecutive messages
        all_messages = self.merge_consecutive_messages(all_messages)
        
        print(f"📊 Total messages after merging: {len(all_messages)}")
        
        return all_messages


class MessageChunker:
    """Chunk messages into token-limited segments for synthetic generation (finetune chunks)."""
    
    def __init__(self, max_tokens: int = MAX_TOKENS_PER_CHUNK):
        self.max_tokens = max_tokens
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(tokenizer.encode(text))
    
    def create_finetune_chunks(self, messages: List[Dict]) -> List[Dict]:
        """Create finetune chunks of messages up to max_tokens."""
        finetune_chunks = []
        current_finetune_chunk = []
        current_tokens = 0
        
        for msg in messages:
            # Format message
            formatted = f"{msg['sender']}: {msg['text']}"
            msg_tokens = self.count_tokens(formatted)
            
            # If adding this message exceeds limit, save current finetune chunk
            if current_tokens + msg_tokens > self.max_tokens and current_finetune_chunk:
                chunk_text = '\n'.join([
                    f"{m['sender']}: {m['text']}" for m in current_finetune_chunk
                ])
                
                finetune_chunks.append({
                    'chunk_id': len(finetune_chunks),
                    'messages': current_finetune_chunk,
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'message_count': len(current_finetune_chunk)
                })
                
                current_finetune_chunk = []
                current_tokens = 0
            
            # Add message to current finetune chunk
            current_finetune_chunk.append(msg)
            current_tokens += msg_tokens
        
        # Add final finetune chunk
        if current_finetune_chunk:
            chunk_text = '\n'.join([
                f"{m['sender']}: {m['text']}" for m in current_finetune_chunk
            ])
            
            finetune_chunks.append({
                'chunk_id': len(finetune_chunks),
                'messages': current_finetune_chunk,
                'text': chunk_text,
                'token_count': current_tokens,
                'message_count': len(current_finetune_chunk)
            })
        
        return finetune_chunks


def main():
    """Main extraction pipeline."""
    print("=" * 60)
    print("🚀 MESSAGE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Create output directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Extract messages
    extractor = MessageExtractor()
    messages = extractor.extract_all()
    
    if not messages:
        print("\n❌ No messages extracted!")
        return
    
    # Save cleaned messages
    print(f"\n💾 Saving cleaned messages to {OUTPUT_CLEANED.name}")
    with open(OUTPUT_CLEANED, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(messages)} cleaned messages")
    
    # Create finetune chunks
    print(f"\n🔨 Creating finetune chunks (max {MAX_TOKENS_PER_CHUNK:,} tokens each)")
    chunker = MessageChunker(max_tokens=MAX_TOKENS_PER_CHUNK)
    finetune_chunks = chunker.create_finetune_chunks(messages)
    
    # Save finetune chunks
    print(f"\n💾 Saving finetune chunks to {OUTPUT_FINETUNE_CHUNKS.name}")
    with open(OUTPUT_FINETUNE_CHUNKS, 'w', encoding='utf-8') as f:
        for chunk in finetune_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(finetune_chunks)} finetune chunks")
    
    # Statistics
    print("\n" + "=" * 60)
    print("📊 EXTRACTION STATISTICS")
    print("=" * 60)
    print(f"Total messages: {len(messages)}")
    print(f"Total finetune chunks: {len(finetune_chunks)}")
    print(f"Avg messages per finetune chunk: {len(messages) / len(finetune_chunks):.1f}")
    
    # Token statistics
    total_tokens = sum(chunk['token_count'] for chunk in finetune_chunks)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Avg tokens per finetune chunk: {total_tokens / len(finetune_chunks):,.0f}")
    
    # Source breakdown
    source_counts = defaultdict(int)
    for msg in messages:
        source_counts[msg['source']] += 1
    
    print("\nMessages by source:")
    for source, count in source_counts.items():
        print(f"  {source.capitalize()}: {count} ({count/len(messages)*100:.1f}%)")
    
    print("\n✅ Extraction complete!")
    print(f"\nNext steps:")
    print(f"  1. Review {OUTPUT_CLEANED.name} for quality")
    print(f"  2. Run generate_local_synthetic.py to create training data (fully on-prem)")


if __name__ == "__main__":
    main()