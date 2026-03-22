# GitHub Copilot Instructions

## Project Overview
KayaChatBot is an AI "extra member" for a Portuguese friend group chat. The goal is for the bot to feel like someone who was always part of the group — it has a long-term memory of facts, events, and people learned from real WhatsApp and Instagram conversation history (via RAG + fine-tuning). It communicates in **European Portuguese or English**; it does NOT need to use the group's specific slang or lingo. The focus is on natural language ability and factual memory, not mimicking any particular speech style.

## Environment Setup
- Always run code using the virtual environment named 'kaya_chatbot' located in the `kaya_chatbot_env/` directory
- Ensure the virtual environment is activated before executing any Python scripts or commands
- Always install Python packages within the 'kaya_chatbot' virtual environment
- **Prefer using Python executable directly** (e.g., `python script.py`) always inside virtual environment

## Coding Preferences
- Avoid creating backup and temporary code files when rewriting existing ones; either replace the existing file or create a new one and delete the old one

## Docker Usage
- When requested and building images, make sure to erase previously built images, containers, volumes, or builds to prevent storage overload (e.g. using `docker system prune` or similar)
- After completing any change, always test it inside Docker to verify it works correctly in the containerized environment
