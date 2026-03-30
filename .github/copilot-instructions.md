# GitHub Copilot Instructions

## Project Overview
KayaChatBot is an AI assistant bot for a Portuguese friend group chat called **Kaya**. The bot is NOT a group member — it is an assistant with access to the group's collective memory. It has long-term memory of facts, events, and people learned from real WhatsApp and Instagram conversation history (via RAG + fine-tuning). It communicates in **European Portuguese or English**; it does NOT need to use the group's specific slang or lingo. The focus is on natural language ability and factual memory, not mimicking any particular speech style.

**Key architecture decisions:**
- RAG is **always on** — every message (casual or Q&A) retrieves context from conversation history and the curated knowledge base. The model never answers from fine-tune memory alone.
- Group member knowledge is stored in `data/group_members.json` (injected into system prompt) and `data/group_knowledge.json` (embedded into ChromaDB `kaya_knowledge_base` collection).
- The `rag.knowledge_approach` config toggle (`both` / `json_only` / `chromadb_only` / `none`) enables benchmarking different knowledge injection strategies.

## Environment Setup
- Always run code using the virtual environment named 'kaya_chatbot' located in the `kaya_chatbot_env/` directory
- Ensure the virtual environment is activated before executing any Python scripts or commands
- Always install Python packages within the 'kaya_chatbot' virtual environment
- **Prefer using Python executable directly** (e.g., `python script.py`) always inside virtual environment

## Coding Preferences
- Avoid creating backup and temporary code files when rewriting existing ones; either replace the existing file or create a new one and delete the old one

 - Branching & PRs in Plans: Whenever you're asked to create a plan, include explicit steps to:
	 - create a new Git branch for the work,
	 - open a pull request (PR) for that branch,
	 - run tests and verify the change (including running the project in Docker where applicable),
	 - iterate until the implementation is well tested and well implemented,
	 - merge the PR after tests pass and approvals are obtained.

## Docker Usage
- When requested and building images, make sure to erase previously built images, containers, volumes, or builds to prevent storage overload (e.g. using `docker system prune` or similar)
- After completing any change, always test it inside Docker to verify it works correctly in the containerized environment
