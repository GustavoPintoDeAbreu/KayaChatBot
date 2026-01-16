"""
FastAPI backend for Kaya Chatbot Web App.

Features:
- Email-based authentication
- Chat endpoint with Ollama integration
- RAG support for conversation retrieval
- Conversation history management
"""

import os
import yaml
import secrets
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# API Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
ALLOWED_EMAILS = os.getenv("ALLOWED_EMAILS", "friend1@example.com,friend2@example.com").split(",")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

# Initialize FastAPI
app = FastAPI(
    title="Kaya Chatbot API",
    description="Web API for Kaya Chatbot with RAG support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory session store (in production, use Redis/database)
sessions: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class LoginRequest(BaseModel):
    email: EmailStr

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    rag_context_used: bool
    retrieved_chunks: int

class ConversationHistory(BaseModel):
    messages: List[Dict[str, str]]
    created_at: datetime
    last_activity: datetime

# RAG Retriever (lazy loaded)
retriever = None

def get_retriever():
    """Lazy load RAG retriever."""
    global retriever
    if retriever is None:
        try:
            from src.chat.retriever import get_retriever
            retriever = get_retriever(config)
        except ImportError:
            # Try Docker import path
            from chat.retriever import get_retriever
            retriever = get_retriever(config)
    return retriever

def authenticate_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authenticate user with token."""
    token = credentials.credentials
    if token not in sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    session = sessions[token]
    if datetime.now() > session["expires_at"]:
        del sessions[token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )

    return session["email"]

@app.post("/auth/login", response_model=dict)
async def login(request: LoginRequest):
    """Authenticate user and return session token."""
    email = request.email

    # Check if email is allowed
    if email not in ALLOWED_EMAILS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not authorized"
        )

    # Create session token
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "email": email,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=24)
    }

    return {"token": token, "message": "Login successful"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, email: str = Depends(authenticate_user)):
    """Chat with Kaya using Ollama and RAG."""
    message = request.message
    conversation_id = request.conversation_id or secrets.token_urlsafe(16)

    # Get or create conversation history
    if conversation_id not in sessions:
        sessions[conversation_id] = {
            "messages": [],
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }

    conversation = sessions[conversation_id]
    conversation["last_activity"] = datetime.now()

    # Add user message to history
    conversation["messages"].append({"role": "user", "content": message})

    # Keep history manageable (last 20 messages)
    if len(conversation["messages"]) > 20:
        conversation["messages"] = conversation["messages"][-20:]

    # Detect if this is a question (for RAG)
    is_question = any(keyword in message.lower() for keyword in [
        'o que', 'como', 'quando', 'onde', 'quem', 'porque', 'porquê', 'qual',
        'quantos', 'quantas', '?', 'diz', 'dizes', 'sabes', 'conheces',
        'what', 'how', 'who', 'where', 'when', 'why'
    ])

    # Retrieve RAG context if it's a question
    context = ""
    retrieved_chunks = 0
    rag_used = False

    if is_question:
        try:
            rag_retriever = get_retriever()
            retrieved = rag_retriever.retrieve(message)
            if retrieved:
                context = rag_retriever.format_context(retrieved)
                retrieved_chunks = len(retrieved)
                rag_used = True
        except Exception as e:
            print(f"RAG error: {e}")
            context = ""

    # Build prompt for Ollama
    if rag_used and context:
        # Q&A mode with RAG context
        prompt = f"{context}\n\nCom base nestas conversas passadas, responde:\n{message}"
    else:
        # Casual conversation mode
        if len(conversation["messages"]) > 2:
            # Include recent history
            recent_messages = conversation["messages"][-5:]  # Last 5 exchanges
            history_text = "\n".join([
                f"{msg['role'].title()}: {msg['content']}"
                for msg in recent_messages[:-1]  # Exclude current message
            ])
            prompt = f"Conversa recente:\n{history_text}\n\nUser: {message}"
        else:
            prompt = message

    # Call Ollama API
    try:
        ollama_payload = {
            "model": "kaya-chatbot",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config["inference"]["temperature"],
                "top_p": config["inference"]["top_p"],
                "num_predict": config["inference"]["max_new_tokens"],
                "repeat_penalty": config["inference"]["repetition_penalty"]
            }
        }

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=ollama_payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        bot_response = result["response"].strip()

        # Clean up response
        bot_response = bot_response.split('\n')[0]  # Take first line only
        bot_response = bot_response.replace("User:", "").replace("Assistant:", "").strip()

        if not bot_response:
            bot_response = "..."

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

    # Add bot response to history
    conversation["messages"].append({"role": "assistant", "content": bot_response})

    return ChatResponse(
        response=bot_response,
        conversation_id=conversation_id,
        rag_context_used=rag_used,
        retrieved_chunks=retrieved_chunks
    )

@app.get("/conversations", response_model=List[Dict[str, Any]])
async def get_conversations(email: str = Depends(authenticate_user)):
    """Get user's conversation history."""
    user_conversations = []
    for conv_id, conv_data in sessions.items():
        if isinstance(conv_data, dict) and "messages" in conv_data:
            user_conversations.append({
                "id": conv_id,
                "message_count": len(conv_data["messages"]),
                "created_at": conv_data["created_at"],
                "last_activity": conv_data["last_activity"]
            })

    # Sort by last activity
    user_conversations.sort(key=lambda x: x["last_activity"], reverse=True)
    return user_conversations

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, email: str = Depends(authenticate_user)):
    """Delete a conversation."""
    if conversation_id in sessions:
        del sessions[conversation_id]
        return {"message": "Conversation deleted"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)