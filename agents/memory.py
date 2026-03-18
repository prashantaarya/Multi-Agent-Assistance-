# agents/memory.py
"""
JARVIS Memory System (Industry Best Practice)

Implements two types of memory:
1. ConversationMemory - Short-term chat history with context window
2. SemanticMemory - Long-term knowledge with embedding-based retrieval

Storage Backends:
- LOCAL (default): JSON files + pickle (for testing/development)
- FUTURE: MongoDB, PostgreSQL, Redis, Pinecone, Chroma

To switch backends:
1. Set MEMORY_STORAGE_BACKEND in .env (e.g., "mongodb")
2. Set the corresponding database URL (e.g., MONGODB_URL)
3. Implement the StorageBackend interface for your database

This enables JARVIS to:
- Remember conversation context
- Learn from past interactions
- Retrieve relevant information for better responses
"""

import json
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Protocol
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import deque
import hashlib
import pickle

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger("jarvis.memory")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration from .env
# ──────────────────────────────────────────────────────────────────────────────

STORAGE_BACKEND = os.getenv("MEMORY_STORAGE_BACKEND", "local").lower()
MONGODB_URL = os.getenv("MONGODB_URL")
POSTGRES_URL = os.getenv("POSTGRES_URL")
REDIS_URL = os.getenv("REDIS_URL")


# ──────────────────────────────────────────────────────────────────────────────
# Storage Backend Interface (for future database support)
# ──────────────────────────────────────────────────────────────────────────────

class StorageBackend(ABC):
    """
    Abstract interface for memory storage backends.
    
    Implement this interface to add support for new databases:
    - MongoDBBackend
    - PostgreSQLBackend
    - RedisBackend
    """
    
    @abstractmethod
    def save_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Save a conversation session"""
        pass
    
    @abstractmethod
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation session"""
        pass
    
    @abstractmethod
    def list_sessions(self, limit: int = 20) -> List[str]:
        """List recent session IDs"""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        pass
    
    @abstractmethod
    def save_memory_entries(self, entries: List[Dict[str, Any]]) -> bool:
        """Save semantic memory entries"""
        pass
    
    @abstractmethod
    def load_memory_entries(self) -> List[Dict[str, Any]]:
        """Load all semantic memory entries"""
        pass
    
    @abstractmethod
    def clear_memory(self) -> bool:
        """Clear all semantic memory"""
        pass


class LocalStorageBackend(StorageBackend):
    """
    Local file-based storage backend (JSON + pickle).
    Perfect for development and testing.
    """
    
    def __init__(self, base_dir: str = "memory"):
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "conversation_logs"
        self.vector_store_path = self.base_dir / "vector_store.pkl"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.sessions_dir / "sessions_index.json"
    
    def save_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self._update_index(session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
        return None
    
    def list_sessions(self, limit: int = 20) -> List[str]:
        try:
            if self._index_file.exists():
                with open(self._index_file, "r") as f:
                    index = json.load(f)
                    return index.get("recent_sessions", [])[:limit]
        except Exception:
            pass
        return []
    
    def delete_session(self, session_id: str) -> bool:
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def save_memory_entries(self, entries: List[Dict[str, Any]]) -> bool:
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.vector_store_path, "wb") as f:
                pickle.dump(entries, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save memory entries: {e}")
            return False
    
    def load_memory_entries(self) -> List[Dict[str, Any]]:
        try:
            if self.vector_store_path.exists():
                # Check if file is empty or corrupted
                if self.vector_store_path.stat().st_size == 0:
                    return []
                with open(self.vector_store_path, "rb") as f:
                    data = pickle.load(f)
                    # Handle old format (list of MemoryEntry objects) vs new format (list of dicts)
                    if data and hasattr(data[0], 'to_dict'):
                        return [e.to_dict() for e in data]
                    return data
        except Exception as e:
            logger.warning(f"Failed to load memory entries: {e}")
            # Remove corrupted file
            try:
                self.vector_store_path.unlink()
            except:
                pass
        return []
    
    def clear_memory(self) -> bool:
        try:
            if self.vector_store_path.exists():
                self.vector_store_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False
    
    def _update_index(self, session_id: str):
        try:
            if self._index_file.exists():
                with open(self._index_file, "r") as f:
                    index = json.load(f)
            else:
                index = {"recent_sessions": []}
            
            if session_id in index["recent_sessions"]:
                index["recent_sessions"].remove(session_id)
            index["recent_sessions"].insert(0, session_id)
            index["recent_sessions"] = index["recent_sessions"][:100]
            
            with open(self._index_file, "w") as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update index: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Future Database Backends (Stubs)
# ──────────────────────────────────────────────────────────────────────────────

class MongoDBBackend(StorageBackend):
    """
    MongoDB storage backend (FUTURE IMPLEMENTATION).
    
    To use:
    1. pip install pymongo
    2. Set MONGODB_URL in .env
    3. Set MEMORY_STORAGE_BACKEND="mongodb"
    """
    
    def __init__(self, connection_url: str):
        self.url = connection_url
        # TODO: Initialize MongoDB client
        # from pymongo import MongoClient
        # self.client = MongoClient(connection_url)
        # self.db = self.client.jarvis
        raise NotImplementedError("MongoDB backend not yet implemented. Set MEMORY_STORAGE_BACKEND=local")
    
    def save_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        raise NotImplementedError()
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError()
    
    def list_sessions(self, limit: int = 20) -> List[str]:
        raise NotImplementedError()
    
    def delete_session(self, session_id: str) -> bool:
        raise NotImplementedError()
    
    def save_memory_entries(self, entries: List[Dict[str, Any]]) -> bool:
        raise NotImplementedError()
    
    def load_memory_entries(self) -> List[Dict[str, Any]]:
        raise NotImplementedError()
    
    def clear_memory(self) -> bool:
        raise NotImplementedError()


def get_storage_backend() -> StorageBackend:
    """
    Factory function to get the appropriate storage backend.
    Based on MEMORY_STORAGE_BACKEND environment variable.
    """
    if STORAGE_BACKEND == "mongodb":
        if not MONGODB_URL:
            raise ValueError("MONGODB_URL not set in .env")
        return MongoDBBackend(MONGODB_URL)
    elif STORAGE_BACKEND == "postgresql":
        raise NotImplementedError("PostgreSQL backend not yet implemented")
    elif STORAGE_BACKEND == "redis":
        raise NotImplementedError("Redis backend not yet implemented")
    else:
        # Default to local file storage
        return LocalStorageBackend()


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

class MemoryMessage(BaseModel):
    """Single message in conversation history"""
    id: str = Field(default_factory=lambda: hashlib.md5(str(datetime.now(timezone.utc).timestamp()).encode()).hexdigest()[:12])
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional: capability that was used
    capability: Optional[str] = None
    # Optional: confidence of the response
    confidence: Optional[float] = None


class ConversationSession(BaseModel):
    """A conversation session with multiple messages"""
    session_id: str
    user_id: Optional[str] = None
    messages: List[MemoryMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Summary for long conversations
    summary: Optional[str] = None
    
    def add_message(self, role: str, content: str, **kwargs) -> MemoryMessage:
        """Add a message to the session"""
        msg = MemoryMessage(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        return msg
    
    def get_recent(self, n: int = 10) -> List[MemoryMessage]:
        """Get the n most recent messages"""
        return self.messages[-n:] if self.messages else []
    
    def to_prompt_format(self, max_messages: int = 10) -> str:
        """Convert recent messages to a prompt-friendly format"""
        recent = self.get_recent(max_messages)
        lines = []
        for msg in recent:
            role_prefix = "User" if msg.role == "user" else "JARVIS"
            lines.append(f"{role_prefix}: {msg.content}")
        return "\n".join(lines)


class MemorySearchResult(BaseModel):
    """Result from semantic memory search"""
    content: str
    score: float
    source: str  # "conversation" | "knowledge" | "task"
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation Memory (Short-term)
# ──────────────────────────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Short-term conversation memory with context window management.
    
    Features:
    - Session-based conversation tracking
    - Configurable context window (number of messages to remember)
    - Automatic summarization for long conversations
    - Pluggable storage backend (local files, MongoDB, PostgreSQL, etc.)
    
    Current: Local file storage (JSON)
    Future: Set MEMORY_STORAGE_BACKEND in .env to use database
    """
    
    def __init__(
        self,
        storage_backend: StorageBackend = None,
        context_window: int = 20,
        max_sessions: int = 100
    ):
        # Use provided backend or get from environment
        self._storage = storage_backend or get_storage_backend()
        self.context_window = context_window
        self.max_sessions = max_sessions
        
        # In-memory session cache
        self._sessions: Dict[str, ConversationSession] = {}
        self._active_session_id: Optional[str] = None
        
        # Load existing sessions
        self._load_sessions()
    
    def _load_sessions(self):
        """Load sessions from storage backend"""
        try:
            session_ids = self._storage.list_sessions(limit=self.max_sessions)
            for session_id in session_ids:
                self._load_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
    
    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load a single session from storage"""
        try:
            data = self._storage.load_session(session_id)
            if data:
                session = ConversationSession(**data)
                self._sessions[session_id] = session
                return session
        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
        return None
    
    def _save_session(self, session: ConversationSession):
        """Save a session to storage"""
        try:
            self._storage.save_session(
                session.session_id, 
                session.model_dump(mode="json")
            )
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
    
    def create_session(self, user_id: Optional[str] = None) -> ConversationSession:
        """Create a new conversation session"""
        now = datetime.now(timezone.utc)
        session_id = f"session_{now.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(now.timestamp()).encode()).hexdigest()[:6]}"
        session = ConversationSession(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        self._active_session_id = session_id
        self._save_session(session)
        logger.info(f"Created new session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID"""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self._load_session(session_id)
    
    def get_or_create_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> ConversationSession:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                self._active_session_id = session_id
                return session
        
        # Use active session if available
        if self._active_session_id and self._active_session_id in self._sessions:
            return self._sessions[self._active_session_id]
        
        return self.create_session(user_id)
    
    def add_interaction(
        self,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        capability: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[MemoryMessage, MemoryMessage]:
        """Add a user-assistant interaction to memory"""
        session = self.get_or_create_session(session_id)
        
        # Add user message
        user_msg = session.add_message(
            role="user",
            content=user_message,
            metadata=metadata or {}
        )
        
        # Add assistant response
        assistant_msg = session.add_message(
            role="assistant",
            content=assistant_response,
            capability=capability,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Save to disk
        self._save_session(session)
        
        return user_msg, assistant_msg
    
    def get_context(self, session_id: Optional[str] = None, max_messages: Optional[int] = None) -> str:
        """Get conversation context for prompting"""
        session = self.get_or_create_session(session_id)
        n = max_messages or self.context_window
        return session.to_prompt_format(n)
    
    def get_recent_messages(self, session_id: Optional[str] = None, n: int = 10) -> List[MemoryMessage]:
        """Get recent messages from a session"""
        session = self.get_or_create_session(session_id)
        return session.get_recent(n)
    
    def clear_session(self, session_id: str):
        """Clear a session's messages"""
        if session_id in self._sessions:
            self._sessions[session_id].messages = []
            self._save_session(self._sessions[session_id])
    
    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions"""
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True
        )[:limit]
        
        return [
            {
                "session_id": s.session_id,
                "message_count": len(s.messages),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "preview": s.messages[-1].content[:100] if s.messages else ""
            }
            for s in sessions
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Memory (Long-term with embeddings)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """Entry in semantic memory"""
    id: str
    content: str
    source: str  # "conversation", "knowledge", "task", "user_fact"
    timestamp: datetime
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "embedding": self.embedding,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {})
        )


class SemanticMemory:
    """
    Long-term semantic memory with embedding-based retrieval.
    
    Features:
    - Store important facts, knowledge, and user preferences
    - Retrieve relevant memories based on semantic similarity
    - Automatic importance scoring
    - Pluggable storage backend
    
    Current: Local file storage (pickle)
    Future: Set MEMORY_STORAGE_BACKEND in .env to use:
    - Vector databases (Pinecone, Weaviate, Chroma)
    - MongoDB with vector search
    - PostgreSQL with pgvector
    """
    
    def __init__(
        self,
        storage_backend: StorageBackend = None,
        max_entries: int = 1000
    ):
        # Use provided backend or get from environment
        self._storage = storage_backend or get_storage_backend()
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []
        
        # Load existing memories
        self._load()
    
    def _load(self):
        """Load memories from storage backend"""
        try:
            entries_data = self._storage.load_memory_entries()
            self._entries = [
                MemoryEntry.from_dict(e) if isinstance(e, dict) else e 
                for e in entries_data
            ]
            if self._entries:
                logger.info(f"Loaded {len(self._entries)} memories")
        except Exception as e:
            logger.warning(f"Failed to load semantic memory: {e}")
            self._entries = []
    
    def _save(self):
        """Save memories to storage backend"""
        try:
            entries_data = [e.to_dict() for e in self._entries]
            self._storage.save_memory_entries(entries_data)
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")
    
    def _simple_embed(self, text: str) -> List[float]:
        """
        Simple embedding using word frequency.
        For production, replace with:
        - OpenAI embeddings
        - Sentence transformers
        - Other embedding models
        """
        # Normalize and tokenize
        words = text.lower().split()
        
        # Create a simple bag-of-words vector (very basic)
        # In production, use proper embeddings
        word_freq: Dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create a fixed-size vector using hash
        vector = [0.0] * 128
        for word, freq in word_freq.items():
            idx = hash(word) % 128
            vector[idx] += freq
        
        # Normalize
        magnitude = sum(v * v for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        
        return vector
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        if mag1 * mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)
    
    def add(
        self,
        content: str,
        source: str = "knowledge",
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Add a memory entry"""
        entry = MemoryEntry(
            id=hashlib.md5(f"{content}{datetime.now(timezone.utc)}".encode()).hexdigest()[:12],
            content=content,
            source=source,
            timestamp=datetime.now(timezone.utc),
            embedding=self._simple_embed(content),
            metadata=metadata or {}
        )
        
        self._entries.append(entry)
        
        # Trim if over limit
        if len(self._entries) > self.max_entries:
            # Remove oldest entries
            self._entries = sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:self.max_entries]
        
        self._save()
        return entry
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        min_score: float = 0.1
    ) -> List[MemorySearchResult]:
        """Search for relevant memories"""
        query_embedding = self._simple_embed(query)
        
        results = []
        for entry in self._entries:
            if source_filter and entry.source != source_filter:
                continue
            
            if entry.embedding:
                score = self._cosine_similarity(query_embedding, entry.embedding)
                if score >= min_score:
                    results.append(MemorySearchResult(
                        content=entry.content,
                        score=score,
                        source=entry.source,
                        timestamp=entry.timestamp,
                        metadata=entry.metadata
                    ))
        
        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def add_user_fact(self, fact: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a fact about the user (for personalization)"""
        return self.add(fact, source="user_fact", metadata=metadata)
    
    def add_knowledge(self, knowledge: str, metadata: Optional[Dict[str, Any]] = None):
        """Add general knowledge"""
        return self.add(knowledge, source="knowledge", metadata=metadata)
    
    def get_user_facts(self, query: str, top_k: int = 3) -> List[MemorySearchResult]:
        """Get relevant user facts"""
        return self.search(query, top_k=top_k, source_filter="user_fact")
    
    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Get relevant memories as context string"""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        
        lines = ["## Relevant Memory:"]
        for r in results:
            lines.append(f"- [{r.source}] {r.content}")
        return "\n".join(lines)
    
    def clear(self):
        """Clear all memories"""
        self._entries = []
        self._storage.clear_memory()
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        sources = {}
        for entry in self._entries:
            sources[entry.source] = sources.get(entry.source, 0) + 1
        
        return {
            "total_entries": len(self._entries),
            "by_source": sources,
            "oldest": self._entries[0].timestamp.isoformat() if self._entries else None,
            "newest": self._entries[-1].timestamp.isoformat() if self._entries else None
        }


# ──────────────────────────────────────────────────────────────────────────────
# Unified Memory Manager
# ──────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Unified interface for all memory operations.
    
    Combines:
    - ConversationMemory (short-term)
    - SemanticMemory (long-term)
    
    Storage Backend:
    - Default: Local files (JSON + pickle)
    - Future: Set MEMORY_STORAGE_BACKEND in .env to use database
    
    Provides context injection for prompts.
    """
    
    def __init__(
        self,
        storage_backend: StorageBackend = None,
        context_window: int = 20
    ):
        # Use shared storage backend for both memory types
        self._storage = storage_backend or get_storage_backend()
        
        self.conversation = ConversationMemory(
            storage_backend=self._storage,
            context_window=context_window
        )
        self.semantic = SemanticMemory(storage_backend=self._storage)
        
        logger.info(f"Memory initialized with backend: {type(self._storage).__name__}")
    
    def remember_interaction(
        self,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        capability: Optional[str] = None,
        confidence: Optional[float] = None,
        extract_facts: bool = True
    ):
        """
        Remember a user-assistant interaction.
        Optionally extract and store important facts.
        """
        # Store in conversation memory
        self.conversation.add_interaction(
            user_message=user_message,
            assistant_response=assistant_response,
            session_id=session_id,
            capability=capability,
            confidence=confidence
        )
        
        # Extract and store facts (simple heuristic)
        if extract_facts:
            self._extract_facts(user_message, assistant_response)
    
    def _extract_facts(self, user_message: str, assistant_response: str):
        """Extract important facts from interaction (simple heuristic)"""
        # Look for user preferences/facts
        preference_indicators = [
            "i like", "i prefer", "i want", "i need", "i am", "i'm",
            "my name is", "i live", "i work", "my favorite"
        ]
        
        user_lower = user_message.lower()
        for indicator in preference_indicators:
            if indicator in user_lower:
                self.semantic.add_user_fact(user_message)
                break
    
    def get_context_for_prompt(
        self,
        query: str,
        session_id: Optional[str] = None,
        include_conversation: bool = True,
        include_semantic: bool = True,
        max_conversation_messages: int = 5,
        max_semantic_results: int = 3
    ) -> str:
        """
        Get combined context from all memory sources for prompt injection.
        """
        context_parts = []
        
        # Conversation history
        if include_conversation:
            conv_context = self.conversation.get_context(
                session_id=session_id,
                max_messages=max_conversation_messages
            )
            if conv_context:
                context_parts.append("## Recent Conversation:\n" + conv_context)
        
        # Semantic memory
        if include_semantic:
            sem_context = self.semantic.get_relevant_context(
                query=query,
                top_k=max_semantic_results
            )
            if sem_context:
                context_parts.append(sem_context)
        
        return "\n\n".join(context_parts)
    
    def learn(self, content: str, source: str = "knowledge"):
        """Explicitly teach JARVIS something"""
        self.semantic.add(content, source=source)
    
    def forget_session(self, session_id: str):
        """Forget a conversation session"""
        self.conversation.clear_session(session_id)
    
    def forget_all(self):
        """Clear all memories (use with caution!)"""
        self.semantic.clear()
        logger.warning("All semantic memories cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get combined memory statistics"""
        return {
            "conversation": {
                "sessions": len(self.conversation._sessions),
                "active_session": self.conversation._active_session_id
            },
            "semantic": self.semantic.stats()
        }


# ──────────────────────────────────────────────────────────────────────────────
# Global Instance
# ──────────────────────────────────────────────────────────────────────────────

# Create global memory manager instance
memory_manager = MemoryManager()

def get_memory() -> MemoryManager:
    """Get the global memory manager"""
    return memory_manager
