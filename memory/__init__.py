# memory/__init__.py
"""
JARVIS Memory System

The "Second Brain" memory layer with multiple specialized stores:

- contacts.py    → ContactMemory: People in your network
- preferences.py → PreferenceMemory: Your learned preferences  
- unified.py     → UnifiedMemory: Combined interface to all memories

Usage:
    from memory import get_unified_memory, get_contact_memory, get_preference_memory
    
    # Unified (recommended - combines all)
    memory = get_unified_memory()
    context = memory.get_email_context(sender_email="rahul@company.com")
    
    # Direct access
    contacts = get_contact_memory()
    contacts.add(name="Rahul", email="rahul@company.com", relationship="manager")
    
    prefs = get_preference_memory()
    prefs.set("email", "signature", "Regards, John")
"""

from memory.contacts import (
    ContactMemory,
    Contact,
    ContactSearchResult,
    get_contact_memory,
)

from memory.preferences import (
    PreferenceMemory,
    Preference,
    get_preference_memory,
)

from memory.unified import (
    UnifiedMemory,
    EmailContext,
    MeetingContext,
    GeneralContext,
    get_unified_memory,
)

# Re-export existing memory from agents/memory.py
from agents.memory import (
    ConversationMemory,
    SemanticMemory,
    MemoryMessage,
    ConversationSession,
    get_memory,
)

__all__ = [
    # Contact memory
    "ContactMemory",
    "Contact", 
    "ContactSearchResult",
    "get_contact_memory",
    
    # Preference memory
    "PreferenceMemory",
    "Preference",
    "get_preference_memory",
    
    # Unified memory
    "UnifiedMemory",
    "EmailContext",
    "MeetingContext", 
    "GeneralContext",
    "get_unified_memory",
    
    # Conversation memory (existing)
    "ConversationMemory",
    "SemanticMemory",
    "MemoryMessage",
    "ConversationSession",
    "get_memory",
]
