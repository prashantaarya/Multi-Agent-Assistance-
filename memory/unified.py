# memory/unified.py
"""
Unified Memory Interface

Single entry point to query ALL memory systems:
- ConversationMemory (chat history)
- ContactMemory (people)
- PreferenceMemory (your preferences)
- SemanticMemory (learned knowledge)

This is what agents use to get full context about a situation.

Example:
    memory = UnifiedMemory()
    
    # Get everything relevant to an email from rahul@company.com
    context = memory.get_context_for_email(
        sender_email="rahul@company.com",
        subject="Q3 Review"
    )
    
    # Returns:
    # {
    #     "sender": Contact(name="Rahul", relationship="manager", ...),
    #     "recent_interactions": [...],
    #     "relevant_conversations": [...],
    #     "your_preferences": {"email_style": "formal", ...},
    #     "related_knowledge": ["Q3 targets were...", ...]
    # }
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Import all memory systems
from memory.contacts import get_contact_memory, Contact, ContactMemory
from memory.preferences import get_preference_memory, PreferenceMemory
from agents.memory import get_memory, ConversationMemory

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Context Models
# ──────────────────────────────────────────────────────────────────────────────

class EmailContext(BaseModel):
    """Full context for handling an email"""
    sender: Optional[Contact] = None
    sender_is_known: bool = False
    relationship: str = "unknown"
    recent_interactions: List[str] = []
    email_style: Dict[str, Any] = {}
    relevant_notes: List[str] = []
    suggested_priority: str = "normal"  # low, normal, high, urgent


class MeetingContext(BaseModel):
    """Full context for a meeting"""
    participants: List[Contact] = []
    known_participants: int = 0
    calendar_preferences: Dict[str, Any] = {}
    relevant_conversations: List[str] = []
    suggested_prep: List[str] = []


class GeneralContext(BaseModel):
    """General context for any query"""
    relevant_contacts: List[Contact] = []
    relevant_preferences: Dict[str, str] = {}
    recent_conversations: List[str] = []
    related_knowledge: List[str] = []


# ──────────────────────────────────────────────────────────────────────────────
# Unified Memory Class
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedMemory:
    """
    Unified interface to all memory systems.
    
    Provides high-level methods for getting contextual information
    that combines data from multiple memory sources.
    """
    
    def __init__(self):
        self.contacts = get_contact_memory()
        self.preferences = get_preference_memory()
        self.conversations = get_memory()
    
    # ──────────────────────────────────────────────────────────────────────────
    # Email Context
    # ──────────────────────────────────────────────────────────────────────────
    
    def get_email_context(
        self,
        sender_email: str,
        subject: Optional[str] = None,
        body_preview: Optional[str] = None
    ) -> EmailContext:
        """
        Get full context for handling an email.
        
        Combines:
        - Sender information from ContactMemory
        - Your email style preferences
        - Recent conversation history with sender
        - Priority suggestion based on relationship
        """
        context = EmailContext()
        
        # 1. Get sender info
        sender = self.contacts.find_by_email(sender_email)
        if sender:
            context.sender = sender
            context.sender_is_known = True
            context.relationship = sender.relationship
            
            # Get recent interaction notes
            if sender.notes:
                # Extract last few notes
                notes = sender.notes.split("\n")
                context.recent_interactions = [n.strip() for n in notes[-5:] if n.strip()]
        else:
            # Unknown sender - create a minimal contact
            context.sender_is_known = False
            context.relationship = "unknown"
        
        # 2. Get email style preferences based on relationship
        is_formal = context.relationship in ["unknown", "vendor", "client", "executive"]
        context.email_style = self.preferences.get_email_style(is_formal=is_formal)
        
        # 3. Determine priority
        priority_map = {
            "manager": "high",
            "executive": "urgent",
            "client": "high",
            "colleague": "normal",
            "vendor": "normal",
            "friend": "low",
            "unknown": "normal",
        }
        context.suggested_priority = priority_map.get(context.relationship, "normal")
        
        # Boost priority if subject contains urgent keywords
        if subject:
            urgent_keywords = ["urgent", "asap", "immediately", "critical", "deadline"]
            if any(kw in subject.lower() for kw in urgent_keywords):
                context.suggested_priority = "urgent"
        
        return context
    
    # ──────────────────────────────────────────────────────────────────────────
    # Meeting Context
    # ──────────────────────────────────────────────────────────────────────────
    
    def get_meeting_context(
        self,
        participant_emails: List[str],
        subject: Optional[str] = None
    ) -> MeetingContext:
        """
        Get full context for a meeting.
        
        Combines:
        - Participant info from ContactMemory
        - Your calendar preferences
        - Relevant past conversations
        - Suggested preparation items
        """
        context = MeetingContext()
        
        # 1. Get participant info
        for email in participant_emails:
            contact = self.contacts.find_by_email(email)
            if contact:
                context.participants.append(contact)
                context.known_participants += 1
            else:
                # Create placeholder for unknown participant
                context.participants.append(Contact(
                    name=email.split("@")[0].replace(".", " ").title(),
                    email=email,
                    relationship="unknown"
                ))
        
        # 2. Get calendar preferences
        context.calendar_preferences = self.preferences.get_calendar_preferences()
        
        # 3. Suggest preparation based on participants
        for p in context.participants:
            if p.relationship == "manager" and p.notes:
                context.suggested_prep.append(f"Review recent topics with {p.name}")
            if p.relationship == "client":
                context.suggested_prep.append(f"Prepare status update for {p.name}")
        
        return context
    
    # ──────────────────────────────────────────────────────────────────────────
    # General Context
    # ──────────────────────────────────────────────────────────────────────────
    
    def get_general_context(
        self,
        query: str,
        include_contacts: bool = True,
        include_conversations: bool = True,
        max_results: int = 5
    ) -> GeneralContext:
        """
        Get general context relevant to any query.
        
        Searches across all memory systems for relevant information.
        """
        context = GeneralContext()
        
        # 1. Search contacts
        if include_contacts:
            results = self.contacts.search(query, limit=max_results)
            context.relevant_contacts = [r.contact for r in results]
        
        # 2. Get recent conversations (if conversation memory supports search)
        if include_conversations and self.conversations:
            try:
                recent = self.conversations.get_context(n=max_results)
                if recent:
                    context.recent_conversations = [recent]
            except Exception as e:
                logger.debug(f"Could not get conversation context: {e}")
        
        return context
    
    # ──────────────────────────────────────────────────────────────────────────
    # Direct Access to Sub-Memories
    # ──────────────────────────────────────────────────────────────────────────
    
    def add_contact(self, **kwargs) -> Contact:
        """Convenience: Add a contact"""
        return self.contacts.add(**kwargs)
    
    def find_contact(self, email: str) -> Optional[Contact]:
        """Convenience: Find contact by email"""
        return self.contacts.find_by_email(email)
    
    def set_preference(self, category: str, key: str, value: str) -> None:
        """Convenience: Set a preference"""
        self.preferences.set(category, key, value)
    
    def get_preference(self, category: str, key: str, default: str = None) -> str:
        """Convenience: Get a preference"""
        return self.preferences.get(category, key, default)
    
    def record_interaction(self, email: str, summary: str = None) -> None:
        """Convenience: Record interaction with contact"""
        self.contacts.record_interaction(email=email, summary=summary)


# ──────────────────────────────────────────────────────────────────────────────
# Singleton instance
# ──────────────────────────────────────────────────────────────────────────────

_unified_memory: Optional[UnifiedMemory] = None


def get_unified_memory() -> UnifiedMemory:
    """Get the singleton UnifiedMemory instance"""
    global _unified_memory
    if _unified_memory is None:
        _unified_memory = UnifiedMemory()
    return _unified_memory
