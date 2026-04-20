# memory/contacts.py
"""
Contact Memory System

Stores information about people in your life:
- Who they are (name, email, phone)
- Your relationship (manager, friend, vendor, etc.)
- Interaction history
- Notes and sentiment

This powers the "second brain" ability to:
- "Who is this email from?" → Full context about the person
- "When did I last talk to Rahul?" → Interaction history
- "Find emails from my manager" → Relationship-aware search
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from core.database import get_cursor, fetch_one, fetch_all, execute

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

class Contact(BaseModel):
    """A person in your network"""
    id: Optional[int] = None
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    relationship: str = "unknown"  # manager, colleague, friend, vendor, family, etc.
    organization: Optional[str] = None
    notes: Optional[str] = None
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    sentiment: str = "neutral"  # positive, neutral, needs-attention, negative
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ContactSearchResult(BaseModel):
    """Search result with relevance info"""
    contact: Contact
    match_reason: str  # "email_exact", "name_partial", "organization", etc.
    relevance_score: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Contact Memory Class
# ──────────────────────────────────────────────────────────────────────────────

class ContactMemory:
    """
    Manages contact information with persistence.
    
    Usage:
        contacts = ContactMemory()
        
        # Add a contact
        contact = contacts.add(name="Rahul", email="rahul@company.com", relationship="manager")
        
        # Find by email
        contact = contacts.find_by_email("rahul@company.com")
        
        # Search
        results = contacts.search("rahul")
        
        # Record interaction
        contacts.record_interaction("rahul@company.com", "Discussed Q3 review")
    """
    
    def __init__(self):
        # Database is initialized lazily via get_cursor()
        pass
    
    def add(
        self,
        name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        relationship: str = "unknown",
        organization: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Contact:
        """
        Add a new contact or update existing (by email).
        """
        now = datetime.now(timezone.utc).isoformat()
        
        # Normalize email
        if email:
            email = email.lower().strip()
        
        # Check if contact exists by email
        if email:
            existing = self.find_by_email(email)
            if existing:
                # Update existing contact
                return self.update(
                    existing.id,
                    name=name,
                    phone=phone,
                    relationship=relationship,
                    organization=organization,
                    notes=notes
                )
        
        # Insert new contact
        with get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO contacts (name, email, phone, relationship, organization, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, email, phone, relationship, organization, notes, now, now))
            
            contact_id = cursor.lastrowid
        
        logger.info(f"Added contact: {name} ({email})")
        return self.get_by_id(contact_id)
    
    def get_by_id(self, contact_id: int) -> Optional[Contact]:
        """Get contact by ID"""
        row = fetch_one("SELECT * FROM contacts WHERE id = ?", (contact_id,))
        return Contact(**row) if row else None
    
    def find_by_email(self, email: str) -> Optional[Contact]:
        """Find contact by email address"""
        row = fetch_one("SELECT * FROM contacts WHERE email = ?", (email.lower().strip(),))
        return Contact(**row) if row else None
    
    def find_by_phone(self, phone: str) -> Optional[Contact]:
        """Find contact by phone number"""
        # Normalize phone (remove spaces, dashes)
        normalized = ''.join(c for c in phone if c.isdigit() or c == '+')
        row = fetch_one("SELECT * FROM contacts WHERE phone LIKE ?", (f"%{normalized[-10:]}%",))
        return Contact(**row) if row else None
    
    def search(self, query: str, limit: int = 10) -> List[ContactSearchResult]:
        """
        Search contacts by name, email, organization, or notes.
        Returns ranked results.
        """
        query_lower = query.lower().strip()
        results = []
        
        # Exact email match (highest priority)
        if "@" in query:
            contact = self.find_by_email(query)
            if contact:
                results.append(ContactSearchResult(
                    contact=contact,
                    match_reason="email_exact",
                    relevance_score=1.0
                ))
                return results
        
        # Search by name, email, organization
        rows = fetch_all("""
            SELECT *, 
                CASE 
                    WHEN LOWER(name) = ? THEN 1.0
                    WHEN LOWER(name) LIKE ? THEN 0.9
                    WHEN LOWER(email) LIKE ? THEN 0.8
                    WHEN LOWER(organization) LIKE ? THEN 0.7
                    WHEN LOWER(notes) LIKE ? THEN 0.5
                    ELSE 0.3
                END as relevance
            FROM contacts
            WHERE LOWER(name) LIKE ? 
               OR LOWER(email) LIKE ? 
               OR LOWER(organization) LIKE ?
               OR LOWER(notes) LIKE ?
            ORDER BY relevance DESC, interaction_count DESC
            LIMIT ?
        """, (
            query_lower,
            f"%{query_lower}%", f"%{query_lower}%", f"%{query_lower}%", f"%{query_lower}%",
            f"%{query_lower}%", f"%{query_lower}%", f"%{query_lower}%", f"%{query_lower}%",
            limit
        ))
        
        for row in rows:
            relevance = row.pop("relevance", 0.5)
            contact = Contact(**row)
            
            # Determine match reason
            if query_lower in (contact.name or "").lower():
                reason = "name_match"
            elif query_lower in (contact.email or "").lower():
                reason = "email_match"
            elif query_lower in (contact.organization or "").lower():
                reason = "organization_match"
            else:
                reason = "notes_match"
            
            results.append(ContactSearchResult(
                contact=contact,
                match_reason=reason,
                relevance_score=relevance
            ))
        
        return results
    
    def find_by_relationship(self, relationship: str) -> List[Contact]:
        """Find all contacts with a specific relationship"""
        rows = fetch_all(
            "SELECT * FROM contacts WHERE LOWER(relationship) = ? ORDER BY interaction_count DESC",
            (relationship.lower(),)
        )
        return [Contact(**row) for row in rows]
    
    def update(
        self,
        contact_id: int,
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        relationship: Optional[str] = None,
        organization: Optional[str] = None,
        notes: Optional[str] = None,
        sentiment: Optional[str] = None
    ) -> Optional[Contact]:
        """Update an existing contact"""
        existing = self.get_by_id(contact_id)
        if not existing:
            return None
        
        now = datetime.now(timezone.utc).isoformat()
        
        with get_cursor() as cursor:
            cursor.execute("""
                UPDATE contacts SET
                    name = COALESCE(?, name),
                    email = COALESCE(?, email),
                    phone = COALESCE(?, phone),
                    relationship = COALESCE(?, relationship),
                    organization = COALESCE(?, organization),
                    notes = COALESCE(?, notes),
                    sentiment = COALESCE(?, sentiment),
                    updated_at = ?
                WHERE id = ?
            """, (name, email, phone, relationship, organization, notes, sentiment, now, contact_id))
        
        logger.info(f"Updated contact: {contact_id}")
        return self.get_by_id(contact_id)
    
    def record_interaction(
        self,
        email: Optional[str] = None,
        contact_id: Optional[int] = None,
        summary: Optional[str] = None
    ) -> Optional[Contact]:
        """
        Record that you interacted with this contact.
        Updates interaction_count and last_interaction.
        Optionally appends to notes.
        """
        # Find the contact
        contact = None
        if contact_id:
            contact = self.get_by_id(contact_id)
        elif email:
            contact = self.find_by_email(email)
        
        if not contact:
            logger.warning(f"Contact not found for interaction: {email or contact_id}")
            return None
        
        now = datetime.now(timezone.utc).isoformat()
        new_count = contact.interaction_count + 1
        
        # Optionally append summary to notes
        new_notes = contact.notes
        if summary:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            note_entry = f"\n[{timestamp}] {summary}"
            new_notes = (contact.notes or "") + note_entry
        
        with get_cursor() as cursor:
            cursor.execute("""
                UPDATE contacts SET
                    interaction_count = ?,
                    last_interaction = ?,
                    notes = ?,
                    updated_at = ?
                WHERE id = ?
            """, (new_count, now, new_notes, now, contact.id))
        
        logger.info(f"Recorded interaction with: {contact.name}")
        return self.get_by_id(contact.id)
    
    def get_recent_contacts(self, limit: int = 10) -> List[Contact]:
        """Get contacts you've interacted with recently"""
        rows = fetch_all("""
            SELECT * FROM contacts 
            WHERE last_interaction IS NOT NULL
            ORDER BY last_interaction DESC
            LIMIT ?
        """, (limit,))
        return [Contact(**row) for row in rows]
    
    def get_all(self, limit: int = 100) -> List[Contact]:
        """Get all contacts"""
        rows = fetch_all("SELECT * FROM contacts ORDER BY name LIMIT ?", (limit,))
        return [Contact(**row) for row in rows]
    
    def delete(self, contact_id: int) -> bool:
        """Delete a contact"""
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info(f"Deleted contact: {contact_id}")
        return deleted
    
    def get_or_create(
        self,
        email: str,
        name: Optional[str] = None,
        **kwargs
    ) -> Contact:
        """
        Get existing contact by email, or create if not exists.
        Useful for processing emails from unknown senders.
        """
        existing = self.find_by_email(email)
        if existing:
            return existing
        
        # Extract name from email if not provided
        if not name:
            name = email.split("@")[0].replace(".", " ").title()
        
        return self.add(name=name, email=email, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Singleton instance
# ──────────────────────────────────────────────────────────────────────────────

_contact_memory: Optional[ContactMemory] = None


def get_contact_memory() -> ContactMemory:
    """Get the singleton ContactMemory instance"""
    global _contact_memory
    if _contact_memory is None:
        _contact_memory = ContactMemory()
    return _contact_memory
