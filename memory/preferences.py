# memory/preferences.py
"""
Preference Memory System

Stores learned preferences about YOU:
- Communication style preferences
- Scheduling preferences  
- Work habits
- Contact-specific preferences

This powers the "second brain" ability to:
- Draft emails in YOUR style
- Schedule meetings at times YOU prefer
- Prioritize notifications based on YOUR patterns
- Learn from your approvals/rejections

Preferences are stored as key-value pairs with:
- Category: grouping (e.g., "email", "calendar", "communication")
- Key: specific preference (e.g., "preferred_greeting", "meeting_buffer_minutes")
- Value: the preference value
- Confidence: how sure we are (0.0-1.0), increases with more evidence
- Source: where we learned this (e.g., "user_explicit", "learned_from_emails")
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from core.database import get_cursor, fetch_one, fetch_all

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

class Preference(BaseModel):
    """A single preference"""
    id: Optional[int] = None
    category: str
    key: str
    value: str
    confidence: float = 0.5  # 0.0 = guess, 1.0 = explicitly set
    source: Optional[str] = None  # "user_explicit", "learned", "default"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def value_as_bool(self) -> bool:
        """Parse value as boolean"""
        return self.value.lower() in ("true", "yes", "1", "on")
    
    @property
    def value_as_int(self) -> int:
        """Parse value as integer"""
        return int(self.value)
    
    @property
    def value_as_list(self) -> List[str]:
        """Parse value as JSON list"""
        try:
            return json.loads(self.value)
        except:
            return [self.value]


# ──────────────────────────────────────────────────────────────────────────────
# Default Preferences (used when nothing learned yet)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PREFERENCES = {
    # Email preferences
    "email.greeting_formal": "Hello",
    "email.greeting_casual": "Hi",
    "email.closing_formal": "Best regards,",
    "email.closing_casual": "Thanks!",
    "email.signature": "",  # User should set this
    "email.max_length": "short",  # short, medium, long
    "email.use_bullet_points": "true",
    
    # Calendar preferences
    "calendar.preferred_meeting_times": '["10:00", "14:00", "16:00"]',
    "calendar.avoid_times": '["12:00", "13:00"]',  # Lunch
    "calendar.meeting_buffer_minutes": "15",
    "calendar.default_meeting_duration": "30",
    "calendar.prefer_morning": "true",
    
    # Communication preferences
    "communication.response_urgency_hours": "4",
    "communication.auto_archive_newsletters": "true",
    "communication.notify_channels": '["telegram"]',
    
    # Work preferences
    "work.timezone": "Asia/Kolkata",
    "work.work_start_hour": "9",
    "work.work_end_hour": "18",
    "work.work_days": '["mon", "tue", "wed", "thu", "fri"]',
}


# ──────────────────────────────────────────────────────────────────────────────
# Preference Memory Class
# ──────────────────────────────────────────────────────────────────────────────

class PreferenceMemory:
    """
    Manages user preferences with persistence and learning.
    
    Usage:
        prefs = PreferenceMemory()
        
        # Get a preference (returns default if not set)
        greeting = prefs.get("email", "greeting_formal")
        
        # Set explicitly (confidence = 1.0)
        prefs.set("email", "signature", "Regards, John", source="user_explicit")
        
        # Learn from behavior (increases confidence gradually)
        prefs.learn("email", "use_bullet_points", "true", evidence_strength=0.3)
        
        # Get all email preferences
        email_prefs = prefs.get_category("email")
    """
    
    def __init__(self):
        pass
    
    def get(
        self,
        category: str,
        key: str,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a preference value.
        Returns stored value, or default from DEFAULT_PREFERENCES, or provided default.
        """
        row = fetch_one(
            "SELECT value FROM preferences WHERE category = ? AND key = ?",
            (category, key)
        )
        
        if row:
            return row["value"]
        
        # Check defaults
        full_key = f"{category}.{key}"
        if full_key in DEFAULT_PREFERENCES:
            return DEFAULT_PREFERENCES[full_key]
        
        return default
    
    def get_full(self, category: str, key: str) -> Optional[Preference]:
        """Get full preference object including confidence and source"""
        row = fetch_one(
            "SELECT * FROM preferences WHERE category = ? AND key = ?",
            (category, key)
        )
        return Preference(**row) if row else None
    
    def set(
        self,
        category: str,
        key: str,
        value: str,
        source: str = "user_explicit",
        confidence: float = 1.0
    ) -> Preference:
        """
        Set a preference explicitly.
        Use this when user directly tells you their preference.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        with get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO preferences (category, key, value, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(category, key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    updated_at = excluded.updated_at
            """, (category, key, value, confidence, source, now, now))
        
        logger.info(f"Set preference: {category}.{key} = {value} (confidence: {confidence})")
        return self.get_full(category, key)
    
    def learn(
        self,
        category: str,
        key: str,
        value: str,
        evidence_strength: float = 0.1
    ) -> Preference:
        """
        Learn a preference from observed behavior.
        
        Confidence increases gradually with more evidence.
        If the same value is observed repeatedly, confidence grows.
        If a different value is observed, confidence adjusts.
        
        Args:
            category: Preference category
            key: Preference key
            value: Observed value
            evidence_strength: How much this observation should affect confidence (0.0-1.0)
        """
        existing = self.get_full(category, key)
        now = datetime.now(timezone.utc).isoformat()
        
        if existing:
            if existing.value == value:
                # Same value observed again - increase confidence
                new_confidence = min(1.0, existing.confidence + evidence_strength * (1 - existing.confidence))
            else:
                # Different value - decrease confidence, maybe switch
                new_confidence = existing.confidence - evidence_strength
                if new_confidence < 0.3:
                    # Switch to new value
                    value = value
                    new_confidence = evidence_strength
                else:
                    # Keep old value with reduced confidence
                    value = existing.value
                    new_confidence = max(0.1, new_confidence)
            
            with get_cursor() as cursor:
                cursor.execute("""
                    UPDATE preferences SET
                        value = ?,
                        confidence = ?,
                        source = 'learned',
                        updated_at = ?
                    WHERE category = ? AND key = ?
                """, (value, new_confidence, now, category, key))
        else:
            # New preference - start with low confidence
            with get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO preferences (category, key, value, confidence, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'learned', ?, ?)
                """, (category, key, value, evidence_strength, now, now))
        
        logger.debug(f"Learned preference: {category}.{key} = {value}")
        return self.get_full(category, key)
    
    def get_category(self, category: str) -> Dict[str, str]:
        """Get all preferences in a category as a dict"""
        rows = fetch_all(
            "SELECT key, value FROM preferences WHERE category = ?",
            (category,)
        )
        
        result = {}
        
        # Start with defaults
        for full_key, default_value in DEFAULT_PREFERENCES.items():
            if full_key.startswith(f"{category}."):
                key = full_key.split(".", 1)[1]
                result[key] = default_value
        
        # Override with stored values
        for row in rows:
            result[row["key"]] = row["value"]
        
        return result
    
    def get_all(self) -> List[Preference]:
        """Get all stored preferences"""
        rows = fetch_all("SELECT * FROM preferences ORDER BY category, key")
        return [Preference(**row) for row in rows]
    
    def delete(self, category: str, key: str) -> bool:
        """Delete a preference (reverts to default)"""
        with get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM preferences WHERE category = ? AND key = ?",
                (category, key)
            )
            return cursor.rowcount > 0
    
    def reset_category(self, category: str) -> int:
        """Reset all preferences in a category to defaults"""
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM preferences WHERE category = ?", (category,))
            return cursor.rowcount
    
    # ──────────────────────────────────────────────────────────────────────────
    # Convenience methods for common preferences
    # ──────────────────────────────────────────────────────────────────────────
    
    def get_email_style(self, is_formal: bool = True) -> Dict[str, str]:
        """Get email writing style preferences"""
        suffix = "formal" if is_formal else "casual"
        return {
            "greeting": self.get("email", f"greeting_{suffix}", "Hi"),
            "closing": self.get("email", f"closing_{suffix}", "Thanks!"),
            "signature": self.get("email", "signature", ""),
            "max_length": self.get("email", "max_length", "short"),
            "use_bullets": self.get("email", "use_bullet_points", "true") == "true",
        }
    
    def get_calendar_preferences(self) -> Dict[str, Any]:
        """Get calendar/scheduling preferences"""
        return {
            "preferred_times": json.loads(self.get("calendar", "preferred_meeting_times", "[]")),
            "avoid_times": json.loads(self.get("calendar", "avoid_times", "[]")),
            "buffer_minutes": int(self.get("calendar", "meeting_buffer_minutes", "15")),
            "default_duration": int(self.get("calendar", "default_meeting_duration", "30")),
            "prefer_morning": self.get("calendar", "prefer_morning", "true") == "true",
        }
    
    def get_work_hours(self) -> Dict[str, Any]:
        """Get work schedule preferences"""
        return {
            "timezone": self.get("work", "timezone", "UTC"),
            "start_hour": int(self.get("work", "work_start_hour", "9")),
            "end_hour": int(self.get("work", "work_end_hour", "18")),
            "work_days": json.loads(self.get("work", "work_days", '["mon","tue","wed","thu","fri"]')),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Singleton instance
# ──────────────────────────────────────────────────────────────────────────────

_preference_memory: Optional[PreferenceMemory] = None


def get_preference_memory() -> PreferenceMemory:
    """Get the singleton PreferenceMemory instance"""
    global _preference_memory
    if _preference_memory is None:
        _preference_memory = PreferenceMemory()
    return _preference_memory
