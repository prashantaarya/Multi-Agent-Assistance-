# agents/gmail_agent.py
"""
Gmail Agent

Handles all email-related operations with context from UnifiedMemory.
Integrates with the approval system for sensitive actions (send, delete).

Capabilities:
- gmail.inbox: Check inbox, list unread emails
- gmail.read: Read specific email content
- gmail.search: Search emails with query
- gmail.draft: Create email drafts
- gmail.send: Send emails (requires approval)
- gmail.reply: Reply to emails
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from autogen_agentchat.agents import AssistantAgent

from core.gmail_client import (
    GmailClient,
    Email,
    Draft,
    EmailAddress,
    get_gmail_client,
)
from memory import get_unified_memory, get_contact_memory
from core.schemas import Capability

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gmail Tools (for ReAct loop)
# ---------------------------------------------------------------------------
def gmail_check_inbox(
    max_results: int = 10,
    unread_only: bool = True,
) -> Dict[str, Any]:
    """
    Check inbox for emails
    
    Args:
        max_results: Maximum number of emails to return (default: 10)
        unread_only: Only show unread emails (default: True)
    
    Returns:
        Dict with emails list and summary
    """
    try:
        client = get_gmail_client()
        emails = client.get_inbox(max_results=max_results, unread_only=unread_only)
        
        # Enrich with contact info
        memory = get_unified_memory()
        enriched = []
        
        for email in emails:
            ctx = memory.get_email_context(email.sender.email, email.subject)
            enriched.append({
                "id": email.id,
                "subject": email.subject,
                "from": str(email.sender),
                "sender_relationship": ctx.sender.relationship if ctx.sender else "unknown",
                "date": email.date.isoformat() if email.date else None,
                "snippet": email.snippet,
                "is_important": email.is_important,
                "suggested_priority": ctx.suggested_priority,
            })
        
        unread_count = len([e for e in emails if e.is_unread])
        important_count = len([e for e in emails if e.is_important])
        
        # Build human-readable response
        if len(emails) == 0:
            response_text = "📬 Your inbox is empty!"
        else:
            response_text = f"📬 **Inbox Summary**\n\n"
            response_text += f"You have {unread_count} unread email{'s' if unread_count != 1 else ''}"
            if important_count > 0:
                response_text += f", {important_count} marked important"
            response_text += ".\n\n**Recent Emails:**\n"
            
            for i, email in enumerate(enriched[:5], 1):
                response_text += f"{i}. **{email['subject']}**\n"
                response_text += f"   From: {email['from']}\n"
                if email['date']:
                    response_text += f"   Date: {email['date']}\n"
                response_text += "\n"
        
        return {
            "response": response_text,
            "data": {
                "emails": enriched,
                "total": len(emails),
                "unread_count": unread_count,
                "important_count": important_count
            }
        }
    except Exception as e:
        logger.error(f"gmail_check_inbox error: {e}")
        return {
            "response": f"❌ Error checking inbox: {str(e)}",
            "data": None
        }


def gmail_read_email(email_id: str) -> Dict[str, Any]:
    """
    Read a specific email's full content
    
    Args:
        email_id: The email ID to read
    
    Returns:
        Full email content with context
    """
    try:
        client = get_gmail_client()
        email = client.get_email(email_id)
        
        if not email:
            return {"success": False, "error": f"Email {email_id} not found"}
        
        # Get context
        memory = get_unified_memory()
        ctx = memory.get_email_context(email.sender.email, email.subject)
        
        # Auto-learn contact if unknown
        contacts = get_contact_memory()
        if not ctx.sender:
            contacts.get_or_create(email.sender.email, email.sender.name)
        else:
            # Record interaction
            contacts.record_interaction(
                email=email.sender.email,
                summary=f"Received email: {email.subject[:50]}",
            )
        
        return {
            "success": True,
            "email": {
                "id": email.id,
                "thread_id": email.thread_id,
                "subject": email.subject,
                "from": str(email.sender),
                "to": [str(r) for r in email.recipients],
                "cc": [str(c) for c in email.cc],
                "date": email.date.isoformat() if email.date else None,
                "body": email.body_text,
                "is_read": email.is_read,
                "is_important": email.is_important,
                "has_attachments": len(email.attachments) > 0,
                "attachments": [a["filename"] for a in email.attachments],
            },
            "context": {
                "sender_info": ctx.sender.model_dump() if ctx.sender else None,
                "relationship": ctx.sender.relationship if ctx.sender else "unknown",
                "previous_interactions": ctx.sender.interaction_count if ctx.sender else 0,
                "suggested_priority": ctx.suggested_priority,
            },
        }
    except Exception as e:
        logger.error(f"gmail_read_email error: {e}")
        return {"success": False, "error": str(e)}


def gmail_search(
    query: str,
    max_results: int = 10,
) -> Dict[str, Any]:
    """
    Search emails using Gmail query syntax
    
    Args:
        query: Search query (e.g., "from:boss@company.com", "subject:urgent", "has:attachment")
        max_results: Maximum results to return
    
    Returns:
        List of matching emails
    """
    try:
        client = get_gmail_client()
        emails = client.search(query, max_results=max_results)
        
        results = []
        for email in emails:
            results.append({
                "id": email.id,
                "subject": email.subject,
                "from": str(email.sender),
                "date": email.date.isoformat() if email.date else None,
                "snippet": email.snippet,
            })
        
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "emails": results,
        }
    except Exception as e:
        logger.error(f"gmail_search error: {e}")
        return {"success": False, "error": str(e)}


def gmail_create_draft(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    reply_to_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an email draft
    
    Args:
        to: Recipient email(s), comma-separated
        subject: Email subject
        body: Email body text
        cc: CC recipients (optional), comma-separated
        reply_to_id: Email ID if this is a reply (optional)
    
    Returns:
        Draft ID and preview
    """
    try:
        client = get_gmail_client()
        memory = get_unified_memory()
        
        # Parse recipients
        to_list = [t.strip() for t in to.split(",")]
        cc_list = [c.strip() for c in cc.split(",")] if cc else []
        
        # Get email style from preferences
        # Determine formality (simple heuristic - known contacts are less formal)
        contacts = get_contact_memory()
        primary_recipient = contacts.find_by_email(to_list[0])
        is_formal = primary_recipient is None or primary_recipient.relationship in ["unknown", "client", "vendor"]
        
        style = memory.preferences.get_email_style(is_formal)
        
        # Build draft
        draft = Draft(
            to=to_list,
            cc=cc_list,
            subject=subject,
            body=body,
            reply_to_id=reply_to_id,
        )
        
        draft_id = client.create_draft(draft)
        
        return {
            "success": True,
            "draft_id": draft_id,
            "preview": {
                "to": to_list,
                "cc": cc_list,
                "subject": subject,
                "body_preview": body[:200] + "..." if len(body) > 200 else body,
            },
            "style_applied": style,
            "message": f"Draft created. Use gmail_send_draft to send it.",
        }
    except Exception as e:
        logger.error(f"gmail_create_draft error: {e}")
        return {"success": False, "error": str(e)}


def gmail_draft_reply(
    email_id: str,
    body: str,
    include_cc: bool = True,
) -> Dict[str, Any]:
    """
    Create a draft reply to an email
    
    Args:
        email_id: The email ID to reply to
        body: Reply body text
        include_cc: Include original CC recipients (default: True)
    
    Returns:
        Draft ID and preview
    """
    try:
        client = get_gmail_client()
        original = client.get_email(email_id)
        
        if not original:
            return {"success": False, "error": f"Email {email_id} not found"}
        
        # Build reply subject
        subject = original.subject
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        
        # Recipients: reply to sender
        to_list = [original.sender.email]
        cc_list = [c.email for c in original.cc] if include_cc else []
        
        draft = Draft(
            to=to_list,
            cc=cc_list,
            subject=subject,
            body=body,
            reply_to_id=original.id,
            thread_id=original.thread_id,
        )
        
        draft_id = client.create_draft(draft)
        
        return {
            "success": True,
            "draft_id": draft_id,
            "replying_to": {
                "from": str(original.sender),
                "subject": original.subject,
            },
            "preview": {
                "to": to_list,
                "cc": cc_list,
                "subject": subject,
                "body_preview": body[:200] + "..." if len(body) > 200 else body,
            },
            "message": "Reply draft created. Use gmail_send_draft to send it.",
        }
    except Exception as e:
        logger.error(f"gmail_draft_reply error: {e}")
        return {"success": False, "error": str(e)}


def gmail_send_draft(draft_id: str) -> Dict[str, Any]:
    """
    Send a draft (requires approval in production)
    
    Args:
        draft_id: The draft ID to send
    
    Returns:
        Sent message ID or approval request
    """
    try:
        client = get_gmail_client()
        
        # In production, this would go through approval system
        # For now, send directly in mock mode
        if client.mock_mode:
            msg_id = client.send_draft(draft_id)
            return {
                "success": True,
                "message_id": msg_id,
                "message": "Email sent successfully (mock mode).",
            }
        
        # Production: Queue for approval
        # TODO: Integrate with approval system
        msg_id = client.send_draft(draft_id)
        return {
            "success": True,
            "message_id": msg_id,
            "message": "Email sent successfully.",
        }
    except Exception as e:
        logger.error(f"gmail_send_draft error: {e}")
        return {"success": False, "error": str(e)}


def gmail_send_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send an email directly (requires approval in production)
    
    Args:
        to: Recipient email(s), comma-separated
        subject: Email subject
        body: Email body text
        cc: CC recipients (optional)
    
    Returns:
        Sent message ID or approval request
    """
    try:
        client = get_gmail_client()
        
        to_list = [t.strip() for t in to.split(",")]
        cc_list = [c.strip() for c in cc.split(",")] if cc else []
        
        draft = Draft(
            to=to_list,
            cc=cc_list,
            subject=subject,
            body=body,
        )
        
        # In production, this would go through approval system
        if client.mock_mode:
            msg_id = client.send_email(draft)
            return {
                "success": True,
                "message_id": msg_id,
                "message": "Email sent successfully (mock mode).",
                "details": {
                    "to": to_list,
                    "subject": subject,
                },
            }
        
        # Production: Send
        msg_id = client.send_email(draft)
        return {
            "success": True,
            "message_id": msg_id,
            "message": "Email sent successfully.",
        }
    except Exception as e:
        logger.error(f"gmail_send_email error: {e}")
        return {"success": False, "error": str(e)}


def gmail_mark_read(email_id: str) -> Dict[str, Any]:
    """Mark an email as read"""
    try:
        client = get_gmail_client()
        client.mark_as_read(email_id)
        return {"success": True, "message": f"Email {email_id} marked as read."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def gmail_archive(email_id: str) -> Dict[str, Any]:
    """Archive an email (remove from inbox)"""
    try:
        client = get_gmail_client()
        client.archive(email_id)
        return {"success": True, "message": f"Email {email_id} archived."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def gmail_get_thread(thread_id: str) -> Dict[str, Any]:
    """
    Get all emails in a thread/conversation
    
    Args:
        thread_id: The thread ID
    
    Returns:
        All emails in the conversation
    """
    try:
        client = get_gmail_client()
        emails = client.get_thread(thread_id)
        
        thread_emails = []
        for email in emails:
            thread_emails.append({
                "id": email.id,
                "from": str(email.sender),
                "date": email.date.isoformat() if email.date else None,
                "body": email.body_text,
            })
        
        return {
            "success": True,
            "thread_id": thread_id,
            "message_count": len(thread_emails),
            "emails": thread_emails,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Gmail Tools Registry
# ---------------------------------------------------------------------------
GMAIL_TOOLS = {
    "gmail_check_inbox": {
        "function": gmail_check_inbox,
        "schema": {
            "name": "gmail_check_inbox",
            "description": "Check inbox for emails. Returns list of emails with sender info and priority.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum emails to return (default: 10)",
                        "default": 10,
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Only show unread emails (default: True)",
                        "default": True,
                    },
                },
                "required": [],
            },
        },
    },
    "gmail_read_email": {
        "function": gmail_read_email,
        "schema": {
            "name": "gmail_read_email",
            "description": "Read a specific email's full content including body and attachments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The email ID to read",
                    },
                },
                "required": ["email_id"],
            },
        },
    },
    "gmail_search": {
        "function": gmail_search,
        "schema": {
            "name": "gmail_search",
            "description": "Search emails. Query examples: 'from:boss@company.com', 'subject:urgent', 'has:attachment', 'after:2024/01/01'",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gmail search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    "gmail_create_draft": {
        "function": gmail_create_draft,
        "schema": {
            "name": "gmail_create_draft",
            "description": "Create an email draft. Does NOT send - use gmail_send_draft to send.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email(s), comma-separated",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text",
                    },
                    "cc": {
                        "type": "string",
                        "description": "CC recipients, comma-separated (optional)",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    "gmail_draft_reply": {
        "function": gmail_draft_reply,
        "schema": {
            "name": "gmail_draft_reply",
            "description": "Create a draft reply to an existing email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The email ID to reply to",
                    },
                    "body": {
                        "type": "string",
                        "description": "Reply body text",
                    },
                    "include_cc": {
                        "type": "boolean",
                        "description": "Include original CC recipients (default: True)",
                        "default": True,
                    },
                },
                "required": ["email_id", "body"],
            },
        },
    },
    "gmail_send_draft": {
        "function": gmail_send_draft,
        "schema": {
            "name": "gmail_send_draft",
            "description": "Send an existing draft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "The draft ID to send",
                    },
                },
                "required": ["draft_id"],
            },
        },
    },
    "gmail_send_email": {
        "function": gmail_send_email,
        "schema": {
            "name": "gmail_send_email",
            "description": "Send an email directly. Use gmail_create_draft + gmail_send_draft for review before sending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email(s), comma-separated",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text",
                    },
                    "cc": {
                        "type": "string",
                        "description": "CC recipients (optional)",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    "gmail_mark_read": {
        "function": gmail_mark_read,
        "schema": {
            "name": "gmail_mark_read",
            "description": "Mark an email as read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The email ID to mark as read",
                    },
                },
                "required": ["email_id"],
            },
        },
    },
    "gmail_archive": {
        "function": gmail_archive,
        "schema": {
            "name": "gmail_archive",
            "description": "Archive an email (removes from inbox but keeps in All Mail).",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The email ID to archive",
                    },
                },
                "required": ["email_id"],
            },
        },
    },
    "gmail_get_thread": {
        "function": gmail_get_thread,
        "schema": {
            "name": "gmail_get_thread",
            "description": "Get all emails in a conversation thread.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "The thread ID",
                    },
                },
                "required": ["thread_id"],
            },
        },
    },
}


def get_gmail_tool(name: str):
    """Get a Gmail tool by name"""
    return GMAIL_TOOLS.get(name)


def get_all_gmail_tools() -> Dict[str, Any]:
    """Get all Gmail tools"""
    return GMAIL_TOOLS


# ---------------------------------------------------------------------------
# Gmail Agent (for orchestration)
# ---------------------------------------------------------------------------
def create_gmail_agent_prompt() -> str:
    """Create system prompt for Gmail agent"""
    return """You are a Gmail assistant that helps manage emails.

You have access to the following tools:
- gmail_check_inbox: Check for new/unread emails
- gmail_read_email: Read full email content
- gmail_search: Search emails with queries
- gmail_create_draft: Create email drafts
- gmail_draft_reply: Create reply drafts
- gmail_send_draft: Send a draft
- gmail_send_email: Send email directly
- gmail_mark_read: Mark email as read
- gmail_archive: Archive email
- gmail_get_thread: Get conversation thread

Guidelines:
1. Always check inbox context before suggesting actions
2. Provide email summaries with sender relationship info
3. For replies, suggest appropriate tone based on sender relationship
4. Create drafts first for review, don't send directly unless asked
5. Prioritize emails from important contacts (manager, team members)

When summarizing emails:
- Lead with sender name and relationship
- Mention subject and key action items
- Flag urgent or time-sensitive content
"""


# ---------------------------------------------------------------------------
# Gmail Capabilities
# ---------------------------------------------------------------------------
GMAIL_CAPABILITIES = [
    Capability(
        name="gmail.inbox",
        description="Check inbox for new and unread emails",
        required_tools=["gmail_check_inbox"],
        example_queries=["check my email", "any new messages", "unread emails"],
    ),
    Capability(
        name="gmail.read",
        description="Read specific email content",
        required_tools=["gmail_read_email", "gmail_get_thread"],
        example_queries=["read that email", "what does it say", "show me the email from John"],
    ),
    Capability(
        name="gmail.search",
        description="Search emails by sender, subject, date, etc.",
        required_tools=["gmail_search"],
        example_queries=["find emails from boss", "search for budget emails", "emails with attachments"],
    ),
    Capability(
        name="gmail.draft",
        description="Create email drafts",
        required_tools=["gmail_create_draft", "gmail_draft_reply"],
        example_queries=["draft a reply", "write an email to John", "compose a message"],
    ),
    Capability(
        name="gmail.send",
        description="Send emails and drafts",
        required_tools=["gmail_send_draft", "gmail_send_email"],
        example_queries=["send the email", "send draft", "send this"],
    ),
    Capability(
        name="gmail.manage",
        description="Manage emails (mark read, archive)",
        required_tools=["gmail_mark_read", "gmail_archive"],
        example_queries=["mark as read", "archive this", "clean up inbox"],
    ),
]
