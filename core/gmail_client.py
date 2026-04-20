# core/gmail_client.py
"""
Gmail API Client

Handles OAuth 2.0 authentication and Gmail API operations.
Local-first: Can run in mock mode without credentials for testing.

Production Setup:
1. Go to Google Cloud Console → APIs & Services → Credentials
2. Create OAuth 2.0 Client ID (Desktop App)
3. Download credentials.json to project root
4. Set GMAIL_CREDENTIALS_PATH env var (or use default)

Scopes Used:
- gmail.readonly: Read emails
- gmail.send: Send emails
- gmail.compose: Create drafts
- gmail.modify: Mark read/unread, labels
"""

import os
import json
import base64
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", "data/gmail_token.json")
MOCK_MODE = os.getenv("GMAIL_MOCK_MODE", "true").lower() == "true"

# Gmail API Scopes
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class EmailAddress:
    """Parsed email address"""
    email: str
    name: Optional[str] = None
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email
    
    @classmethod
    def parse(cls, raw: str) -> "EmailAddress":
        """Parse 'Name <email>' or plain email"""
        if "<" in raw and ">" in raw:
            name = raw.split("<")[0].strip().strip('"')
            email = raw.split("<")[1].split(">")[0].strip()
            return cls(email=email, name=name if name else None)
        return cls(email=raw.strip())


@dataclass
class Email:
    """Represents an email message"""
    id: str
    thread_id: str
    subject: str
    sender: EmailAddress
    recipients: List[EmailAddress]
    cc: List[EmailAddress] = field(default_factory=list)
    date: Optional[datetime] = None
    snippet: str = ""
    body_text: str = ""
    body_html: str = ""
    labels: List[str] = field(default_factory=list)
    is_read: bool = False
    is_important: bool = False
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def is_unread(self) -> bool:
        return not self.is_read
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "from": str(self.sender),
            "to": [str(r) for r in self.recipients],
            "cc": [str(c) for c in self.cc],
            "date": self.date.isoformat() if self.date else None,
            "snippet": self.snippet,
            "body_text": self.body_text[:500] + "..." if len(self.body_text) > 500 else self.body_text,
            "labels": self.labels,
            "is_read": self.is_read,
            "is_important": self.is_important,
            "has_attachments": len(self.attachments) > 0,
        }


@dataclass
class Draft:
    """Represents an email draft"""
    id: Optional[str] = None
    to: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    reply_to_id: Optional[str] = None
    thread_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Mock Data for Testing (when GMAIL_MOCK_MODE=true)
# ---------------------------------------------------------------------------
MOCK_EMAILS = [
    Email(
        id="mock_001",
        thread_id="thread_001",
        subject="Q1 Budget Review Meeting",
        sender=EmailAddress("rahul.sharma@company.com", "Rahul Sharma"),
        recipients=[EmailAddress("you@company.com", "You")],
        date=datetime.now(timezone.utc),
        snippet="Hi, can we schedule a meeting to review the Q1 budget?",
        body_text="Hi,\n\nCan we schedule a meeting to review the Q1 budget? I'd like to discuss the variance in the marketing spend.\n\nPlease let me know your availability for tomorrow.\n\nThanks,\nRahul",
        labels=["INBOX", "UNREAD", "IMPORTANT"],
        is_read=False,
        is_important=True,
    ),
    Email(
        id="mock_002",
        thread_id="thread_002",
        subject="Project Alpha Update",
        sender=EmailAddress("priya.patel@company.com", "Priya Patel"),
        recipients=[EmailAddress("you@company.com", "You")],
        date=datetime.now(timezone.utc),
        snippet="Quick update on Project Alpha - we're on track for the deadline.",
        body_text="Hi Team,\n\nQuick update on Project Alpha:\n- Backend API: 90% complete\n- Frontend: 85% complete\n- Testing: Started\n\nWe're on track for the March 25th deadline.\n\nBest,\nPriya",
        labels=["INBOX", "UNREAD"],
        is_read=False,
        is_important=False,
    ),
    Email(
        id="mock_003",
        thread_id="thread_003",
        subject="Team Lunch Friday",
        sender=EmailAddress("amit.kumar@company.com", "Amit Kumar"),
        recipients=[EmailAddress("you@company.com", "You")],
        cc=[EmailAddress("team@company.com", "Team")],
        date=datetime.now(timezone.utc),
        snippet="Hey everyone! Let's do a team lunch this Friday.",
        body_text="Hey everyone!\n\nLet's do a team lunch this Friday at 12:30 PM. I'm thinking we could try that new Thai place near the office.\n\nLet me know if you're in!\n\nCheers,\nAmit",
        labels=["INBOX"],
        is_read=True,
        is_important=False,
    ),
]


# ---------------------------------------------------------------------------
# Gmail Client
# ---------------------------------------------------------------------------
class GmailClient:
    """
    Gmail API Client with OAuth 2.0
    
    Supports mock mode for testing without credentials.
    """
    
    def __init__(self, mock_mode: Optional[bool] = None):
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE
        self._service = None
        self._user_email: Optional[str] = None
        
        if not self.mock_mode:
            self._init_gmail_service()
        else:
            logger.info("Gmail client running in MOCK MODE")
            self._user_email = "you@company.com"
    
    def _init_gmail_service(self):
        """Initialize Gmail API service with OAuth"""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            logger.error("Google API libraries not installed. Run: pip install google-auth-oauthlib google-api-python-client")
            raise ImportError("Install: pip install google-auth-oauthlib google-api-python-client")
        
        creds = None
        token_path = Path(TOKEN_PATH)
        
        # Load existing token
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        
        # Refresh or get new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not Path(CREDENTIALS_PATH).exists():
                    raise FileNotFoundError(
                        f"Gmail credentials not found at {CREDENTIALS_PATH}. "
                        "Download from Google Cloud Console or set GMAIL_MOCK_MODE=true"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save token
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, "w") as f:
                f.write(creds.to_json())
        
        self._service = build("gmail", "v1", credentials=creds)
        
        # Get user's email
        profile = self._service.users().getProfile(userId="me").execute()
        self._user_email = profile.get("emailAddress")
        logger.info(f"Gmail client initialized for {self._user_email}")
    
    @property
    def user_email(self) -> str:
        return self._user_email or "unknown@email.com"
    
    @property
    def is_authenticated(self) -> bool:
        return self._service is not None or self.mock_mode
    
    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------
    def get_inbox(
        self,
        max_results: int = 20,
        unread_only: bool = False,
        query: Optional[str] = None,
    ) -> List[Email]:
        """
        Fetch emails from inbox
        
        Args:
            max_results: Maximum emails to return
            unread_only: Only return unread emails
            query: Gmail search query (e.g., "from:boss@company.com")
        """
        if self.mock_mode:
            emails = MOCK_EMAILS.copy()
            if unread_only:
                emails = [e for e in emails if e.is_unread]
            if query:
                query_lower = query.lower()
                emails = [e for e in emails if 
                         query_lower in e.subject.lower() or 
                         query_lower in e.sender.email.lower() or
                         query_lower in e.body_text.lower()]
            return emails[:max_results]
        
        # Build query
        q_parts = ["in:inbox"]
        if unread_only:
            q_parts.append("is:unread")
        if query:
            q_parts.append(query)
        
        results = self._service.users().messages().list(
            userId="me",
            q=" ".join(q_parts),
            maxResults=max_results,
        ).execute()
        
        messages = results.get("messages", [])
        return [self._fetch_email(msg["id"]) for msg in messages]
    
    def get_email(self, email_id: str) -> Optional[Email]:
        """Fetch a specific email by ID"""
        if self.mock_mode:
            return next((e for e in MOCK_EMAILS if e.id == email_id), None)
        
        return self._fetch_email(email_id)
    
    def _fetch_email(self, email_id: str) -> Email:
        """Fetch and parse a single email"""
        msg = self._service.users().messages().get(
            userId="me",
            id=email_id,
            format="full",
        ).execute()
        
        headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
        
        # Parse body
        body_text, body_html = self._extract_body(msg["payload"])
        
        # Parse attachments
        attachments = self._extract_attachments(msg["payload"])
        
        # Parse date
        date = None
        if "date" in headers:
            try:
                from email.utils import parsedate_to_datetime
                date = parsedate_to_datetime(headers["date"])
            except:
                pass
        
        labels = msg.get("labelIds", [])
        
        return Email(
            id=msg["id"],
            thread_id=msg["threadId"],
            subject=headers.get("subject", "(No Subject)"),
            sender=EmailAddress.parse(headers.get("from", "")),
            recipients=[EmailAddress.parse(r.strip()) for r in headers.get("to", "").split(",") if r.strip()],
            cc=[EmailAddress.parse(c.strip()) for c in headers.get("cc", "").split(",") if c.strip()],
            date=date,
            snippet=msg.get("snippet", ""),
            body_text=body_text,
            body_html=body_html,
            labels=labels,
            is_read="UNREAD" not in labels,
            is_important="IMPORTANT" in labels,
            attachments=attachments,
        )
    
    def _extract_body(self, payload: Dict) -> tuple[str, str]:
        """Extract text and HTML body from email payload"""
        text_body = ""
        html_body = ""
        
        def extract_parts(part):
            nonlocal text_body, html_body
            mime_type = part.get("mimeType", "")
            
            if mime_type == "text/plain" and "data" in part.get("body", {}):
                text_body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
            elif mime_type == "text/html" and "data" in part.get("body", {}):
                html_body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
            elif "parts" in part:
                for p in part["parts"]:
                    extract_parts(p)
        
        extract_parts(payload)
        return text_body, html_body
    
    def _extract_attachments(self, payload: Dict) -> List[Dict[str, Any]]:
        """Extract attachment metadata"""
        attachments = []
        
        def find_attachments(part):
            if part.get("filename"):
                attachments.append({
                    "filename": part["filename"],
                    "mime_type": part.get("mimeType"),
                    "size": part.get("body", {}).get("size", 0),
                    "attachment_id": part.get("body", {}).get("attachmentId"),
                })
            for p in part.get("parts", []):
                find_attachments(p)
        
        find_attachments(payload)
        return attachments
    
    def get_thread(self, thread_id: str) -> List[Email]:
        """Fetch all emails in a thread"""
        if self.mock_mode:
            return [e for e in MOCK_EMAILS if e.thread_id == thread_id]
        
        thread = self._service.users().threads().get(
            userId="me",
            id=thread_id,
            format="full",
        ).execute()
        
        emails = []
        for msg in thread.get("messages", []):
            emails.append(self._fetch_email(msg["id"]))
        return emails
    
    def search(self, query: str, max_results: int = 10) -> List[Email]:
        """Search emails with Gmail query syntax"""
        return self.get_inbox(max_results=max_results, query=query)
    
    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------
    def create_draft(self, draft: Draft) -> str:
        """
        Create an email draft
        
        Returns: Draft ID
        """
        if self.mock_mode:
            draft_id = f"mock_draft_{datetime.now().timestamp()}"
            logger.info(f"[MOCK] Created draft {draft_id}: To={draft.to}, Subject={draft.subject}")
            return draft_id
        
        message = self._build_message(draft)
        
        body = {"message": {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode()}}
        
        if draft.thread_id:
            body["message"]["threadId"] = draft.thread_id
        
        result = self._service.users().drafts().create(userId="me", body=body).execute()
        
        logger.info(f"Created draft {result['id']}")
        return result["id"]
    
    def send_email(self, draft: Draft) -> str:
        """
        Send an email directly
        
        Returns: Sent message ID
        """
        if self.mock_mode:
            msg_id = f"mock_sent_{datetime.now().timestamp()}"
            logger.info(f"[MOCK] Sent email {msg_id}: To={draft.to}, Subject={draft.subject}")
            return msg_id
        
        message = self._build_message(draft)
        
        body = {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode()}
        
        if draft.thread_id:
            body["threadId"] = draft.thread_id
        
        result = self._service.users().messages().send(userId="me", body=body).execute()
        
        logger.info(f"Sent email {result['id']}")
        return result["id"]
    
    def send_draft(self, draft_id: str) -> str:
        """Send an existing draft"""
        if self.mock_mode:
            msg_id = f"mock_sent_{datetime.now().timestamp()}"
            logger.info(f"[MOCK] Sent draft {draft_id} as {msg_id}")
            return msg_id
        
        result = self._service.users().drafts().send(
            userId="me",
            body={"id": draft_id},
        ).execute()
        
        return result["id"]
    
    def _build_message(self, draft: Draft) -> MIMEMultipart:
        """Build MIME message from draft"""
        message = MIMEMultipart("alternative")
        message["to"] = ", ".join(draft.to)
        message["subject"] = draft.subject
        message["from"] = self._user_email
        
        if draft.cc:
            message["cc"] = ", ".join(draft.cc)
        
        if draft.reply_to_id:
            message["In-Reply-To"] = draft.reply_to_id
            message["References"] = draft.reply_to_id
        
        # Add plain text body
        message.attach(MIMEText(draft.body, "plain"))
        
        return message
    
    # -------------------------------------------------------------------------
    # Modify Operations
    # -------------------------------------------------------------------------
    def mark_as_read(self, email_id: str) -> bool:
        """Mark email as read"""
        if self.mock_mode:
            logger.info(f"[MOCK] Marked {email_id} as read")
            return True
        
        self._service.users().messages().modify(
            userId="me",
            id=email_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()
        return True
    
    def mark_as_unread(self, email_id: str) -> bool:
        """Mark email as unread"""
        if self.mock_mode:
            logger.info(f"[MOCK] Marked {email_id} as unread")
            return True
        
        self._service.users().messages().modify(
            userId="me",
            id=email_id,
            body={"addLabelIds": ["UNREAD"]},
        ).execute()
        return True
    
    def archive(self, email_id: str) -> bool:
        """Archive email (remove from inbox)"""
        if self.mock_mode:
            logger.info(f"[MOCK] Archived {email_id}")
            return True
        
        self._service.users().messages().modify(
            userId="me",
            id=email_id,
            body={"removeLabelIds": ["INBOX"]},
        ).execute()
        return True
    
    def add_label(self, email_id: str, label: str) -> bool:
        """Add a label to email"""
        if self.mock_mode:
            logger.info(f"[MOCK] Added label {label} to {email_id}")
            return True
        
        self._service.users().messages().modify(
            userId="me",
            id=email_id,
            body={"addLabelIds": [label]},
        ).execute()
        return True


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------
_gmail_client: Optional[GmailClient] = None

def get_gmail_client() -> GmailClient:
    """Get or create Gmail client singleton"""
    global _gmail_client
    if _gmail_client is None:
        _gmail_client = GmailClient()
    return _gmail_client
