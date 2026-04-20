# agents/tool_agents.py
from autogen_agentchat.agents import AssistantAgent
from .docker_manager import DockerManager
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

# Gmail tools import
from agents.gmail_agent import (
    GMAIL_TOOLS,
    gmail_check_inbox,
    gmail_read_email,
    gmail_search,
    gmail_create_draft,
    gmail_draft_reply,
    gmail_send_draft,
    gmail_send_email,
    gmail_mark_read,
    gmail_archive,
    gmail_get_thread,
)

PROMPT_V = "tool@v1.1"
SYSTEM_MESSAGE = f"""
[{PROMPT_V}]
You are the ToolAgent. You execute Python code safely in a sandbox and return stdout/stderr.
If instructions are unclear or unsafe, ask for clarification.
""".strip()


class ToolAgent(AssistantAgent):
    """
    Code execution agent with sandboxed Docker environment.
    Safely executes Python code and returns results.
    """
    
    def __init__(self, name="tool", model_client=None):
        super().__init__(name=name, model_client=model_client, system_message=SYSTEM_MESSAGE)
        self.prompt_version = PROMPT_V
        self.docker = DockerManager()
        
        # ──────────────────────────────────────────────────────────────────────
        # Register capability with FULL SCHEMA (Industry Best Practice)
        # ──────────────────────────────────────────────────────────────────────
        
        register(
            capability="code.execute",
            agent_name=self.name,
            handler=self._cap_exec,
            description="Execute Python code safely in a sandboxed Docker environment and return the output",
            parameters=[
                ToolParameter(
                    name="code_snippet",
                    type=ParameterType.STRING,
                    description="The Python code to execute. Can be single or multi-line code.",
                    required=True
                )
            ],
            category="code",
            examples=[
                {"code_snippet": "print(2 + 2)"},
                {"code_snippet": "import math\nprint(math.sqrt(16))"},
                {"code_snippet": "for i in range(5): print(i)"}
            ]
        )

    async def _cap_exec(self, code_snippet: str) -> str:
        output = self.docker.run_code(code_snippet)
        return f"```python\n{output}\n```"

    async def aask(self, user_proxy, message: str, **kwargs) -> str:
        # optional: parse "run: <code>"
        low = message.strip().lower()
        if low.startswith("run:"):
            return await self._cap_exec(message.split(":",1)[1].strip())
        return await super().aask(user_proxy, message, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Gmail Tool Registration
# ──────────────────────────────────────────────────────────────────────────────

def register_gmail_tools():
    """Register all Gmail tools with the capability system"""
    
    # gmail.inbox - Check inbox
    register(
        capability="gmail.inbox",
        agent_name="gmail",
        handler=gmail_check_inbox,
        description="Check inbox for new and unread emails with sender context",
        parameters=[
            ToolParameter(
                name="max_results",
                type=ParameterType.INTEGER,
                description="Maximum emails to return (default: 10)",
                required=False
            ),
            ToolParameter(
                name="unread_only",
                type=ParameterType.BOOLEAN,
                description="Only show unread emails (default: True)",
                required=False
            ),
        ],
        category="email",
        examples=[
            {"max_results": 5, "unread_only": True},
            {"max_results": 20, "unread_only": False},
        ]
    )
    
    # gmail.read - Read specific email
    register(
        capability="gmail.read",
        agent_name="gmail",
        handler=gmail_read_email,
        description="Read a specific email's full content including body and attachments",
        parameters=[
            ToolParameter(
                name="email_id",
                type=ParameterType.STRING,
                description="The email ID to read",
                required=True
            ),
        ],
        category="email",
        examples=[{"email_id": "mock_001"}]
    )
    
    # gmail.search - Search emails
    register(
        capability="gmail.search",
        agent_name="gmail",
        handler=gmail_search,
        description="Search emails with Gmail query syntax (from:, subject:, has:attachment, etc.)",
        parameters=[
            ToolParameter(
                name="query",
                type=ParameterType.STRING,
                description="Gmail search query (e.g., 'from:boss@company.com', 'subject:urgent')",
                required=True
            ),
            ToolParameter(
                name="max_results",
                type=ParameterType.INTEGER,
                description="Maximum results to return (default: 10)",
                required=False
            ),
        ],
        category="email",
        examples=[
            {"query": "from:rahul@company.com"},
            {"query": "subject:budget is:unread"},
        ]
    )
    
    # gmail.draft - Create draft
    register(
        capability="gmail.draft",
        agent_name="gmail",
        handler=gmail_create_draft,
        description="Create an email draft (does NOT send)",
        parameters=[
            ToolParameter(
                name="to",
                type=ParameterType.STRING,
                description="Recipient email(s), comma-separated",
                required=True
            ),
            ToolParameter(
                name="subject",
                type=ParameterType.STRING,
                description="Email subject",
                required=True
            ),
            ToolParameter(
                name="body",
                type=ParameterType.STRING,
                description="Email body text",
                required=True
            ),
            ToolParameter(
                name="cc",
                type=ParameterType.STRING,
                description="CC recipients, comma-separated (optional)",
                required=False
            ),
        ],
        category="email",
        examples=[
            {"to": "john@example.com", "subject": "Meeting", "body": "Hi John, let's meet tomorrow."},
        ]
    )
    
    # gmail.reply - Draft reply
    register(
        capability="gmail.reply",
        agent_name="gmail",
        handler=gmail_draft_reply,
        description="Create a draft reply to an existing email",
        parameters=[
            ToolParameter(
                name="email_id",
                type=ParameterType.STRING,
                description="The email ID to reply to",
                required=True
            ),
            ToolParameter(
                name="body",
                type=ParameterType.STRING,
                description="Reply body text",
                required=True
            ),
            ToolParameter(
                name="include_cc",
                type=ParameterType.BOOLEAN,
                description="Include original CC recipients (default: True)",
                required=False
            ),
        ],
        category="email",
        examples=[
            {"email_id": "mock_001", "body": "Thanks for the update. I'll review it today."},
        ]
    )
    
    # gmail.send_draft - Send draft
    register(
        capability="gmail.send_draft",
        agent_name="gmail",
        handler=gmail_send_draft,
        description="Send an existing email draft",
        parameters=[
            ToolParameter(
                name="draft_id",
                type=ParameterType.STRING,
                description="The draft ID to send",
                required=True
            ),
        ],
        category="email",
        examples=[{"draft_id": "draft_123"}]
    )
    
    # gmail.send - Send email directly
    register(
        capability="gmail.send",
        agent_name="gmail",
        handler=gmail_send_email,
        description="Send an email directly (prefer draft+send for review)",
        parameters=[
            ToolParameter(
                name="to",
                type=ParameterType.STRING,
                description="Recipient email(s), comma-separated",
                required=True
            ),
            ToolParameter(
                name="subject",
                type=ParameterType.STRING,
                description="Email subject",
                required=True
            ),
            ToolParameter(
                name="body",
                type=ParameterType.STRING,
                description="Email body text",
                required=True
            ),
            ToolParameter(
                name="cc",
                type=ParameterType.STRING,
                description="CC recipients (optional)",
                required=False
            ),
        ],
        category="email",
        examples=[
            {"to": "team@company.com", "subject": "Quick Update", "body": "FYI - project is on track."},
        ]
    )
    
    # gmail.mark_read - Mark as read
    register(
        capability="gmail.mark_read",
        agent_name="gmail",
        handler=gmail_mark_read,
        description="Mark an email as read",
        parameters=[
            ToolParameter(
                name="email_id",
                type=ParameterType.STRING,
                description="The email ID to mark as read",
                required=True
            ),
        ],
        category="email",
        examples=[{"email_id": "mock_001"}]
    )
    
    # gmail.archive - Archive email
    register(
        capability="gmail.archive",
        agent_name="gmail",
        handler=gmail_archive,
        description="Archive an email (removes from inbox, keeps in All Mail)",
        parameters=[
            ToolParameter(
                name="email_id",
                type=ParameterType.STRING,
                description="The email ID to archive",
                required=True
            ),
        ],
        category="email",
        examples=[{"email_id": "mock_001"}]
    )
    
    # gmail.thread - Get conversation thread
    register(
        capability="gmail.thread",
        agent_name="gmail",
        handler=gmail_get_thread,
        description="Get all emails in a conversation thread",
        parameters=[
            ToolParameter(
                name="thread_id",
                type=ParameterType.STRING,
                description="The thread ID",
                required=True
            ),
        ],
        category="email",
        examples=[{"thread_id": "thread_001"}]
    )


# Auto-register Gmail tools on module import
register_gmail_tools()
