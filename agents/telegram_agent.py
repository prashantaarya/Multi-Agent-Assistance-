# agents/telegram_agent.py
"""
Telegram Agent — sends messages, reads updates, and manages Telegram bots.

Setup:
    1. Create a bot via @BotFather on Telegram → get the BOT_TOKEN.
    2. Add TELEGRAM_BOT_TOKEN=<token> to your .env file.
    3. Find your TELEGRAM_CHAT_ID by messaging your bot and calling getUpdates.
    4. Optionally add TELEGRAM_DEFAULT_CHAT_ID=<chat_id> to .env for a default target.

Capabilities registered:
    telegram.send_message  — send a text message to a chat
    telegram.get_updates   — fetch recent incoming messages (inbox)
    telegram.get_chat_info — get info about a chat or user
    telegram.send_alert    — send a formatted alert/notification message
"""

import os
import logging
from typing import Optional, List, Dict, Any

import aiohttp
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

load_dotenv()

logger = logging.getLogger("jarvis.telegram")

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramAgent(AssistantAgent):
    """
    Telegram messaging agent.

    Registers 4 capabilities with the tool registry so the planner
    can route messaging/notification tasks here.

    All API calls are async (aiohttp) — no blocking, no extra SDK dependency.
    The bot token and default chat are read from environment variables:
        TELEGRAM_BOT_TOKEN       (required)
        TELEGRAM_DEFAULT_CHAT_ID (optional — used when no chat_id is passed)
    """

    def __init__(self, name: str = "telegram", model_client=None):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=(
                "You are the TelegramAgent. You can send messages, read updates, "
                "and manage Telegram communication on behalf of the user."
            ),
        )

        self._token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self._default_chat_id: Optional[str] = os.getenv("TELEGRAM_DEFAULT_CHAT_ID")

        if not self._token:
            logger.warning(
                "⚠️  TELEGRAM_BOT_TOKEN not set — Telegram capabilities will return errors. "
                "Add it to your .env file."
            )

        # ──────────────────────────────────────────────────────────────────────
        # Register capabilities — handler + schema in the global tool registry
        # ──────────────────────────────────────────────────────────────────────

        register(
            capability="telegram.send_message",
            agent_name=self.name,
            handler=self.send_message,
            description=(
                "Send a text message to a Telegram chat or user. "
                "Use this to notify, reply, or communicate via Telegram."
            ),
            parameters=[
                ToolParameter(
                    name="text",
                    type=ParameterType.STRING,
                    description="The message text to send (supports Markdown formatting)",
                    required=True,
                ),
                ToolParameter(
                    name="chat_id",
                    type=ParameterType.STRING,
                    description=(
                        "Telegram chat ID or @username to send to. "
                        "Leave empty to use the default chat from TELEGRAM_DEFAULT_CHAT_ID."
                    ),
                    required=False,
                    default="",
                ),
            ],
            category="messaging",
            examples=[
                {"text": "Meeting starts in 10 minutes!"},
                {"text": "Task completed ✅", "chat_id": "123456789"},
            ],
        )

        register(
            capability="telegram.get_updates",
            agent_name=self.name,
            handler=self.get_updates,
            description=(
                "Fetch recent incoming messages (updates) from Telegram. "
                "Use this to read what people have sent to the bot — acts as an inbox."
            ),
            parameters=[
                ToolParameter(
                    name="limit",
                    type=ParameterType.INTEGER,
                    description="Maximum number of recent messages to retrieve (1–20, default 10)",
                    required=False,
                    default=10,
                ),
            ],
            category="messaging",
            examples=[
                {"limit": 5},
                {"limit": 10},
            ],
        )

        register(
            capability="telegram.get_chat_info",
            agent_name=self.name,
            handler=self.get_chat_info,
            description="Get details about a Telegram chat or user (name, type, member count, etc.)",
            parameters=[
                ToolParameter(
                    name="chat_id",
                    type=ParameterType.STRING,
                    description="The Telegram chat ID or @username to look up",
                    required=True,
                ),
            ],
            category="messaging",
            examples=[
                {"chat_id": "123456789"},
                {"chat_id": "@mygroup"},
            ],
        )

        register(
            capability="telegram.send_alert",
            agent_name=self.name,
            handler=self.send_alert,
            description=(
                "Send a formatted alert or notification to Telegram. "
                "Adds a header emoji and bold title — ideal for system alerts, reminders, or summaries."
            ),
            parameters=[
                ToolParameter(
                    name="title",
                    type=ParameterType.STRING,
                    description="Short title/subject for the alert",
                    required=True,
                ),
                ToolParameter(
                    name="body",
                    type=ParameterType.STRING,
                    description="Detailed message body",
                    required=True,
                ),
                ToolParameter(
                    name="chat_id",
                    type=ParameterType.STRING,
                    description="Target chat ID (uses TELEGRAM_DEFAULT_CHAT_ID if omitted)",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="level",
                    type=ParameterType.STRING,
                    description="Alert severity: 'info', 'warning', or 'error'",
                    required=False,
                    default="info",
                ),
            ],
            category="messaging",
            examples=[
                {"title": "Daily Summary", "body": "You have 3 tasks due today."},
                {"title": "System Error", "body": "API call failed", "level": "error"},
            ],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _url(self, method: str) -> str:
        """Build Telegram Bot API URL for a method."""
        return TELEGRAM_API_BASE.format(token=self._token, method=method)

    def _resolve_chat(self, chat_id: str) -> Optional[str]:
        """Return provided chat_id or fall back to env default."""
        return chat_id.strip() if chat_id and chat_id.strip() else self._default_chat_id

    def _check_token(self) -> Optional[Dict]:
        """Return an error dict if token is missing."""
        if not self._token:
            return {
                "success": False,
                "error": (
                    "TELEGRAM_BOT_TOKEN is not set. "
                    "Add it to your .env file and restart the server."
                ),
            }
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Capability handlers
    # ──────────────────────────────────────────────────────────────────────────

    async def send_message(self, text: str, chat_id: str = "") -> Dict[str, Any]:
        """Send a plain text (Markdown) message to a Telegram chat."""
        err = self._check_token()
        if err:
            return err

        target = self._resolve_chat(chat_id)
        if not target:
            return {
                "success": False,
                "error": (
                    "No chat_id provided and TELEGRAM_DEFAULT_CHAT_ID is not set. "
                    "Pass a chat_id or set TELEGRAM_DEFAULT_CHAT_ID in .env."
                ),
            }

        payload = {
            "chat_id": target,
            "text": text,
            "parse_mode": "Markdown",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._url("sendMessage"), json=payload) as resp:
                    data = await resp.json()

            if data.get("ok"):
                msg = data["result"]
                return {
                    "success": True,
                    "message": f"✅ Message sent to chat {target}",
                    "message_id": msg.get("message_id"),
                    "chat_id": target,
                    "text_preview": text[:100],
                }
            return {
                "success": False,
                "error": data.get("description", "Unknown Telegram API error"),
                "error_code": data.get("error_code"),
            }
        except Exception as e:
            logger.error(f"Telegram send_message error: {e}")
            return {"success": False, "error": str(e)}

    async def get_updates(self, limit: int = 10) -> Dict[str, Any]:
        """Fetch recent incoming Telegram messages (the bot's inbox)."""
        err = self._check_token()
        if err:
            return err

        # Clamp limit to a safe range
        limit = max(1, min(int(limit), 20))

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._url("getUpdates"),
                    params={"limit": limit, "allowed_updates": '["message"]'},
                ) as resp:
                    data = await resp.json()

            if not data.get("ok"):
                return {
                    "success": False,
                    "error": data.get("description", "Failed to fetch updates"),
                }

            updates = data.get("result", [])
            messages: List[Dict] = []
            for update in updates:
                msg = update.get("message", {})
                if msg:
                    messages.append(
                        {
                            "from": msg.get("from", {}).get("first_name", "Unknown"),
                            "username": msg.get("from", {}).get("username", ""),
                            "chat_id": msg.get("chat", {}).get("id", ""),
                            "text": msg.get("text", "(non-text message)"),
                            "date": msg.get("date", 0),
                        }
                    )

            return {
                "success": True,
                "total": len(messages),
                "messages": messages,
                "summary": (
                    f"📨 {len(messages)} recent message(s) retrieved"
                    if messages
                    else "📭 No recent messages found"
                ),
            }
        except Exception as e:
            logger.error(f"Telegram get_updates error: {e}")
            return {"success": False, "error": str(e)}

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Fetch information about a Telegram chat or user."""
        err = self._check_token()
        if err:
            return err

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._url("getChat"), params={"chat_id": chat_id}
                ) as resp:
                    data = await resp.json()

            if not data.get("ok"):
                return {
                    "success": False,
                    "error": data.get("description", "Chat not found"),
                }

            chat = data["result"]
            return {
                "success": True,
                "chat_id": chat.get("id"),
                "type": chat.get("type"),
                "title": chat.get("title") or chat.get("first_name", "Unknown"),
                "username": chat.get("username", ""),
                "member_count": chat.get("member_count"),
                "description": chat.get("description", ""),
            }
        except Exception as e:
            logger.error(f"Telegram get_chat_info error: {e}")
            return {"success": False, "error": str(e)}

    async def send_alert(
        self,
        title: str,
        body: str,
        chat_id: str = "",
        level: str = "info",
    ) -> Dict[str, Any]:
        """Send a formatted alert/notification message."""
        level_icons = {"info": "ℹ️", "warning": "⚠️", "error": "🚨"}
        icon = level_icons.get(level.lower(), "ℹ️")

        formatted = f"{icon} *{title}*\n\n{body}"
        return await self.send_message(text=formatted, chat_id=chat_id)
