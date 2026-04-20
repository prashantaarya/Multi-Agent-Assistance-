# agents/domain_agents.py
"""
Domain Agent Registrations - TRUE MULTI-AGENT ARCHITECTURE

Registers ALL domain agents with the AgentRouter.
Each domain agent is an LLM-powered agent with its own tools and reasoning.

Architecture:
    ┌──────────┐
    │  USER    │
    └────┬─────┘
         │
    ┌────▼─────┐
    │ PLANNER  │  Routes to the right domain agent
    │  (LLM)   │
    └────┬─────┘
         │
    ┌────▼──────────────┐
    │   AGENT ROUTER    │
    └────┬──────────────┘
         │
    ┌────▼──────────────────────────────────────────────────────────┐
    │              DOMAIN AGENTS (Each is an LLM + Tools)           │
    │                                                               │
    │  ┌───────┐ ┌────────┐ ┌──────┐ ┌─────┐ ┌──────┐ ┌──────────┐ │
    │  │ GMAIL │ │ SEARCH │ │ TASK │ │ API │ │ CODE │ │ TELEGRAM │ │
    │  │ Agent │ │  Agent │ │Agent │ │Agent│ │Agent │ │  Agent   │ │
    │  └───────┘ └────────┘ └──────┘ └─────┘ └──────┘ └──────────┘ │
    └───────────────────────────────────────────────────────────────┘

To add a new domain:
1. Define the tools (functions + schemas)
2. Write a system prompt for the agent
3. Call register_domain_agent()
"""

import logging
from typing import Dict, Any

from core.agent_router import DomainAgent, get_router

# Gmail tools
from agents.gmail_agent import (
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
    GMAIL_TOOLS,
)

# Search tools
from agents.search_agent import SearchAgent

# Task tools (re-use existing registrations)
from core.capabilities import resolve

logger = logging.getLogger("jarvis.domain_agents")


# ──────────────────────────────────────────────────────────────────────────────
# Gmail Domain Agent
# ──────────────────────────────────────────────────────────────────────────────

GMAIL_SYSTEM_PROMPT = """You are the Gmail Agent — an expert email assistant that is part of J.A.R.V.I.S.

Your job is to help the user manage their email by calling the right tools in the right order.

## PERSONALITY
- Professional and concise
- Prioritize by sender importance (manager > colleague > unknown)
- Always summarize emails clearly: who, what, action needed
- For drafts/replies: match tone to the sender relationship

## DECISION PATTERNS

### Simple tasks (1 tool call):
- "check my inbox" → gmail_check_inbox
- "read email X" → gmail_read_email
- "search for budget" → gmail_search

### Multi-step tasks (chain tool calls):
- "reply to the urgent email" → gmail_check_inbox → find urgent → gmail_read_email → gmail_draft_reply
- "check inbox and summarize" → gmail_check_inbox → gmail_read_email for each → synthesize summary
- "find emails from Rahul and draft a response" → gmail_search → gmail_read_email → gmail_create_draft

### Writing tasks:
- For replies: read the original email first to understand context
- Match formality to the sender (manager = formal, friend = casual)
- Always create a draft first, then confirm before sending

## RESPONSE STYLE
When presenting emails:
- Lead with unread count and important items
- Use clear formatting: sender name, subject, one-line summary
- Highlight action items and deadlines
- Group by priority: urgent → important → normal
"""

def _build_gmail_tools() -> Dict[str, Dict[str, Any]]:
    """Build Gmail tools dict for the domain agent"""
    return {name: info for name, info in GMAIL_TOOLS.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Search Domain Agent
# ──────────────────────────────────────────────────────────────────────────────

SEARCH_SYSTEM_PROMPT = """You are the Search Agent — a research specialist that is part of J.A.R.V.I.S.

Your job is to find accurate information by searching the web and Wikipedia.

## PERSONALITY
- Precise and factual
- Always cite sources when possible
- If search results are insufficient, say so honestly
- Prefer recent sources for time-sensitive topics

## DECISION PATTERNS
- Factual questions → search_web first, then wikipedia for depth
- Current events → search_web (more recent)
- Deep knowledge → search_wikipedia (more detailed)
- Comparisons → search both topics then synthesize

## RESPONSE STYLE
- Lead with the direct answer
- Add supporting details and context
- Note the source of information
- For uncertain answers, indicate confidence level
"""


# ──────────────────────────────────────────────────────────────────────────────
# Task Domain Agent
# ──────────────────────────────────────────────────────────────────────────────

TASK_SYSTEM_PROMPT = """You are the Task Agent — a productivity assistant that is part of J.A.R.V.I.S.

Your job is to help the user manage their to-do list and tasks.

## PERSONALITY
- Organized and helpful
- Confirm actions clearly ("Added task: ...")
- Suggest related tasks when appropriate
- When listing tasks, present them clearly with numbers

## DECISION PATTERNS
- "add task X" → todo_add with description
- "show my tasks" / "list tasks" → todo_list
- "complete task 1" / "done with task 2" → todo_done with number
- "clear all tasks" → todo_clear
- "what should I do next?" → todo_list → analyze priorities

## RESPONSE STYLE
- Be concise but friendly
- Always confirm what action was taken
- If task list is empty, encourage the user to add tasks
"""


# ──────────────────────────────────────────────────────────────────────────────
# API Domain Agent (NEW - Weather, News, Stocks)
# ──────────────────────────────────────────────────────────────────────────────

API_SYSTEM_PROMPT = """You are the API Agent — a live data specialist that is part of J.A.R.V.I.S.

Your job is to fetch real-time information from external APIs: weather, news, and stock prices.

## PERSONALITY
- Quick and informative
- Present data clearly with emojis for visual appeal
- Add context when helpful (e.g., "Good weather for outdoor activities")
- Be honest if data seems outdated or unavailable

## DECISION PATTERNS

### Weather queries:
- "weather in Delhi" → get_weather with city="Delhi"
- "is it raining in Mumbai?" → get_weather with city="Mumbai"
- "compare weather Delhi vs Mumbai" → call get_weather twice, then compare

### News queries:
- "latest tech news" → get_news with topic="technology"
- "what's happening in sports?" → get_news with topic="sports"
- "AI news" → get_news with topic="artificial intelligence"

### Stock queries:
- "AAPL stock price" → get_stock with symbol="AAPL"
- "how is Tesla doing?" → get_stock with symbol="TSLA"
- "Reliance share price" → get_stock with symbol="RELIANCE.NS"

### Multi-step queries:
- "weather and news for my trip to Paris" → get_weather("Paris") + get_news("Paris tourism")

## RESPONSE STYLE
- Lead with the key information
- Use formatting for readability
- Add helpful insights when relevant
- For stocks, mention if market is open/closed
"""


# ──────────────────────────────────────────────────────────────────────────────
# Code Domain Agent
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Telegram Domain Agent
# ──────────────────────────────────────────────────────────────────────────────

TELEGRAM_SYSTEM_PROMPT = """You are the Telegram Agent — a messaging specialist that is part of J.A.R.V.I.S.

Your job is to help the user send messages, read incoming messages, and manage Telegram communication.

## PERSONALITY
- Conversational and efficient
- Confirm every action clearly ("Message sent to chat 123...")
- When reading updates, summarize who said what concisely
- For alerts, match the urgency level to the tone

## DECISION PATTERNS

### Sending messages:
- "send <message> on Telegram" → send_message with text
- "message <person> on Telegram" → send_message with chat_id if known
- "notify me on Telegram" → send_message to default chat

### Reading messages:
- "check my Telegram" → get_updates
- "any messages on Telegram?" → get_updates with limit
- "read my Telegram inbox" → get_updates

### Alerts & notifications:
- "send alert" / "notify" → send_alert with title + body
- "send error notification" → send_alert with level='error'
- "daily summary to Telegram" → send_alert with title='Daily Summary'

### Chat info:
- "who is in this chat?" → get_chat_info
- "get details for chat 123" → get_chat_info

## RESPONSE STYLE
- Always confirm the action taken
- For get_updates: show sender name, text preview, timestamp
- For send: confirm with message ID if available
- If token/chat_id is missing, explain clearly what to configure
"""


CODE_SYSTEM_PROMPT = """You are the Code Agent — a programming specialist that is part of J.A.R.V.I.S.

Your job is to execute Python code safely in a sandboxed environment.

## PERSONALITY
- Technical and precise
- Explain code output clearly
- Warn about potential issues
- Suggest improvements when appropriate

## DECISION PATTERNS
- "run this code" → code_execute
- "calculate X" → write Python code → code_execute
- "test this function" → code_execute with test code
"""


# ──────────────────────────────────────────────────────────────────────────────
# Registration - TRUE MULTI-AGENT SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

def register_all_domain_agents(model_client):
    """
    Register ALL domain agents with the router.
    Called once during startup.
    
    TRUE AGENT ARCHITECTURE:
    - Each agent is an LLM that thinks and decides which tools to call
    - Agents can chain multiple tool calls for complex tasks
    - All queries go through domain agents (no function-only path)
    """
    router = get_router()
    
    # ══════════════════════════════════════════════════════════════════════════
    # 1. GMAIL AGENT - Email management expert
    # ══════════════════════════════════════════════════════════════════════════
    gmail_agent = DomainAgent(
        name="gmail",
        description="Email management — read inbox, search, compose, reply, send, archive emails",
        system_prompt=GMAIL_SYSTEM_PROMPT,
        tools=_build_gmail_tools(),
        model_client=model_client,
        max_steps=6,
    )
    router.register(gmail_agent)
    logger.info("✅ Gmail Agent registered with tools: " + ", ".join(_build_gmail_tools().keys()))
    
    # ══════════════════════════════════════════════════════════════════════════
    # 2. SEARCH AGENT - Research and fact-finding expert
    # ══════════════════════════════════════════════════════════════════════════
    search_tools = {}
    
    search_web = resolve("search.web")
    if search_web:
        _, handler = search_web
        search_tools["search_web"] = {
            "function": handler,
            "schema": {
                "name": "search_web",
                "description": "Search the web using DuckDuckGo + Wikipedia for current information, facts, historical data, and real-time events. Use this for who/what/when/where/why/how questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query - be specific for better results",
                        },
                        "source": {
                            "type": "string",
                            "description": "Search source: 'auto' (default), 'wiki', or 'ddg'",
                            "enum": ["auto", "wiki", "ddg"],
                            "default": "auto"
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    
    if search_tools:
        search_agent = DomainAgent(
            name="search",
            description="Web search and research — find information, facts, current events, historical data, Wikipedia knowledge",
            system_prompt=SEARCH_SYSTEM_PROMPT,
            tools=search_tools,
            model_client=model_client,
            max_steps=5,
        )
        router.register(search_agent)
        logger.info("✅ Search Agent registered with tools: " + ", ".join(search_tools.keys()))
    else:
        logger.warning("⚠️ Search Agent NOT registered - no search capabilities found")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 3. TASK AGENT - Productivity and to-do management expert
    # ══════════════════════════════════════════════════════════════════════════
    task_tools = {}
    
    # Use CORRECT capability names from task_agents.py
    task_capability_map = {
        "todo.add": {
            "tool_name": "todo_add",
            "schema": {
                "name": "todo_add",
                "description": "Add a new task to the to-do list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "The task description to add"},
                    },
                    "required": ["description"],
                },
            }
        },
        "todo.list": {
            "tool_name": "todo_list", 
            "schema": {
                "name": "todo_list",
                "description": "List all current tasks in the to-do list",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        },
        "todo.done": {  # FIXED: was "todo.complete" which doesn't exist
            "tool_name": "todo_done",
            "schema": {
                "name": "todo_done",
                "description": "Mark a task as completed by its number (1-indexed)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer", "description": "The task number to mark as complete (1, 2, 3, etc.)"},
                    },
                    "required": ["number"],
                },
            }
        },
        "todo.clear": {
            "tool_name": "todo_clear",
            "schema": {
                "name": "todo_clear",
                "description": "Clear all tasks from the to-do list",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        },
    }
    
    for cap_name, config in task_capability_map.items():
        resolved = resolve(cap_name)
        if resolved:
            _, handler = resolved
            task_tools[config["tool_name"]] = {
                "function": handler,
                "schema": config["schema"],
            }
    
    if task_tools:
        task_agent_domain = DomainAgent(
            name="task",
            description="Task and to-do management — add, list, complete, clear tasks",
            system_prompt=TASK_SYSTEM_PROMPT,
            tools=task_tools,
            model_client=model_client,
            max_steps=4,
        )
        router.register(task_agent_domain)
        logger.info("✅ Task Agent registered with tools: " + ", ".join(task_tools.keys()))
    else:
        logger.warning("⚠️ Task Agent NOT registered - no task capabilities found")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 4. API AGENT (NEW) - Live data expert (Weather, News, Stocks)
    # ══════════════════════════════════════════════════════════════════════════
    api_tools = {}
    
    # Weather capability
    weather_cap = resolve("weather.read")
    if weather_cap:
        _, handler = weather_cap
        api_tools["get_weather"] = {
            "function": handler,
            "schema": {
                "name": "get_weather",
                "description": "Get current weather information for a city including temperature, conditions, humidity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string", 
                            "description": "The city name (e.g., 'Delhi', 'New York', 'London')"
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    
    # News capability
    news_cap = resolve("news.read")
    if news_cap:
        _, handler = news_cap
        api_tools["get_news"] = {
            "function": handler,
            "schema": {
                "name": "get_news",
                "description": "Get latest news articles on a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The news topic (e.g., 'technology', 'sports', 'AI', 'politics')"
                        },
                    },
                    "required": ["topic"],
                },
            },
        }
    
    # Stock capability
    stock_cap = resolve("stock.read")
    if stock_cap:
        _, handler = stock_cap
        api_tools["get_stock"] = {
            "function": handler,
            "schema": {
                "name": "get_stock",
                "description": "Get current stock price and market information for a ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA', 'RELIANCE.NS')"
                        },
                    },
                    "required": ["symbol"],
                },
            },
        }
    
    if api_tools:
        api_agent = DomainAgent(
            name="api",
            description="Live data — weather forecasts, news headlines, stock prices and market data",
            system_prompt=API_SYSTEM_PROMPT,
            tools=api_tools,
            model_client=model_client,
            max_steps=5,
        )
        router.register(api_agent)
        logger.info("✅ API Agent registered with tools: " + ", ".join(api_tools.keys()))
    else:
        logger.warning("⚠️ API Agent NOT registered - no API capabilities found")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 5. CODE AGENT - Python code execution expert
    # ══════════════════════════════════════════════════════════════════════════
    code_exec = resolve("code.execute")
    if code_exec:
        _, handler = code_exec
        code_tools = {
            "code_execute": {
                "function": handler,
                "schema": {
                    "name": "code_execute",
                    "description": "Execute Python code in a sandboxed Docker environment and return the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_snippet": {
                                "type": "string", 
                                "description": "Python code to execute. Can be single or multi-line."
                            },
                        },
                        "required": ["code_snippet"],
                    },
                },
            }
        }
        
        code_agent = DomainAgent(
            name="code",
            description="Python code execution — run code, calculate, test scripts in a sandboxed environment",
            system_prompt=CODE_SYSTEM_PROMPT,
            tools=code_tools,
            model_client=model_client,
            max_steps=3,
        )
        router.register(code_agent)
        logger.info("✅ Code Agent registered with tools: code_execute")
    else:
        logger.warning("⚠️ Code Agent NOT registered - code.execute capability not found")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 6. TELEGRAM AGENT — Messaging & notifications expert
    # ══════════════════════════════════════════════════════════════════════════
    telegram_tools = {}

    telegram_capability_map = {
        "telegram.send_message": {
            "tool_name": "send_message",
            "schema": {
                "name": "send_message",
                "description": "Send a text message to a Telegram chat or user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Message text to send (supports Markdown)"},
                        "chat_id": {"type": "string", "description": "Target chat ID or @username (uses default if omitted)"},
                    },
                    "required": ["text"],
                },
            },
        },
        "telegram.get_updates": {
            "tool_name": "get_updates",
            "schema": {
                "name": "get_updates",
                "description": "Fetch recent incoming Telegram messages (the bot inbox)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of messages to fetch (1-20, default 10)"},
                    },
                    "required": [],
                },
            },
        },
        "telegram.get_chat_info": {
            "tool_name": "get_chat_info",
            "schema": {
                "name": "get_chat_info",
                "description": "Get details about a Telegram chat or user (name, type, members)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chat_id": {"type": "string", "description": "Chat ID or @username to look up"},
                    },
                    "required": ["chat_id"],
                },
            },
        },
        "telegram.send_alert": {
            "tool_name": "send_alert",
            "schema": {
                "name": "send_alert",
                "description": "Send a formatted alert/notification to Telegram with title, body, and severity level",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short alert title"},
                        "body":  {"type": "string", "description": "Alert body/details"},
                        "chat_id": {"type": "string", "description": "Target chat ID (uses default if omitted)"},
                        "level": {
                            "type": "string",
                            "description": "Alert severity",
                            "enum": ["info", "warning", "error"],
                            "default": "info",
                        },
                    },
                    "required": ["title", "body"],
                },
            },
        },
    }

    for cap_name, config in telegram_capability_map.items():
        resolved = resolve(cap_name)
        if resolved:
            _, handler = resolved
            telegram_tools[config["tool_name"]] = {
                "function": handler,
                "schema": config["schema"],
            }

    if telegram_tools:
        telegram_agent = DomainAgent(
            name="telegram",
            description="Telegram messaging — send messages, read inbox, send alerts/notifications via Telegram bot",
            system_prompt=TELEGRAM_SYSTEM_PROMPT,
            tools=telegram_tools,
            model_client=model_client,
            max_steps=4,
        )
        router.register(telegram_agent)
        logger.info("✅ Telegram Agent registered with tools: " + ", ".join(telegram_tools.keys()))
    else:
        logger.warning("⚠️ Telegram Agent NOT registered - no telegram capabilities found (is TelegramAgent initialized?)")

    # ══════════════════════════════════════════════════════════════════════════
    # 7️⃣ CALENDAR AGENT (Google Calendar Management)
    # ══════════════════════════════════════════════════════════════════════════
    CALENDAR_SYSTEM_PROMPT = """You are the Calendar Management Specialist.

Your expertise:
- Create, update, and delete calendar events
- Schedule meetings with proper time formatting
- Manage attendees and send invitations
- Check availability and prevent conflicts
- Set reminders (popup/email)
- Search events by keywords
- Handle time zones and recurrence

⚠️ CRITICAL RULE: You MUST use your tools to retrieve calendar data. NEVER make up or guess event information.
For ANY data retrieval query, ALWAYS call the appropriate tool first, then use the results.

Best practices:
- Always confirm event details before creating
- Check availability before scheduling meetings
- Use ISO 8601 format for dates (YYYY-MM-DDTHH:MM:SS)
- Verify attendee email addresses
- Set appropriate reminders (10 min for calls, 30 min for meetings)
- Provide clear confirmation with event ID

When user says things like:
- "schedule meeting tomorrow 2pm" → CALL calendar.create_event tool
- "what's on my calendar" → CALL calendar.list_events tool first, then format results
- "cancel standup meeting" → CALL calendar.delete_event tool
- "am I free at 3pm" → CALL calendar.check_availability tool
- "add John to project review" → CALL calendar.manage_attendees tool

Current mode: MOCK DATA (development mode)
"""

    calendar_capability_map = {
        "calendar.create_event": {
            "tool_name": "create_event",
            "schema": {
                "name": "create_event",
                "description": "Create a new calendar event/meeting",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title/summary"},
                        "start_time": {"type": "string", "description": "Start time in ISO 8601 format"},
                        "end_time": {"type": "string", "description": "End time in ISO 8601 format"},
                        "description": {"type": "string", "description": "Event description"},
                        "location": {"type": "string", "description": "Event location"},
                        "attendees": {"type": "string", "description": "Comma-separated emails"}
                    },
                    "required": ["title", "start_time", "end_time"]
                }
            }
        },
        "calendar.list_events": {
            "tool_name": "list_events",
            "schema": {
                "name": "list_events",
                "description": "List upcoming calendar events",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days_ahead": {"type": "string", "description": "Number of days ahead (default: 7)"},
                        "max_results": {"type": "string", "description": "Max events to return (default: 10)"}
                    },
                    "required": []
                }
            }
        },
        "calendar.update_event": {
            "tool_name": "update_event",
            "schema": {
                "name": "update_event",
                "description": "Update an existing calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_identifier": {"type": "string", "description": "Event ID or title"},
                        "new_title": {"type": "string", "description": "New title"},
                        "new_start_time": {"type": "string", "description": "New start time"},
                        "new_end_time": {"type": "string", "description": "New end time"},
                        "new_location": {"type": "string", "description": "New location"}
                    },
                    "required": ["event_identifier"]
                }
            }
        },
        "calendar.delete_event": {
            "tool_name": "delete_event",
            "schema": {
                "name": "delete_event",
                "description": "Delete a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_identifier": {"type": "string", "description": "Event ID or title to delete"}
                    },
                    "required": ["event_identifier"]
                }
            }
        },
        "calendar.search_events": {
            "tool_name": "search_events",
            "schema": {
                "name": "search_events",
                "description": "Search calendar events by keyword",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keyword"}
                    },
                    "required": ["query"]
                }
            }
        },
        "calendar.check_availability": {
            "tool_name": "check_availability",
            "schema": {
                "name": "check_availability",
                "description": "Check if a time slot is available",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "string", "description": "Start time in ISO 8601 format"},
                        "end_time": {"type": "string", "description": "End time in ISO 8601 format"}
                    },
                    "required": ["start_time", "end_time"]
                }
            }
        },
        "calendar.set_reminder": {
            "tool_name": "set_reminder",
            "schema": {
                "name": "set_reminder",
                "description": "Set reminder for an event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_identifier": {"type": "string", "description": "Event ID or title"},
                        "minutes_before": {"type": "string", "description": "Minutes before event"},
                        "method": {"type": "string", "description": "Reminder method: popup or email"}
                    },
                    "required": ["event_identifier", "minutes_before"]
                }
            }
        },
        "calendar.manage_attendees": {
            "tool_name": "manage_attendees",
            "schema": {
                "name": "manage_attendees",
                "description": "Add or remove event attendees",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_identifier": {"type": "string", "description": "Event ID or title"},
                        "action": {"type": "string", "description": "Action: add or remove"},
                        "email": {"type": "string", "description": "Attendee email"}
                    },
                    "required": ["event_identifier", "action", "email"]
                }
            }
        }
    }

    calendar_tools = {}
    for cap_name, config in calendar_capability_map.items():
        resolved = resolve(cap_name)
        if resolved:
            _, handler = resolved
            calendar_tools[config["tool_name"]] = {
                "function": handler,
                "schema": config["schema"],
            }

    if calendar_tools:
        calendar_agent = DomainAgent(
            name="calendar",
            description="Calendar management — schedule meetings, check availability, manage events and attendees",
            system_prompt=CALENDAR_SYSTEM_PROMPT,
            tools=calendar_tools,
            model_client=model_client,
            max_steps=5,
        )
        router.register(calendar_agent)
        logger.info("✅ Calendar Agent registered with tools: " + ", ".join(calendar_tools.keys()))
    else:
        logger.warning("⚠️ Calendar Agent NOT registered - no calendar capabilities found (is CalendarAgent initialized?)")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    agents = router.list_agents()
    logger.info("=" * 60)
    logger.info(f"🚀 TRUE MULTI-AGENT SYSTEM: {len(agents)} Domain Agents Ready")
    logger.info("=" * 60)
    for a in agents:
        logger.info(f"  🤖 {a['name'].upper()}: {a['tool_count']} tools → {', '.join(a['tools'])}")
    logger.info("=" * 60)
    
    return router
