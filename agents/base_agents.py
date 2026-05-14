# agents/base_agents.py

import os
import re
import asyncio
import json
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

# Specialized agents (each registers its own capabilities on init)
from .task_agents import TaskAgent
from .tool_agents import ToolAgent
from .api_agent import APIAgent
from .search_agent import SearchAgent
from .telegram_agent import TelegramAgent
from .calendar_agent import CalendarAgent
from .resume_agent import ResumeAgent
from .job_agent import JobAgent

# Capability routing + typed contracts + tracing
from core.schemas import Task, PlannerDecision, Result, Artifact, ExecutionMode, CapabilityCall
from core.capabilities import resolve, list_capabilities, list_tools
from core.tracing import get_tracer

# True Agent Architecture - agent router
from core.agent_router import get_router

# Memory system for context-aware responses
from .memory import get_memory

# ──────────────────────────────────────────────────────────────────────────────
# Heuristics for guardrails (force search for factual/temporal queries)
# ──────────────────────────────────────────────────────────────────────────────

_FACTUAL_RE = re.compile(r"^(who|what|when|where|why|how)\b", re.IGNORECASE)

def _looks_factual_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _FACTUAL_RE.match(t):
        return True
    keywords = [
        "year", "date", "history", "started", "founded",
        "died", "born", "population", "capital", "definition",
        "when did", "when was", "what year"
    ]
    lt = t.lower()
    return any(k in lt for k in keywords)

def _looks_time_sensitive(text: str) -> bool:
    lt = (text or "").lower()
    terms = ["today", "latest", "current", "now", "this year", "this month", "right now"]
    return any(term in lt for term in terms)

# ──────────────────────────────────────────────────────────────────────────────
# Environment & model client
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables!")

# Model configuration (include structured_output for future compatibility)
model_info = ModelInfo(
    family="llama",
    vision=False,
    function_calling=True,
    structured_output=False,
    json_output=False,
)

# Create model client (Groq OpenAI-compatible endpoint)
model_client = OpenAIChatCompletionClient(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
    model_info=model_info,
    create_config={
        "temperature": 0.7,   # Increased from 0.3 to prevent echo behavior
        "max_tokens": 2000,   # Increased from 1000 for longer responses
        "top_p": 0.9,
        "stop": ["TERMINATE"],
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# Specialist agents (each registers its own capabilities on init)
# MUST be initialized BEFORE planner to populate the tool registry
# ──────────────────────────────────────────────────────────────────────────────

task_agent     = TaskAgent(name="task", model_client=model_client)
tool_agent     = ToolAgent(name="tool", model_client=model_client)
api_agent      = APIAgent(name="api", model_client=model_client)
search_agent   = SearchAgent(name="search", model_client=model_client)
telegram_agent = TelegramAgent(name="telegram", model_client=model_client)
calendar_agent = CalendarAgent(name="calendar", model_client=model_client)
resume_agent   = ResumeAgent(name="resume", model_client=model_client)
job_agent      = JobAgent(name="job", model_client=model_client)

# ──────────────────────────────────────────────────────────────────────────────
# True Agent Architecture: Register domain agents
# Each domain agent is an LLM that thinks + calls tools
# ──────────────────────────────────────────────────────────────────────────────

from agents.domain_agents import register_all_domain_agents
agent_router = register_all_domain_agents(model_client)

# ──────────────────────────────────────────────────────────────────────────────
# Planner (brain): delegates to DOMAIN AGENTS (True Agent Architecture)
# Planner picks the right agent, the agent decides which tools to call
# ──────────────────────────────────────────────────────────────────────────────

PLANNER_PROMPT_V = "planner@v6.1"  # Telegram-intent routing hardening

def _build_planner_system_message() -> str:
    """
    Build planner system message for TRUE AGENT Architecture.
    
    ALL queries route to DOMAIN AGENTS (not direct functions).
    Each domain agent is an LLM that decides which tools to call.
    """
    # Get high-level agent descriptions from the router.
    # SECURITY / TOKEN BEST PRACTICE: planner only sees agent names + descriptions.
    # Full tool schemas are passed to each agent separately at execution time.
    router = get_router()
    agents_doc = router.get_agent_descriptions()

    return f"""
[{PLANNER_PROMPT_V}]
You are the Planner (JARVIS's brain). Analyze the user's request and route to the right DOMAIN AGENT.

## TRUE MULTI-AGENT ARCHITECTURE
You have specialized DOMAIN AGENTS. Each agent is an AI expert that thinks and decides which tools to call.
Your job: pick the RIGHT AGENT and describe the TASK clearly. The agent handles the rest.

## AVAILABLE DOMAIN AGENTS
{agents_doc}

## OUTPUT SCHEMA (ALWAYS output valid JSON)
{{
  "agent": "<agent_name>",  // REQUIRED: gmail, search, task, api, code
  "task": "<clear description of what the user wants>",  // REQUIRED
  "capability": "<primary_capability_name>",
  "inputs": {{ <parameters> }},
  "confidence": <0.0 to 1.0>,
  "fallback": "<alternative_agent | null>",
  "mode": "single | react | parallel",
  "reasoning": "<brief explanation>"
}}

## ROUTING RULES (STRICT - ALWAYS route to a domain agent)

| Query Type | Route To | Examples |
|------------|----------|----------|
| EMAIL | agent: "gmail" | inbox, read email, send, draft, reply, search emails |
| FACTUAL/RESEARCH | agent: "search" | who/what/when/where/why/how, history, facts, definitions |
| WEATHER | agent: "api" | weather in <city>, temperature, forecast |
| NEWS | agent: "api" | latest news, headlines, news about <topic> |
| STOCKS | agent: "api" | stock price, AAPL, market, shares |
| TASKS/TODO | agent: "task" | add task, list tasks, complete task, clear tasks |
| CODE | agent: "code" | run Python, calculate, execute code |
| TELEGRAM | agent: "telegram" | send message, check Telegram, notify, alert via Telegram |
| CALENDAR | agent: "calendar" | schedule meeting, check calendar, book appointment, am I free, cancel event |
| RESUME | agent: "resume" | parse resume, review resume, analyze resume, my resume, show profile, update preferences |
| JOBS | agent: "job" | search jobs, find jobs, rank jobs, apply, job listings, job match, find work, looking for job |
| SMALL TALK | NO JSON | hi, hello, how are you, thanks (ONLY when no domain keyword is present) |

## EXAMPLES - WEATHER/NEWS/STOCKS → API AGENT

User: "what's the weather in Delhi?"
{{"agent":"api","task":"Get current weather in Delhi","capability":"weather.read","inputs":{{"city":"Delhi"}},"confidence":0.95,"fallback":"search","mode":"single","reasoning":"Weather query routes to API agent"}}

User: "latest tech news"
{{"agent":"api","task":"Get latest technology news","capability":"news.read","inputs":{{"topic":"technology"}},"confidence":0.92,"fallback":"search","mode":"single","reasoning":"News query routes to API agent"}}

User: "AAPL stock price"
{{"agent":"api","task":"Get current Apple stock price","capability":"stock.read","inputs":{{"symbol":"AAPL"}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Stock query routes to API agent"}}

User: "weather in Mumbai and stock update"
{{"agent":"api","task":"Get Mumbai weather and market update","capability":"weather.read","inputs":{{}},"confidence":0.88,"mode":"single","reasoning":"API agent can handle both weather and stocks"}}

## EXAMPLES - SEARCH → SEARCH AGENT

User: "who invented Python?"
{{"agent":"search","task":"Find who invented the Python programming language","capability":"search.web","inputs":{{"query":"who invented Python"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Factual question → search agent"}}

User: "when did World War 2 end?"
{{"agent":"search","task":"Find when World War 2 ended","capability":"search.web","inputs":{{"query":"when did World War 2 end"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Historical fact → search agent"}}

User: "what is quantum computing?"
{{"agent":"search","task":"Explain what quantum computing is","capability":"search.web","inputs":{{"query":"what is quantum computing"}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Definition → search agent"}}

## EXAMPLES - EMAIL → GMAIL AGENT

User: "check my inbox"
{{"agent":"gmail","task":"Check email inbox for new and unread messages","capability":"gmail.inbox","inputs":{{"max_results":10,"unread_only":true}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Email query → gmail agent"}}

User: "find urgent emails and draft a reply"
{{"agent":"gmail","task":"Find urgent emails and draft professional replies","capability":"gmail.inbox","inputs":{{}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Complex email task → gmail agent handles multi-step"}}

## EXAMPLES - TASKS → TASK AGENT

User: "add task Buy groceries"
{{"agent":"task","task":"Add a new task: Buy groceries","capability":"todo.add","inputs":{{"description":"Buy groceries"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Task management → task agent"}}

User: "show my tasks"
{{"agent":"task","task":"List all current tasks","capability":"todo.list","inputs":{{}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"List tasks → task agent"}}

User: "complete task 1"
{{"agent":"task","task":"Mark task number 1 as done","capability":"todo.done","inputs":{{"number":1}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Complete task → task agent"}}

## EXAMPLES - TELEGRAM → TELEGRAM AGENT

User: "send hello on Telegram"
{{"agent":"telegram","task":"Send 'hello' message via Telegram","capability":"telegram.send_message","inputs":{{"text":"hello"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Telegram messaging → telegram agent"}}

User: "check my Telegram messages"
{{"agent":"telegram","task":"Fetch recent incoming Telegram messages","capability":"telegram.get_updates","inputs":{{"limit":10}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Read Telegram inbox → telegram agent"}}

User: "send a daily summary alert on Telegram"
{{"agent":"telegram","task":"Send formatted daily summary alert via Telegram","capability":"telegram.send_alert","inputs":{{"title":"Daily Summary","body":"Your daily summary","level":"info"}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Alert/notification → telegram agent"}}

## EXAMPLES - CALENDAR → CALENDAR AGENT

User: "schedule a meeting tomorrow at 2pm"
{{"agent":"calendar","task":"Create calendar event for meeting tomorrow at 2pm","capability":"calendar.create_event","inputs":{{"title":"Meeting","start_time":"2026-04-16T14:00:00","end_time":"2026-04-16T15:00:00"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Schedule meeting → calendar agent"}}

User: "what's on my calendar this week"
{{"agent":"calendar","task":"List all calendar events for this week","capability":"calendar.list_events","inputs":{{"days_ahead":"7","max_results":"20"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"View calendar → calendar agent"}}

User: "am I free tomorrow at 3pm"
{{"agent":"calendar","task":"Check availability for tomorrow 3pm-4pm","capability":"calendar.check_availability","inputs":{{"start_time":"2026-04-16T15:00:00","end_time":"2026-04-16T16:00:00"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Check availability → calendar agent"}}

User: "cancel the team standup meeting"
{{"agent":"calendar","task":"Delete team standup meeting from calendar","capability":"calendar.delete_event","inputs":{{"event_identifier":"Team Standup"}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Delete event → calendar agent"}}

User: "add John to the project review meeting"
{{"agent":"calendar","task":"Add John as attendee to project review meeting","capability":"calendar.manage_attendees","inputs":{{"event_identifier":"Project Review","action":"add","email":"john@company.com"}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Manage attendees → calendar agent"}}

## EXAMPLES - RESUME → RESUME AGENT

User: "review my resume"
{{"agent":"resume","task":"Auto-detect and parse user's resume","capability":"resume.auto_detect","inputs":{{"user_id":"default_user"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"User wants resume review without specifying path → auto-detect resume"}}

User: "parse my resume"
{{"agent":"resume","task":"Auto-detect and parse resume","capability":"resume.auto_detect","inputs":{{"user_id":"default_user"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Parse resume without explicit path → auto-detect"}}

User: "show me my resume"
{{"agent":"resume","task":"Auto-detect and display resume","capability":"resume.auto_detect","inputs":{{"user_id":"default_user"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Display resume → auto-detect first"}}

User: "analyze my resume"
{{"agent":"resume","task":"Auto-detect and analyze resume","capability":"resume.auto_detect","inputs":{{"user_id":"default_user"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Analyze resume → auto-detect first"}}

User: "parse my resume from data/resumes/resume.pdf as john_doe"
{{"agent":"resume","task":"Parse specific resume file","capability":"resume.parse","inputs":{{"user_id":"john_doe","file_path":"data/resumes/resume.pdf","use_llm":true}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Explicit file path given → use parse_resume"}}

User: "show my profile"
{{"agent":"resume","task":"Retrieve user's parsed resume profile","capability":"resume.get_profile","inputs":{{"user_id":"default_user"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Get profile → resume agent"}}

User: "how well do I match this job description: Python developer with 3 years experience"
{{"agent":"resume","task":"Analyze job fit for Python developer role","capability":"resume.analyze_fit","inputs":{{"user_id":"default_user","job_description":"Python developer with 3 years experience"}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Analyze job fit → resume agent"}}

User: "update my job preferences for Delhi and Bangalore"
{{"agent":"resume","task":"Update job search preferences","capability":"resume.update_preferences","inputs":{{"user_id":"default_user","preferences":{{"locations":["Delhi","Bangalore"],"roles":["Backend Developer"]}}}},"confidence":0.85,"fallback":null,"mode":"single","reasoning":"Update preferences → resume agent"}}
## EXAMPLES - JOBS → JOB AGENT

User: "find Python jobs in Delhi"
{{"agent":"job","task":"Search for Python developer jobs in Delhi","capability":"job.search","inputs":{{"query":"Python Developer","location":"Delhi, India","num_results":10}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Job search → job agent"}}

User: "rank these jobs for me"
{{"agent":"job","task":"Rank job listings by match score","capability":"job.rank","inputs":{{"user_id":"default_user","min_score":50}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Rank jobs → job agent"}}

User: "show me details for job mock_001"
{{"agent":"job","task":"Get full details for job mock_001","capability":"job.get_details","inputs":{{"job_id":"mock_001"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Job details → job agent"}}

User: "I applied to that ML Engineer job"
{{"agent":"job","task":"Track application to ML Engineer position","capability":"job.track_application","inputs":{{"user_id":"default_user","job_id":"mock_003","status":"applied"}},"confidence":0.85,"fallback":null,"mode":"single","reasoning":"Log application → job agent"}}

User: "find the job related to my resume only 5 jobs"
{{"agent":"job","task":"Find jobs matching the user's resume (infer role from resume)","capability":"job.search","inputs":{{"query":"","num_results":5}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"job.search auto-detects the resume and infers the role — never ask the user for a title"}}

User: "find jobs related to my resume"
{{"agent":"job","task":"Find jobs matching the user's resume","capability":"job.search","inputs":{{"query":"","num_results":10}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"job.search infers the role from the resume"}}

User: "review my resume and find relevant jobs"
{{"agent":"job","task":"Find jobs matching the resume","capability":"job.search","inputs":{{"query":"","num_results":10}},"confidence":0.92,"fallback":null,"mode":"single","reasoning":"job.search auto-parses the resume internally and ranks results — no separate resume step needed"}}

User: "find AI Engineer jobs for me"
{{"agent":"job","task":"Search AI Engineer jobs and rank against resume","capability":"job.search","inputs":{{"query":"AI Engineer","num_results":10}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Explicit role → job.search (auto-uses resume for ranking)"}}

User: "help me find a new job"
{{"agent":"job","task":"Find jobs matching the user's resume","capability":"job.search","inputs":{{"query":"","num_results":10}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"job.search infers role from resume"}}

## EXAMPLES - CODE → CODE AGENT

User: "run print(2+2)"
{{"agent":"code","task":"Execute Python code: print(2+2)","capability":"code.execute","inputs":{{"code_snippet":"print(2+2)"}},"confidence":0.95,"fallback":null,"mode":"single","reasoning":"Code execution → code agent"}}

User: "calculate the square root of 144"
{{"agent":"code","task":"Calculate square root of 144 using Python","capability":"code.execute","inputs":{{"code_snippet":"import math; print(math.sqrt(144))"}},"confidence":0.9,"fallback":"search","mode":"single","reasoning":"Math calculation → code agent"}}

## EXAMPLES - REACT MODE (Complex multi-step)

User: "compare weather in Delhi vs Mumbai for travel"
{{"agent":"api","task":"Compare weather in Delhi and Mumbai and recommend for travel","capability":"weather.read","inputs":{{}},"confidence":0.85,"mode":"react","reasoning":"Comparison needs multiple API calls then analysis"}}

User: "research AI trends and summarize"
{{"agent":"search","task":"Research current AI trends and provide summary","capability":"search.web","inputs":{{}},"confidence":0.8,"mode":"react","reasoning":"Research needs multiple searches and synthesis"}}

## SMALL TALK (NO JSON - respond naturally)

IMPORTANT: If the user mentions a domain keyword (telegram, email, task, weather, news, stock, code, search), DO NOT treat it as small talk.
You MUST output JSON and route to the appropriate domain agent.

User: "Hi!" → Hello! I'm JARVIS. How can I help you today?
User: "thanks" → You're welcome! Let me know if you need anything else.
User: "hi telegram bot" → {{"agent":"telegram","task":"Send a friendly greeting reply via Telegram and confirm bot readiness","capability":"telegram.send_message","inputs":{{"text":"Hi! I am online and ready to help."}},"confidence":0.9,"fallback":null,"mode":"single","reasoning":"Contains telegram domain keyword, should route to Telegram agent"}}

CRITICAL: ALWAYS include "agent" and "task" fields. Output ONLY valid JSON for routing (no markdown).
""".strip()


# Build the system message with auto-generated tools
PLANNER_SYSTEM_MESSAGE = _build_planner_system_message()

# NOTE: Do NOT create a single shared planner_agent — RoundRobinGroupChat keeps
# conversation history on the agent instance, causing previous queries to bleed
# into the current one. Instead, use _make_planner() to get a fresh agent per request.
def _make_planner() -> AssistantAgent:
    """Return a stateless planner agent (fresh per request, no history bleed)."""
    return AssistantAgent(
        name="planner",
        model_client=model_client,
        system_message=PLANNER_SYSTEM_MESSAGE,
    )


def _looks_telegram_intent(text: str) -> bool:
    """Detect explicit Telegram intent keywords for routing safety."""
    lt = (text or "").lower()
    telegram_terms = [
        "telegram",
        "tg",
        "bot",
        "message on telegram",
        "send on telegram",
        "notify on telegram",
    ]
    return any(term in lt for term in telegram_terms)


# Deterministic job-intent detection. The LLM planner sometimes routes job
# requests to the resume agent (which then refuses to call search_jobs).
# This pre-route short-circuits straight to job.search.
_JOB_INTENT_TERMS = (
    "job", "jobs", "vacancy", "vacancies", "opening", "openings",
    "hiring", "career", "careers", "position", "positions", "role",
    "apply to", "looking for work", "find work", "naukri", "linkedin job",
)
_JOB_NEGATIVE_TERMS = (
    "review my resume", "parse my resume", "show my resume",
    "analyze my resume", "show my profile", "update my preferences",
    "update my profile",
)


def _looks_job_intent(text: str) -> bool:
    """True if the user clearly wants jobs (not just a resume review)."""
    lt = (text or "").lower()
    if any(neg in lt for neg in _JOB_NEGATIVE_TERMS):
        return False
    return any(term in lt for term in _JOB_INTENT_TERMS)


def _parse_job_query(text: str) -> dict:
    """
    Extract (query, location, num_results) hints from a free-form job request.
    Conservative: returns empty query when the user didn't name a role —
    JobAgent.search_jobs will infer from the resume.
    """
    import re

    raw = (text or "").strip()
    lt = raw.lower()

    # ── num_results ────────────────────────────────────────────────────────
    num = 10
    # "5 jobs", "5 results", "5 Python Developer jobs"
    m = re.search(r"\b(\d{1,3})\s+\S+(?:\s+\S+){0,4}?\s+(?:jobs?|roles?|positions?|openings?|results?|listings?)\b", lt)
    if not m:
        m = re.search(r"\b(\d{1,3})\s+(?:jobs?|roles?|positions?|openings?|results?|listings?)\b", lt)
    if not m:
        m = re.search(r"\bonly\s+(\d{1,3})\b", lt)
    if not m:
        m = re.search(r"\btop\s+(\d{1,3})\b", lt)
    if m:
        try:
            num = max(1, min(int(m.group(1)), 50))
        except ValueError:
            pass

    # ── location ───────────────────────────────────────────────────────────
    location = ""
    # Stop on common terminators / end of string
    m = re.search(
        r"\bin\s+([A-Za-z][\w\s,.-]*?)(?:\s+(?:and|or|with|for|that|which|using|on|at|near|using|please|kindly)\b|[.?!]|$)",
        raw,
        re.I,
    )
    if m:
        location = m.group(1).strip(" ,.\t\n")
        # Reject obviously non-location words
        if location.lower() in {"my resume", "my profile", "the role"} or len(location) > 50:
            location = ""

    # ── query ──────────────────────────────────────────────────────────────
    # Resume-driven phrases → leave query empty for inference.
    vague_resume = (
        "related to my resume", "based on my resume", "from my resume",
        "matching my resume", "match my resume",
        "matching my profile", "based on my profile", "for my profile",
    )
    if any(v in lt for v in vague_resume):
        return {"query": "", "location": location, "num_results": num}

    # Generic "find work" / "find a job" / "help me find a job" → empty
    generic_phrases = (
        "find work", "find a job", "find me a job", "looking for a job",
        "help me find a job", "i want to find a job", "i need a job",
        "find job", "find me job", "find some jobs", "find me some jobs",
    )
    if any(p in lt for p in generic_phrases) and not re.search(r"\bin\s+\w", lt):
        return {"query": "", "location": location, "num_results": num}

    # Try to pull a role from "find <role> jobs"
    m = re.search(r"\bfind\s+(?:me\s+)?(.+?)\s+jobs?\b", raw, re.I)
    role = ""
    if m:
        role = m.group(1).strip()
        # Strip leading count + adjectives (counts, "only N", "top N")
        role = re.sub(r"^\d+\s+", "", role).strip()
        role = re.sub(
            r"^(?:only|top|just|any|some|the|a|an|all|more|few|several)\s+",
            "",
            role,
            flags=re.I,
        ).strip()
        # Strip generic temporal/quality adjectives that aren't actual roles
        role = re.sub(
            r"^(?:latest|new|newest|recent|fresh|current|best|good|great|hot|trending|available|open)\s+",
            "",
            role,
            flags=re.I,
        ).strip()
        # Strip any trailing count token, e.g. "10" left over from "latest 10 jobs"
        role = re.sub(r"^\d+\s*$", "", role).strip()
        role = re.sub(r"\s+\d+$", "", role).strip()
        # Words that on their own are NOT a role — treat as "infer from resume"
        GENERIC_NON_ROLES = {
            "", "job", "jobs", "work", "role", "roles", "position", "positions",
            "opening", "openings", "vacancy", "vacancies", "opportunity",
            "opportunities", "career", "careers", "latest", "new", "recent",
            "best", "good", "any", "some", "more", "available", "open",
        }
        if role.lower() in GENERIC_NON_ROLES or len(role) < 2:
            role = ""

    return {"query": role, "location": location, "num_results": num}


# Agent registry (by name) for reference/diagnostics; routing uses capabilities
AGENTS = {
    "planner": _make_planner(),
    "task":    task_agent,
    "tool":    tool_agent,
    "api":     api_agent,
    "search":  search_agent,
}

# ──────────────────────────────────────────────────────────────────────────────
# Advanced Orchestration (Import after agents are defined)
# ──────────────────────────────────────────────────────────────────────────────

from core.orchestration import (
    ReActOrchestrator,
    AgentCollaborator, 
    SupervisorOrchestrator,
    WorkflowOrchestrator,
    ParallelExecutor
)

# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator with Advanced Communication Patterns
# ──────────────────────────────────────────────────────────────────────────────

class JARVISAssistant:
    """
    Main orchestrator for JARVIS with multiple execution modes:
    
    1. SIMPLE MODE (default): Single-hop planner -> agent -> response
    2. REACT MODE: Multi-step reasoning for complex queries
    3. PARALLEL MODE: Multiple capabilities at once
    4. COLLABORATIVE MODE: Agents can invoke other agents
    5. SUPERVISED MODE: Quality control on outputs
    6. MEMORY-AWARE: Uses conversation history and semantic memory
    
    The orchestrator automatically selects the appropriate mode based on query complexity.
    """
    
    def __init__(
        self,
        enable_react: bool = True,
        enable_parallel: bool = True,  # Parallel execution for multi-capability queries
        enable_collaboration: bool = True,
        enable_supervision: bool = False,  # Off by default to save latency
        enable_memory: bool = True,  # Memory system for context
        react_confidence_threshold: float = 0.5,
        parallel_timeout: float = 30.0  # Timeout for parallel execution
    ):
        self.agents = AGENTS
        self.capabilities = list_capabilities()
        
        # Configuration
        self.enable_react = enable_react
        self.enable_parallel = enable_parallel
        self.enable_collaboration = enable_collaboration
        self.enable_supervision = enable_supervision
        self.enable_memory = enable_memory
        self.react_confidence_threshold = react_confidence_threshold
        
        # Memory system
        self.memory = get_memory() if enable_memory else None
        self._current_session_id = None
        
        # Initialize advanced orchestrators
        self.react_orchestrator = ReActOrchestrator(
            model_client=model_client,
            max_iterations=5
        ) if enable_react else None
        
        # Parallel executor for multi-capability queries
        self.parallel_executor = ParallelExecutor(
            timeout=parallel_timeout
        ) if enable_parallel else None
        
        self.collaborator = AgentCollaborator(
            agents=AGENTS
        ) if enable_collaboration else None
        
        self.supervisor = SupervisorOrchestrator(
            model_client=model_client,
            enable_review=enable_supervision
        ) if enable_supervision else None
        
        # Combined workflow orchestrator
        self.workflow = WorkflowOrchestrator(
            model_client=model_client,
            agents=AGENTS,
            enable_react=enable_react,
            enable_collaboration=enable_collaboration,
            enable_supervision=enable_supervision,
            react_threshold=react_confidence_threshold
        )

    def _should_use_react(self, message: str, decision: PlannerDecision = None) -> bool:
        """
        Determine if ReAct multi-step reasoning should be used.
        
        Priority order:
        1. Planner's explicit mode field (if set to "react")
        2. Low confidence from planner (below threshold)
        3. Complexity indicators in the message
        """
        if not self.enable_react:
            return False
        
        # Priority 1: Planner explicitly requested ReAct mode
        if decision and decision.mode == ExecutionMode.REACT:
            return True
        
        # Priority 2: Low confidence triggers ReAct
        if decision and decision.confidence < self.react_confidence_threshold:
            return True
        
        # Priority 3: Check for complexity indicators (fallback heuristics)
        complexity_triggers = [
            "step by step", "detailed analysis", "compare",
            "explain how", "explain why", "multiple", 
            "research", "investigate", "analyze", "vs", "versus"
        ]
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in complexity_triggers)

    async def process_request(self, message: str, request_id: str | None = None) -> str:
        """
        Entry point for 'auto' mode with advanced orchestration.
        
        Flow:
        1) Ask the Planner (one turn)
        2) If planner answers directly (no JSON):
             - If factual/time-sensitive -> force search.web
             - Else return direct answer
        3) Parse PlannerDecision JSON
        4) Check if ReAct is needed (complex query or low confidence)
        5) Execute using appropriate pattern
        6) Apply supervision if enabled
        7) Return final output
        """
        tracer = get_tracer(request_id)
        task = Task(message=message)
        
        # Start request trace
        tracer.start_request(message)

        # ── FAST-PATH: deterministic job-intent override ───────────────────
        # The LLM planner sometimes routes job requests to resume agent which
        # then refuses to actually search. Short-circuit straight to job.search.
        if _looks_job_intent(message):
            hints = _parse_job_query(message)
            try:
                from core.capabilities import resolve as _resolve_cap
                resolved = _resolve_cap("job.search")
                if resolved is not None:
                    _, handler = resolved
                    tracer.route(
                        "orchestrator",
                        f"⚡ Fast-route to job.search (job intent) hints={hints}",
                    )
                    result = await handler(**hints)
                    if isinstance(result, dict) and result.get("response"):
                        return result["response"]
            except Exception as ex:
                tracer.thought("job", f"Fast-route failed, falling back to planner: {ex}")

        # 1) Ask the Planner — fresh agent every call to avoid history bleed
        tracer.thought("planner", f"Analyzing user query: '{message[:60]}...'")
        with tracer.span("planner.call", message_preview=message[:60]):
            planner_team = RoundRobinGroupChat([_make_planner()], max_turns=1)
            buffer = ""
            async for msg in planner_team.run_stream(task=message):
                if getattr(msg, "content", None):
                    buffer += msg.content

        # 2) If planner answered directly (no JSON)...
        s, e = buffer.find("{"), buffer.rfind("}")
        if s == -1 or e == -1 or e <= s:
            # Safety net: if user clearly asked for Telegram, force Telegram domain agent
            if _looks_telegram_intent(message):
                with tracer.span("override.telegram_agent"):
                    router = get_router()
                    telegram_domain = router.get_agent("telegram")
                    if telegram_domain:
                        tracer.route("orchestrator", "🧠 Overriding to Telegram Agent (explicit telegram intent)")
                        try:
                            return await telegram_domain.execute(message, request_id)
                        except Exception as e:
                            tracer.thought("telegram", f"Telegram agent failed: {e}")
                            return buffer
                    return buffer

            # Safety net: if factual or time-sensitive -> route to SEARCH DOMAIN AGENT
            if _looks_factual_question(message) or _looks_time_sensitive(message):
                with tracer.span("override.search_agent"):
                    router = get_router()
                    search_domain = router.get_agent("search")
                    if search_domain:
                        tracer.route("orchestrator", "🧠 Overriding to Search Agent (factual/time-sensitive)")
                        try:
                            return await search_domain.execute(message, request_id)
                        except Exception as e:
                            tracer.thought("search", f"Search agent failed: {e}")
                            return buffer
                    return buffer  # search agent not registered
            # Otherwise accept the planner's direct answer
            with tracer.span("planner.direct_answer"):
                return buffer

        # 3) Parse planner JSON and validate against PlannerDecision
        with tracer.span("planner.parse"):
            plan_json = buffer[s:e+1]
            try:
                raw = json.loads(plan_json)
                decision = PlannerDecision(**raw)
                # Log planner's decision with mode
                tracer.decision("planner", 
                    f"Capability: {decision.capability} | Mode: {decision.mode.value} | "
                    f"Confidence: {decision.confidence:.2f} | Reasoning: {decision.reasoning or 'N/A'}"
                )
            except Exception:
                return buffer  # invalid JSON - use planner text

        # ══════════════════════════════════════════════════════════════════
        # TRUE AGENT ARCHITECTURE: ALL queries route to domain agents
        # Each domain agent is an LLM that thinks and decides which tools
        # to call. This is the core of the multi-agent system.
        # ══════════════════════════════════════════════════════════════════
        
        agent_name = raw.get("agent")
        agent_task = raw.get("task", message)
        router = get_router()

        # ── Safety override: if user clearly wants jobs, force job agent ───
        # Prevents planner from sending job queries to resume.auto_detect.
        if _looks_job_intent(message) and agent_name != "job":
            tracer.route(
                "orchestrator",
                f"⚠️ Planner picked '{agent_name}' but user asked for jobs — overriding to job.search",
            )
            try:
                from core.capabilities import resolve as _resolve_cap
                resolved = _resolve_cap("job.search")
                if resolved is not None:
                    _, handler = resolved
                    hints = _parse_job_query(message)
                    result = await handler(**hints)
                    if isinstance(result, dict) and result.get("response"):
                        return result["response"]
            except Exception as ex:
                tracer.thought("job", f"Job override failed: {ex}")

        # If planner specified an agent, route to it
        if agent_name:
            domain_agent = router.get_agent(agent_name)
            
            if domain_agent:
                tracer.route("orchestrator", f"🧠 Routing to {agent_name.upper()} Agent (TRUE AGENT)")
                with tracer.span(f"domain_agent.{agent_name}", task=agent_task[:80]):
                    try:
                        output = await domain_agent.execute(agent_task, request_id)
                        
                        # Save to memory
                        if self.enable_memory and self.memory:
                            try:
                                self.memory.remember_interaction(
                                    user_message=message,
                                    assistant_response=output,
                                    session_id=self._current_session_id,
                                    capability=f"agent.{agent_name}",
                                    confidence=decision.confidence
                                )
                            except Exception:
                                pass
                        
                        return output
                    except Exception as e:
                        tracer.thought(agent_name, f"Domain agent error: {e}")
                        # Try fallback agent if specified
                        fallback_name = raw.get("fallback") or decision.fallback
                        if fallback_name and fallback_name != agent_name:
                            fallback_agent = router.get_agent(fallback_name)
                            if fallback_agent:
                                tracer.route("orchestrator", f"🔄 Fallback to {fallback_name.upper()} Agent")
                                try:
                                    return await fallback_agent.execute(agent_task, request_id)
                                except Exception:
                                    pass
        
        # ══════════════════════════════════════════════════════════════════
        # SMART FALLBACK: Infer domain agent from capability name
        # Map capabilities to domain agents for TRUE AGENT routing
        # ══════════════════════════════════════════════════════════════════
        
        capability = decision.capability
        
        # Map capabilities to domain agents
        capability_to_agent = {
            # Search capabilities
            "search.web": "search", "search.wikipedia": "search",
            # API capabilities
            "weather.read": "api", "news.read": "api", "news.fetch": "api",
            "stock.read": "api", "stock.price": "api",
            # Task capabilities
            "todo.add": "task", "todo.list": "task", "todo.done": "task", "todo.clear": "task",
            # Gmail capabilities
            "gmail.inbox": "gmail", "gmail.read": "gmail", "gmail.search": "gmail",
            "gmail.draft": "gmail", "gmail.reply": "gmail", "gmail.send": "gmail",
            # Code capabilities
            "code.execute": "code",
            # Telegram capabilities
            "telegram.send_message": "telegram",
            "telegram.get_updates": "telegram",
            "telegram.get_chat_info": "telegram",
            "telegram.send_alert": "telegram",
        }
        
        inferred_agent = capability_to_agent.get(capability)
        if inferred_agent:
            domain_agent = router.get_agent(inferred_agent)
            if domain_agent:
                tracer.route("orchestrator", f"🧠 Inferred routing to {inferred_agent.upper()} Agent from capability '{capability}'")
                with tracer.span(f"domain_agent.{inferred_agent}", task=message[:80]):
                    try:
                        return await domain_agent.execute(message, request_id)
                    except Exception as e:
                        tracer.thought(inferred_agent, f"Inferred agent failed: {e}")
        
        # ══════════════════════════════════════════════════════════════════
        # LEGACY FALLBACK: Only if no domain agent available
        # This path should rarely be used in TRUE AGENT architecture
        # ══════════════════════════════════════════════════════════════════

        # Confidence/time gate: if planner wants to answer directly with low confidence, force search
        if decision.capability == "planner" and (decision.confidence is None or decision.confidence < 0.7):
            with tracer.span("override.planner_low_conf_to_search", confidence=decision.confidence):
                # Route to search domain agent
                search_domain = router.get_agent("search")
                if search_domain:
                    tracer.route("orchestrator", "🔄 Low confidence → routing to SEARCH Agent")
                    try:
                        return await search_domain.execute(message, request_id)
                    except Exception:
                        pass

        # 4) Check execution mode from planner
        
        # 4a) PARALLEL MODE: Execute multiple capabilities at once
        if decision.mode == ExecutionMode.PARALLEL and decision.parallel_capabilities and self.enable_parallel:
            tracer.route("orchestrator", f"Using PARALLEL mode ({len(decision.parallel_capabilities)} capabilities)")
            with tracer.span("parallel.execute", count=len(decision.parallel_capabilities)):
                try:
                    # Convert to CapabilityCall objects
                    caps = [
                        CapabilityCall(
                            capability=c.get("capability") if isinstance(c, dict) else c.capability,
                            inputs=c.get("inputs", {}) if isinstance(c, dict) else c.inputs,
                            label=c.get("label") if isinstance(c, dict) else c.label
                        )
                        for c in decision.parallel_capabilities
                    ]
                    
                    results = await self.parallel_executor.execute(caps, request_id)
                    output = self.parallel_executor.format_results(results, message)
                    
                    # Apply supervision if enabled
                    if self.enable_supervision and self.supervisor:
                        review = await self.supervisor.review_output(
                            task_id=task.id,
                            agent_name="parallel",
                            original_query=message,
                            agent_output=output,
                            request_id=request_id
                        )
                        output = self.supervisor.get_final_output(review)
                    
                    # Save to memory
                    if self.enable_memory and self.memory:
                        try:
                            self.memory.remember_interaction(
                                user_message=message,
                                assistant_response=output,
                                session_id=self._current_session_id,
                                capability="parallel",
                                confidence=decision.confidence
                            )
                        except Exception:
                            pass
                    
                    return output
                except Exception as e:
                    tracer.thought("parallel", f"Parallel execution failed: {e} - falling back to single-hop")
                    # Fall through to single-hop

        # 4b) REACT MODE: Multi-step reasoning
        use_react = self._should_use_react(message, decision)
        if use_react:
            tracer.route("orchestrator", f"Using ReAct mode (planner mode={decision.mode.value}, conf={decision.confidence:.2f})")
            with tracer.span("react.execute", confidence=decision.confidence, mode=decision.mode.value):
                try:
                    output = await self.react_orchestrator.execute(message, request_id)
                    # Apply supervision if enabled
                    if self.enable_supervision and self.supervisor:
                        review = await self.supervisor.review_output(
                            task_id=task.id,
                            agent_name="react",
                            original_query=message,
                            agent_output=output,
                            request_id=request_id
                        )
                        output = self.supervisor.get_final_output(review)
                    return output
                except Exception as e:
                    tracer.thought("react", f"ReAct failed: {e} - falling back to single-hop")
                    pass  # Fall back to single-hop if ReAct fails
        else:
            tracer.route("orchestrator", f"Using single-hop mode -> {decision.capability}")

        # 5) Resolve capability -> (agent_name, handler)
        with tracer.span(
            "capability.resolve",
            capability=decision.capability,
            fallback=decision.fallback,
            confidence=decision.confidence,
        ):
            resolved = resolve(decision.capability)

        if not resolved:
            if decision.fallback:
                fb = resolve(decision.fallback)
                if fb:
                    agent_name, fb_handler = fb
                    with tracer.span("handler.exec.fallback", agent=agent_name):
                        return await (fb_handler(**decision.inputs) if decision.inputs else fb_handler())
            return buffer

        agent_name, handler = resolved

        # 6) Execute capability handler (with graceful error handling)
        try:
            with tracer.span("handler.exec", agent=agent_name, capability=decision.capability):
                output = await (handler(**decision.inputs) if decision.inputs else handler())
        except TypeError:
            with tracer.span("handler.exec.retry_noargs", agent=agent_name):
                try:
                    output = await handler()
                except Exception:
                    output = buffer
        except Exception as e:
            if decision.fallback:
                fb = resolve(decision.fallback)
                if fb:
                    _, fb_handler = fb
                    with tracer.span("handler.exec.fallback_error", error=str(e)):
                        try:
                            output = await (fb_handler(**decision.inputs) if decision.inputs else fb_handler())
                        except Exception:
                            output = f"❌ Error: {e}"
                else:
                    output = f"❌ Error: {e}"
            else:
                output = f"❌ Error: {e}"

        # 7) Apply supervision if enabled (quality control)
        if self.enable_supervision and self.supervisor:
            with tracer.span("supervisor.review", agent=agent_name):
                review = await self.supervisor.review_output(
                    task_id=task.id,
                    agent_name=agent_name,
                    original_query=message,
                    agent_output=output,
                    request_id=request_id
                )
                output = self.supervisor.get_final_output(review)

        # 8) Save to memory (for context-aware future responses)
        if self.enable_memory and self.memory:
            try:
                self.memory.remember_interaction(
                    user_message=message,
                    assistant_response=output,
                    session_id=self._current_session_id,
                    capability=decision.capability,
                    confidence=decision.confidence
                )
            except Exception as e:
                tracer.thought("memory", f"Failed to save to memory: {e}")

        # 9) Wrap a Result (for future logging/metrics) and return its output
        _ = Result(
            task_id=task.id,
            status="ok",
            output=output,
            artifacts=[Artifact(kind="planner_decision", data=decision.model_dump())],
        )
        return output

    def set_session(self, session_id: str):
        """Set the current session ID for memory tracking"""
        self._current_session_id = session_id
    
    def get_memory_context(self, query: str, max_messages: int = 5) -> str:
        """Get relevant memory context for a query"""
        if not self.enable_memory or not self.memory:
            return ""
        return self.memory.get_context_for_prompt(
            query=query,
            session_id=self._current_session_id,
            max_conversation_messages=max_messages
        )

    async def process_request_react(self, message: str, request_id: str | None = None) -> str:
        """
        Force ReAct multi-step reasoning mode regardless of query complexity.
        Useful for complex research tasks that need step-by-step reasoning.
        """
        if not self.react_orchestrator:
            return "ReAct mode is not enabled."
        
        tracer = get_tracer(request_id)
        with tracer.span("react.forced", query=message[:60]):
            return await self.react_orchestrator.execute(message, request_id)

    async def process_request_parallel(
        self, 
        capabilities: list[dict], 
        request_id: str | None = None
    ) -> dict:
        """
        Force parallel execution with explicit capability list.
        
        Args:
            capabilities: List of {"capability": str, "inputs": dict, "label": str}
            request_id: Optional request ID for tracing
        
        Returns:
            Dict with results from all capabilities
        
        Example:
            await jarvis.process_request_parallel([
                {"capability": "weather.read", "inputs": {"city": "Delhi"}, "label": "Delhi Weather"},
                {"capability": "news.fetch", "inputs": {"topic": "tech"}, "label": "Tech News"}
            ])
        """
        if not self.parallel_executor:
            return {"error": "Parallel execution is not enabled."}
        
        tracer = get_tracer(request_id)
        
        # Convert to CapabilityCall objects
        caps = [
            CapabilityCall(
                capability=c.get("capability"),
                inputs=c.get("inputs", {}),
                label=c.get("label")
            )
            for c in capabilities
        ]
        
        with tracer.span("parallel.forced", count=len(caps)):
            results = await self.parallel_executor.execute(caps, request_id)
            
            return {
                "status": "success",
                "results": [
                    {
                        "capability": r.capability,
                        "label": r.label,
                        "status": r.status,
                        "output": r.output if r.status == "ok" else None,
                        "error": r.error if r.status == "error" else None,
                        "execution_time_ms": r.execution_time_ms
                    }
                    for r in results
                ],
                "formatted": self.parallel_executor.format_results(results)
            }

    async def collaborate(
        self, 
        from_agent: str, 
        to_agent: str, 
        message: str, 
        request_id: str | None = None
    ) -> str:
        """
        Enable agent-to-agent communication.
        One agent can send a message to another agent for collaboration.
        """
        if not self.collaborator:
            return "Collaboration mode is not enabled."
        
        tracer = get_tracer(request_id)
        with tracer.span("collaborate", from_a=from_agent, to_a=to_agent):
            response = await self.collaborator.send_message(
                from_agent=from_agent,
                to_agent=to_agent,
                content=message,
                request_id=request_id
            )
            return response.content

    async def _run_agent_task(self, agent: AssistantAgent, task_msg: str) -> str:
        """Helper: run a single agent one turn with plain text."""
        team = RoundRobinGroupChat([agent], max_turns=1)
        out = ""
        async for msg in team.run_stream(task=task_msg):
            if getattr(msg, "content", None):
                out += msg.content
        return out or "No response"

    async def chat_direct(self, message: str, agent_name: str = "planner", request_id: str | None = None) -> str:
        """Direct chat with a specific agent by name (bypasses planner) with tracing."""
        tracer = get_tracer(request_id)
        if agent_name not in self.agents:
            return f"Agent '{agent_name}' not found. Available: {list(self.agents.keys())}"
        with tracer.span("chat_direct", agent=agent_name):
            return await self._run_agent_task(self.agents[agent_name], message)


# Global instance and simple wrappers
jarvis = JARVISAssistant()

async def get_assistant_response(message: str) -> str:
    return await jarvis.process_request(message)

async def run_conversation(message: str) -> str:
    return await jarvis.process_request(message)

def run_sync_conversation(message: str) -> str:
    return asyncio.run(get_assistant_response(message))
