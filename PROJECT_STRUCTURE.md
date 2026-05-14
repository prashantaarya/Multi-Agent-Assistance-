# 📁 JARVIS Project Structure — File-by-File Responsibility Map

Every file listed below is explained with: **what it does**, **what it owns**, and **what other files depend on it**.

---

## 🗂️ Complete File Tree

```
Multi-Agent-Assistance-/
│
├── 📄 main.py                          ⭐ APPLICATION ENTRY POINT
├── 📄 requirements.txt                 📦 Python dependencies
├── 📄 .env                             🔐 API keys (you create this)
├── 📄 tasks.json                       💾 Persistent to-do list storage
│
├── 📄 README.md                        📖 Project overview & quick start
├── 📄 ARCHITECTURE_GUIDE.md            🏗️ Full architecture & design decisions
├── 📄 PROJECT_STRUCTURE.md             🗂️ This file
├── 📄 QUICK_REFERENCE.md               ⚡ Cheat sheet
│
├── 📁 backend/                         🌐 HTTP API LAYER
│   ├── 📄 __init__.py
│   └── 📄 api.py                       🎯 FastAPI routes & middleware
│
├── 📁 agents/                          🧠 ALL AGENTS LIVE HERE
│   ├── 📄 base_agents.py               🎯 MOST IMPORTANT — orchestrator & planner
│   ├── 📄 search_agent.py              🔍 Web search + Wikipedia + LLM summary
│   ├── 📄 api_agent.py                 🌐 Weather / news / stock capabilities
│   ├── 📄 task_agents.py               📝 To-do list management
│   ├── 📄 tool_agents.py               🔧 Code execution + Wolfram Alpha
│   ├── 📄 api_manager.py               🔌 Raw async HTTP to external APIs
│   ├── 📄 docker_manager.py            🐳 Docker sandbox for code execution
│   ├── 📄 memory.py                    💾 Conversation & semantic memory system
│   └── 📄 planner_agent.py             ⚠️ Legacy stub (NOT used in current routing)
│
├── 📁 core/                            ⚙️ INFRASTRUCTURE — routing, schemas, tracing
│   ├── 📄 capabilities.py              🗂️ Capability Registry (central tool store)
│   ├── 📄 schemas.py                   📐 All typed data contracts
│   ├── 📄 orchestration.py             🔄 ReAct loop, Parallel executor, JSON repair
│   └── 📄 tracing.py                   📡 Chain-of-thought logging per request
│
├── 📁 data/                            💾 EXTERNAL DATA FILES
│   └── 📄 reminders.json               (legacy — tasks now in tasks.json)
│
├── 📁 memory/                          💾 MEMORY STORAGE (active)
│   └── 📁 conversation_logs/           📂 Per-session JSON logs (auto-created)
│       └── session_YYYYMMDD_*.json
│
├── 📁 tools/                           🛠️ UTILITY STUBS (not yet wired)
│   ├── 📄 browser.py                   (future: browser automation)
│   ├── 📄 media.py                     (future: media processing)
│   ├── 📄 news_api.py                  (future: enhanced news)
│   └── 📄 wolfram.py                   (future: Wolfram Alpha standalone)
│
└── 📄 test.ipynb                       🧪 Jupyter notebook for manual testing
```

---

## 📄 Root Level Files

### `main.py` — Application Entry Point
**What it does:** Creates and configures the FastAPI application, sets up logging, validates environment on startup, runs uvicorn.

**Owns:**
- `lifespan()` context manager — startup checks + graceful shutdown
- Logging config (dual output: `jarvis.log` + stdout)
- CORS middleware (allows all origins for dev)
- FastAPI app instance

**Key code:**
```python
app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/api/v1")
uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
```

**Depended on by:** Nothing (it's the top)
**Depends on:** `backend/api.py`

---

### `requirements.txt` — Dependencies

```
autogen-agentchat>=0.4.0   # Agent framework
autogen-core>=0.4.0
autogen-ext[openai]>=0.4.0  # OpenAI-compatible model client
fastapi>=0.104.0            # Web framework
uvicorn[standard]>=0.24.0   # ASGI server
pydantic>=2.0.0             # Data validation
aiohttp>=3.9.0              # Async HTTP (Wikipedia, DDG, APIs)
httpx>=0.25.0               # Used by OpenAI client internals
ddgs>=9.11.4                # Real DuckDuckGo web search
python-dotenv>=1.0.0        # .env loader
aiofiles>=23.0.0            # Async file I/O
pytest>=7.0.0               # Testing (dev only)
pytest-asyncio>=0.21.0      # Async test support (dev only)
```

---

### `.env` — API Keys (you create this)
```env
GROQ_API_KEY=gsk_...            # REQUIRED — Groq LLM API
OPENWEATHER_API_KEY=...         # Optional — weather.read capability
NEWSAPI_KEY=...                 # Optional — news.fetch capability
ALPHAVANTAGE_KEY=...            # Optional — stock.price capability
```

---

## 📁 `backend/` — HTTP API Layer

### `backend/api.py` — FastAPI Router
**What it does:** Defines all HTTP endpoints, validates requests, applies rate limiting, calls `jarvis.process_request()`, returns structured JSON.

**Owns:**
- `POST /ask` — main auto-routing endpoint
- `POST /ask-direct` — bypass planner, direct agent targeting
- `GET /health` — health check
- `GET /agents` — list registered agents
- `GET /tools` — list registered capabilities
- Rate limiter (2s cooldown per client via `_last_request_time`)
- Correlation ID (`rid`) generation for log tracing

**Key flow:**
```
POST /ask → apply_rate_limit() → jarvis.process_request(message) → AskResponse
```

**Depends on:** `agents/base_agents.py` (imports `jarvis` instance), `core/capabilities.py`

---

## 📁 `agents/` — The Brain

### `agents/base_agents.py` — Core Orchestrator ⭐ Most Important
**What it does:** Creates all agents, builds the model client, implements `JARVISAssistant` which is the main routing brain.

**Owns:**
- `model_client` — Groq `OpenAIChatCompletionClient` (`openai/gpt-oss-120b`, temp=0.3)
- All 4 specialist agent instances: `task_agent`, `tool_agent`, `api_agent`, `search_agent`
- `_make_planner()` — factory that creates a **fresh** `AssistantAgent` per request (prevents history bleed across requests)
- `PLANNER_SYSTEM_MESSAGE` — auto-generated from Capability Registry; includes all tools + routing rules + mode examples
- `JARVISAssistant` class:
  - `process_request()` — the main dispatcher: planner → parse JSON → pick mode → execute
  - `_should_use_react()` — decides if ReAct is needed (planner says react, OR confidence low, OR complexity keywords)
- `AGENTS` dict — registry of agent instances by name
- Heuristic guards: `_looks_factual_question()`, `_looks_time_sensitive()`

**Key routing logic:**
```
process_request(message)
  → _make_planner() → LLM → JSON { capability, mode, confidence }
  → mode == "parallel" → ParallelExecutor
  → mode == "react" (or _should_use_react()) → ReActOrchestrator
  → else → resolve(capability) → handler(**inputs)
  → memory.remember_interaction()
```

**Depended on by:** `backend/api.py`
**Depends on:** All agents, `core/capabilities.py`, `core/orchestration.py`, `core/tracing.py`, `agents/memory.py`

---

### `agents/planner_agent.py` — ⚠️ Legacy Stub (NOT active)
**Status:** This file exists from an earlier architecture version. It defines a `PlannerAgent` class using the old AutoGen v0.3 API (`user_proxy`, `llm_config`, `aask`). It is **not imported or used** by anything in the current codebase.

**Current planner:** Lives entirely inside `base_agents.py` via `_make_planner()` and `PLANNER_SYSTEM_MESSAGE`.

---

### `agents/search_agent.py` — Web Search + Knowledge
**What it does:** Handles `search.web` capability. Searches DuckDuckGo (real web via `ddgs`) and Wikipedia, applies relevance guard, summarizes with LLM.

**Owns:**
- Registers `search.web` capability on init
- `search(query, source)` — public capability handler
- `_search_ddgs()` — real web search, top 3 results, runs in thread pool (DDGS is sync)
- `_search_ddg_instant()` — legacy DDG Instant Answer API fallback
- `_search_wikipedia_with_resolution()` — MediaWiki `list=search` → best title → `extracts` API
- `_score_and_pick_title()` — heuristic scorer: penalizes institutions/film/lists, rewards short canonical titles
- `_llm_one_shot(prompt)` — calls `model_client.create()` **directly** (not via AutoGen stream) to avoid prompt leaking
- `_summarize_with_llm()` — combines DDG + Wikipedia text into clean 3-6 sentence answer
- `_simplify_query()` — strips filler words, normalizes ("magadh"→"magadha", "ww2"→"world war ii")
- **Relevance guard** — if 0 meaningful query keywords appear in DDG snippets, discards them

**Depends on:** `core/capabilities.py`, `ddgs`, `aiohttp`, `autogen_agentchat`

---

### `agents/api_agent.py` — External Data Fetcher
**What it does:** Handles real-time data capabilities. Delegates to `api_manager.py` for actual HTTP calls.

**Owns:**
- Registers `weather.read`, `news.fetch`, `stock.price` on init
- `get_weather(city)` → delegates to `APIManager.get_weather()`
- `get_news(topic)` → delegates to `APIManager.get_news()`
- `get_stock(symbol)` → delegates to `APIManager.get_stock()`

**Depends on:** `agents/api_manager.py`, `core/capabilities.py`

---

### `agents/api_manager.py` — Raw API Client
**What it does:** Makes the actual async HTTP calls to external services and formats their responses.

**Owns:**
- `get_weather(city)` → OpenWeatherMap REST API → formatted weather string
- `get_news(topic)` → NewsAPI → formatted headlines list
- `get_stock(symbol)` → Alpha Vantage → formatted price string
- All API keys loaded from `.env`
- `aiohttp.ClientSession` for async HTTP
- Response formatting (emoji, human-readable output)

**Depended on by:** `agents/api_agent.py`
**Depends on:** `aiohttp`, `python-dotenv`

---

### `agents/task_agents.py` — To-Do List Manager
**What it does:** Manages a persistent to-do list stored in `tasks.json`.

**Owns:**
- Registers `todo.add`, `todo.list`, `todo.complete`, `todo.clear` on init
- `_cap_add(description)` — appends task to JSON file
- `_cap_list()` — reads and returns all tasks
- `_cap_complete(task_number)` — marks task done
- `_cap_clear()` — clears all tasks
- File I/O to `tasks.json`

**Depends on:** `core/capabilities.py`

---

### `agents/tool_agents.py` — Code Executor
**What it does:** Executes Python code in a Docker sandbox and handles mathematical queries via Wolfram Alpha.

**Owns:**
- Registers `code.execute`, `wolfram.query` on init
- `execute_code(code)` → `DockerManager.run_code()` → result string
- `wolfram_query(query)` → Wolfram Alpha API

**Depends on:** `agents/docker_manager.py`, `core/capabilities.py`

---

### `agents/docker_manager.py` — Code Sandbox
**What it does:** Manages Docker containers for safe code execution.

**Owns:**
- Container creation with resource limits (CPU, memory, network)
- Code injection and execution
- Output capture (stdout + stderr)
- Automatic container cleanup
- Timeout enforcement

**Depended on by:** `agents/tool_agents.py`
**Depends on:** Docker Python SDK

---

### `agents/memory.py` — Memory System (Active)
**What it does:** Provides conversation memory and semantic memory for context-aware responses.

**Owns:**
- `MemoryMessage` — single message model (`id`, `role`, `content`, `timestamp`)
- `ConversationSession` — session with list of messages, `add_message()`, `get_recent()`
- `ConversationMemory` — manages sessions; `remember_interaction()`, `get_context()`
- `SemanticMemory` — `learn()`, `search()` — pickle-backed semantic store
- `LocalStorageBackend` — saves sessions as JSON in `memory/conversation_logs/`
- `get_memory()` — returns the global memory instance
- All `datetime.now(timezone.utc)` — timezone-aware (no deprecation warnings)

**Depended on by:** `agents/base_agents.py`

---

## 📁 `core/` — Infrastructure Layer

### `core/capabilities.py` — Capability Registry ⭐
**What it does:** The central registry where every agent registers its tools. The Planner reads from this to know what's available.

**Owns:**
- `_REGISTRY` dict — maps `capability_name → ToolSchema`
- `register(capability, agent_name, handler, description, parameters, ...)` — called by every agent on init
- `resolve(capability)` → `(agent_name, handler)` — used by orchestrator to call the right function
- `get_tool(capability)` → `ToolSchema` — used by ReAct for input validation
- `list_tools()` → `List[ToolSchema]` — used to auto-generate planner prompt
- `generate_planner_prompt()` — builds the tools section of the planner's system message

**Depended on by:** All agents, `core/orchestration.py`, `agents/base_agents.py`

---

### `core/schemas.py` — Typed Contracts
**What it does:** Defines all Pydantic models used throughout the system.

**Key models:**

| Model | Used For |
|-------|---------|
| `ToolParameter` | Single parameter definition (name, type, required, enum) |
| `ToolSchema` | Full tool definition compatible with OpenAI/Anthropic function calling |
| `PlannerDecision` | Planner JSON output: `capability`, `mode`, `confidence`, `inputs`, `parallel_capabilities` |
| `ExecutionMode` | Enum: `single`, `react`, `parallel` |
| `ReActPlan` | Full ReAct execution plan with steps list |
| `ReActStep` | Single step: `thought_type`, `thought`, `action`, `action_inputs`, `observation` |
| `ThoughtType` | Enum: `observe`, `think`, `act`, `reflect`, `complete` |
| `CapabilityCall` | Used by `ParallelExecutor`: `capability`, `inputs`, `label` |
| `Task` | Incoming user task wrapper |
| `Result` / `Artifact` | Structured output containers |

**Depended on by:** Nearly every file in the project

---

### `core/orchestration.py` — ReAct + Parallel Execution
**What it does:** The most complex file. Implements the ReAct loop, parallel executor, and all the reliability fixes.

**Owns:**
- `REACT_SYSTEM_PROMPT` — dynamic prompt with `{capabilities}`, `{capability_examples}`, `{previous_actions}` variables
- `ReActOrchestrator`:
  - `execute(query)` — runs up to 5 ReAct iterations
  - `_get_next_step()` — calls ReAct LLM agent, parses JSON
  - `_execute_action()` — resolves capability, validates inputs, calls handler
  - `_try_fill_missing_inputs()` — 3-strategy auto-fill (phrase regex → prepositional regex → Title-Case noun fallback)
  - `_is_duplicate_action()` — checks `_action_history` to prevent loops
  - `_record_action()` — only records **successful** actions (no `Error:` prefix)
  - `_synthesize_answer()` — combines all observations into a final answer when max iterations hit
  - `_safe_parse_json()` — 4-strategy fallback: direct → fix trailing commas → fix smart quotes → regex field extraction
  - `_sanitize_for_context()` — strips bullet chars, curly quotes before injecting into LLM prompt
- `ParallelExecutor`:
  - `execute(caps, request_id)` — runs multiple `CapabilityCall`s concurrently with `asyncio.gather`
  - `format_results()` — combines parallel results into one readable response
- `SupervisorOrchestrator` — quality review pass (disabled by default)
- `AgentCollaborator` — agent-to-agent communication pattern
- `WorkflowOrchestrator` — combines all patterns

**Depended on by:** `agents/base_agents.py`
**Depends on:** `core/capabilities.py`, `core/schemas.py`, `core/tracing.py`

---

### `core/tracing.py` — Request Tracing
**What it does:** Provides structured, visual logging for every request with timing information.

**Owns:**
- `RequestTracer` class — created per request with `request_id`
- `tracer.thought(agent, text)` — logs LLM thinking steps (💭)
- `tracer.decision(agent, text)` — logs routing decisions (✅)
- `tracer.action(agent, text)` — logs capability calls (⚡)
- `tracer.observation(agent, text)` — logs results (👁️)
- `tracer.route(agent, text)` — logs mode selection (──▶)
- `tracer.span(name, **kwargs)` — context manager for timed spans
- `get_tracer(request_id)` — factory used everywhere

**Depended on by:** `agents/base_agents.py`, `core/orchestration.py`

---

## 📁 `memory/conversation_logs/` — Active Session Logs

Auto-created by `LocalStorageBackend`. Each file is named `session_YYYYMMDD_HHMMSS_<hash>.json` and contains the full message history for that session.

---

## 📁 `tools/` — Future Stubs (not yet wired)

| File | Planned Purpose |
|------|----------------|
| `browser.py` | Playwright/Selenium browser automation |
| `media.py` | Image/audio processing |
| `news_api.py` | Enhanced news scraping |
| `wolfram.py` | Wolfram Alpha standalone integration |

---

## 🔗 Import Dependency Graph

```
main.py
  └── backend/api.py
        └── agents/base_agents.py
              ├── agents/search_agent.py  ──┐
              ├── agents/api_agent.py       │   all self-register into
              ├── agents/task_agents.py     │   core/capabilities.py
              ├── agents/tool_agents.py   ──┘   on __init__
              ├── agents/memory.py
              ├── core/capabilities.py
              ├── core/schemas.py
              ├── core/tracing.py
              └── core/orchestration.py
                    ├── core/capabilities.py
                    ├── core/schemas.py
                    └── core/tracing.py

agents/api_agent.py
  └── agents/api_manager.py

agents/tool_agents.py
  └── agents/docker_manager.py
```

---

## 🎨 Responsibility Color Map

```
🟦 Infrastructure
   main.py  |  requirements.txt  |  .env

🟩 HTTP Layer
   backend/api.py

🟨 Orchestration Brain
   agents/base_agents.py  |  core/orchestration.py

🟧 Routing & Contracts
   core/capabilities.py  |  core/schemas.py

🟥 Specialist Agents
   agents/search_agent.py  |  agents/api_agent.py
   agents/task_agents.py   |  agents/tool_agents.py

🟪 Support Modules
   agents/api_manager.py  |  agents/docker_manager.py
   agents/memory.py       |  core/tracing.py

⬜ Legacy / Future
   agents/planner_agent.py (legacy stub, not active)
   tools/*  (stubs, not yet wired)
```

---

## 📖 Reading Order (To Understand the Codebase)

1. `core/schemas.py` — understand the data contracts first
2. `core/capabilities.py` — understand the registry
3. `agents/base_agents.py` — understand orchestration
4. `core/orchestration.py` — understand ReAct/Parallel
5. `agents/search_agent.py` — most complex agent
6. `backend/api.py` — HTTP surface
7. `main.py` — startup
