# 🏗️ JARVIS Architecture Guide

Complete technical reference for every design decision, component, and data flow in JARVIS.

---

## 📋 Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Three Execution Modes](#3-three-execution-modes)
4. [Capability Registry](#4-capability-registry)
5. [Component Deep-Dive](#5-component-deep-dive)
6. [End-to-End Data Flows](#6-end-to-end-data-flows)
7. [ReAct Loop Internals](#7-react-loop-internals)
8. [Reliability & Bug Prevention](#8-reliability--bug-prevention)
9. [Memory System](#9-memory-system)
10. [Technology Stack](#10-technology-stack)
11. [How to Add a New Capability](#11-how-to-add-a-new-capability)

---

## 1. System Overview

JARVIS is a **multi-agent AI system** where a central Planner LLM decides — per request — which specialist agent to call and **how** to call it (single call, multi-step reasoning, or parallel execution).

### Design Principles
| Principle | Implementation |
|-----------|---------------|
| **Stateless routing** | Fresh planner agent per request — no history bleed |
| **Self-registering tools** | Agents register capabilities on init; planner prompt is auto-generated |
| **Graceful degradation** | 4-strategy JSON parser, auto-fill missing inputs, fallback chains |
| **Separation of concerns** | Registry in `core/`, HTTP in `backend/`, agents in `agents/` |
| **Observability** | Every step traced with request ID, timing, agent name |

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HTTP REQUEST                                   │
│              POST /api/v1/ask  {"message": "..."}                    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│  backend/api.py — FastAPI Router                                       │
│  • Generates correlation ID (rid)                                      │
│  • Rate limit: 2s cooldown                                             │
│  • Calls jarvis.process_request(message, rid)                          │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│  agents/base_agents.py — JARVISAssistant.process_request()            │
│                                                                        │
│  1. Create fresh planner: _make_planner()                             │
│  2. Run planner via RoundRobinGroupChat(max_turns=1)                  │
│  3. Parse JSON response → PlannerDecision                             │
│  4. Branch on execution mode ──────────────────────────────────┐      │
└────────────────────────────────────────────────────────────────│──────┘
                                                                 │
                    ┌────────────────────────────────────────────┤
                    │                   │                        │
                    ▼                   ▼                        ▼
         mode = "single"        mode = "react"          mode = "parallel"
                    │                   │                        │
                    ▼                   ▼                        ▼
         resolve(capability)   ReActOrchestrator        ParallelExecutor
         → handler(**inputs)   .execute(query)          .execute(caps)
                    │                   │                        │
                    └──────────────┬────┘                        │
                                   │◄────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Capability Registry      │  core/capabilities.py
                    │  resolve(capability_name) │
                    │  → (agent_name, handler)  │
                    └──────────────┬────────────┘
                                   │
          ┌───────────────┬────────┴──────────┬──────────────────┐
          ▼               ▼                   ▼                  ▼
   SearchAgent        APIAgent           TaskAgent           ToolAgent
   search.web         weather.read       todo.*              code.execute
                       news.fetch                            wolfram.query
                       stock.price
          │               │                   │                  │
          ▼               ▼                   ▼                  ▼
  DDG + Wikipedia    External APIs        tasks.json         Docker container
  LLM summary        (aiohttp)           (JSON R/W)         (code execution)

                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  agents/memory.py         │
                    │  memory.remember_         │
                    │  interaction()            │
                    │  → session JSON log       │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  JSON Response            │
                    │  { response, agent_used,  │
                    │    status, request_id }   │
                    └──────────────────────────┘
```

---

## 3. Three Execution Modes

The Planner outputs a `"mode"` field in its JSON. `JARVISAssistant` uses this (plus heuristics) to pick the execution path.

### Mode 1: Single (default, fastest)
```
Planner → { "mode": "single", "capability": "weather.read", "inputs": {"city": "Tokyo"} }
         → resolve("weather.read") → APIAgent.get_weather("Tokyo")
         → Return result directly
```
**Used when:** One clear capability, no inter-step dependencies.

### Mode 2: ReAct (multi-step reasoning)
```
Planner → { "mode": "react", "confidence": 0.82 }
         → ReActOrchestrator.execute(query)
           Iteration 1: OBSERVE  — understand the task
           Iteration 2: THINK    — plan approach
           Iteration 3: ACT      — call capability A → get result
           Iteration 4: ACT      — call capability B → get result
           Iteration 5: COMPLETE — synthesize final answer
```
**Used when:**
- Planner sets `mode="react"` explicitly
- Planner confidence < 0.5 (threshold)
- Query contains complexity keywords: *compare, research, analyze, vs, step by step*

**Max iterations:** 5 (configurable via `max_iterations`)

### Mode 3: Parallel (concurrent execution)
```
Planner → { "mode": "parallel", "parallel_capabilities": [
              { "capability": "weather.read", "inputs": {"city": "Delhi"}, "label": "Delhi Weather" },
              { "capability": "news.fetch",   "inputs": {"topic": "tech"},  "label": "Tech News" }
           ]}
         → ParallelExecutor.execute([cap1, cap2])
         → asyncio.gather(handler1(...), handler2(...))
         → Format combined results
```
**Used when:** Multiple independent tasks can run simultaneously (saves time).

---

## 4. Capability Registry

The registry (`core/capabilities.py`) is the **single source of truth** for what JARVIS can do.

### How It Works

```
STARTUP:
  TaskAgent.__init__()  → register("todo.add",    handler=self._cap_add, ...)
  TaskAgent.__init__()  → register("todo.list",   handler=self._cap_list, ...)
  APIAgent.__init__()   → register("weather.read", handler=self.get_weather, ...)
  SearchAgent.__init__() → register("search.web", handler=self.search, ...)
  ...
  _REGISTRY = {
    "todo.add":      ToolSchema(agent_name="task", handler=<fn>, parameters=[...]),
    "weather.read":  ToolSchema(agent_name="api",  handler=<fn>, parameters=[...]),
    "search.web":    ToolSchema(agent_name="search", handler=<fn>, parameters=[...]),
    ...
  }

PLANNER PROMPT BUILT:
  generate_planner_prompt()  ← reads _REGISTRY
  → Injects live tool docs into PLANNER_SYSTEM_MESSAGE
  → Planner always has up-to-date tool info

REQUEST:
  resolve("weather.read")  → ("api", <APIAgent.get_weather bound method>)
  get_tool("search.web")   → ToolSchema (for input validation in ReAct)
  list_tools()             → [ToolSchema, ...] (for capability examples)
```

### Tool Schema Format (OpenAI/Anthropic Compatible)
```python
ToolSchema(
    name="search.web",
    agent_name="search",
    description="Search the web for factual information...",
    parameters=[
        ToolParameter(name="query",  type=STRING, required=True),
        ToolParameter(name="source", type=STRING, required=False,
                      enum=["auto","wiki","ddg"], default="auto")
    ],
    examples=[{"query": "Who invented Python"}],
    category="search"
)
```

---

## 5. Component Deep-Dive

### Planner Agent

**Created:** Fresh per request via `_make_planner()` — uses `AssistantAgent` from AutoGen v0.4
**Why fresh:** `RoundRobinGroupChat` accumulates message history on the agent instance; reusing it causes previous queries to bleed into the current one.

**System message includes (auto-generated):**
- All registered tools with parameter specs
- Mode selection rules (single / react / parallel) with examples
- Routing rules: factual questions → search.web, weather → weather.read, etc.
- Examples for each mode
- Small talk handling (direct text, no JSON)

**Output format:**
```json
{
  "capability": "search.web",
  "inputs": {"query": "Maratha Empire"},
  "confidence": 0.95,
  "fallback": null,
  "mode": "single",
  "reasoning": "Factual history question",
  "parallel_capabilities": null
}
```

---

### Search Agent (`agents/search_agent.py`)

The most complex agent. Two data sources, one LLM summarizer.

```
search(query, source="auto")
      │
      ├── _simplify_query()           strip fillers, normalize aliases
      │
      ├── [AUTO mode] run both in parallel:
      │     ├── _search_duckduckgo()
      │     │     └── _search_ddgs()         real web, top 3 results (DDGS.text())
      │     │           └── Relevance guard  discard if 0 query keywords in snippets
      │     │           └── Fallback: _search_ddg_instant()  (Instant Answer API)
      │     │
      │     └── _search_wikipedia_with_resolution()
      │           ├── _wiki_candidate_titles()   MediaWiki list=search → 8 titles
      │           ├── _score_and_pick_title()    heuristic scorer
      │           │     • word overlap +1
      │           │     • exact match +5
      │           │     • starts-with query +3
      │           │     • ≤2 words (canonical) +2
      │           │     • institution keywords -4
      │           │     • media keywords -3
      │           │     • list/disambiguation -5
      │           └── extracts API → 2000 chars
      │
      └── _summarize_with_llm(ddg_text, wiki_text, query)
            └── _llm_one_shot(prompt)
                  └── model_client.create([UserMessage(prompt)])
                        Direct API call — no AutoGen stream → no prompt leaking
```

---

### ReAct Orchestrator (`core/orchestration.py`)

```
execute(query)
  │
  ├── Create ReAct AssistantAgent with dynamic prompt
  │     (capabilities doc + examples + loop prevention rules)
  │
  ├── for iteration in range(max_iterations=5):
  │     │
  │     ├── _get_next_step(context, plan)
  │     │     └── RoundRobinGroupChat([react_agent]).run_stream(context)
  │     │     └── _safe_parse_json(buffer)  ← 4-strategy fallback
  │     │
  │     ├── if is_final → break with final_answer
  │     │
  │     ├── if action:
  │     │     ├── _try_fill_missing_inputs(action, inputs, thought, query)
  │     │     │     Strategy 1: "searching for X" / "looking up X" regex
  │     │     │     Strategy 2: "overview of X" / "about X" regex
  │     │     │     Strategy 3: Title-Case noun phrase fallback
  │     │     │
  │     │     ├── _is_duplicate_action(action, inputs) → skip if seen before
  │     │     │
  │     │     ├── _execute_action(action, inputs)
  │     │     │     ├── resolve(action) → handler
  │     │     │     ├── validate required inputs (gives retry hint if missing)
  │     │     │     └── await handler(**inputs)
  │     │     │
  │     │     └── _record_action() ← ONLY if observation does NOT start with "Error:"
  │     │
  │     ├── _sanitize_for_context(observation)  strip bullets/curly quotes
  │     └── plan.add_step(thought, action, inputs, observation)
  │
  └── if not plan.is_complete → _synthesize_answer(plan)
```

---

### Parallel Executor (`core/orchestration.py`)

```
execute(caps: List[CapabilityCall], request_id)
  │
  ├── for each cap: resolve(cap.capability) → handler
  ├── asyncio.gather(*[handler(**cap.inputs) for cap in caps], timeout=30s)
  └── format_results() → labeled sections per capability
```

---

## 6. End-to-End Data Flows

### Flow A: Simple factual search — `"tell me about the Maratha Empire"`

```
POST /ask {"message": "tell me about the Maratha Empire"}
  │
  ▼ backend/api.py
  rid = "383063..."
  jarvis.process_request(message, rid)
  │
  ▼ agents/base_agents.py
  planner = _make_planner()  ← fresh AssistantAgent
  planner runs → JSON:
  { "capability": "search.web", "mode": "single", "confidence": 0.95 }
  │
  ▼ mode = single
  resolve("search.web") → ("search", SearchAgent.search)
  await SearchAgent.search(query="Maratha Empire", source="auto")
  │
  ▼ agents/search_agent.py
  cleaned = _simplify_query("tell me about the Maratha Empire")
           → "maratha empire"

  [parallel]
  ├── DDGS.text("maratha empire", max_results=3)
  │     → 3 snippets (689 chars)
  │     → relevance guard: "maratha" found ✅  → keep
  │
  └── Wikipedia list=search("maratha empire", limit=8)
        → candidates: ["Maratha Empire", "Maratha", ...]
        → _score_and_pick_title() → "Maratha Empire" (score=8)
        → extracts API → 1205 chars

  _summarize_with_llm(ddg_text, wiki_text, "tell me about the Maratha Empire")
  → model_client.create([UserMessage(prompt)])
  → "The Maratha Empire (also called the Maratha Confederacy)..."
  │
  ▼ back in base_agents.py
  memory.remember_interaction(message, response)
  → saves to memory/conversation_logs/session_*.json
  │
  ▼ backend/api.py
  return AskResponse(response="📘 Summary:\nThe Maratha Empire...", status="success")
```

---

### Flow B: Multi-step comparison — `"Compare Mughal and Maratha Empires"`

```
Planner → { "mode": "react", "confidence": 0.82 }
_should_use_react() → True (planner said react)
ReActOrchestrator.execute("Compare Mughal and Maratha Empires")
  │
  Iter 1: OBSERVE — "Need data on both empires"
  Iter 2: THINK   — "Will search Mughal first, then Maratha"
  Iter 3: ACT     — action="search.web", inputs={"query":"Mughal Empire overview"}
                    → execute → 1203 chars from Wikipedia
                    → _record_action() ✅
  Iter 4: ACT     — action="search.web", inputs={}  ← LLM forgot inputs!
                    → _try_fill_missing_inputs():
                        thought="Fetching Maratha Empire..."
                        Strategy 3: Title-Case → "Maratha Empire" ✅
                        inputs filled = {"query": "Maratha Empire"}
                    → execute → 1180 chars from Wikipedia
                    → _record_action() ✅
  Iter 5: COMPLETE — is_final=True
                     final_answer = "### Mughal vs Maratha...\n\n**Chronology**..."
  │
  return final_answer
```

---

### Flow C: Parallel tasks — `"Weather in Delhi and latest tech news"`

```
Planner → {
  "mode": "parallel",
  "parallel_capabilities": [
    {"capability": "weather.read", "inputs": {"city": "Delhi"}, "label": "Delhi Weather"},
    {"capability": "news.fetch",   "inputs": {"topic": "technology"}, "label": "Tech News"}
  ]
}

ParallelExecutor.execute([weather_cap, news_cap])
→ asyncio.gather(
    APIAgent.get_weather("Delhi"),
    APIAgent.get_news("technology")
  )
→ Both run concurrently → combined in ~same time as slowest call
→ format_results() → "**Delhi Weather**\n...\n\n**Tech News**\n..."
```

---

## 7. ReAct Loop Internals

### Prompt Structure
The ReAct agent's system prompt is dynamic — filled at creation time:

```python
REACT_SYSTEM_PROMPT.format(
    capabilities=_get_capabilities_doc(),       # all tools with types
    capability_examples=_get_capability_examples(),  # required inputs per tool
    max_iterations=5,
    current_step=iteration,
    previous_actions=_format_action_history()   # loop prevention
)
```

### Thought Types
| Type | Purpose | Sets action? |
|------|---------|-------------|
| `observe` | Understand what's needed | No |
| `think` | Plan approach | No |
| `act` | Execute a capability | Yes |
| `reflect` | Evaluate result | No |
| `complete` | Final synthesis | No (sets `is_final=True`) |

### Loop Prevention
```python
_action_history: List[Tuple[str, str]]  # (action_name, json_inputs)

_is_duplicate_action(action, inputs):
    inputs_str = json.dumps(inputs, sort_keys=True)
    return (action, inputs_str) in _action_history

_record_action(action, inputs):
    # Only called if observation.startswith("Error:") is False
    _action_history.append((action, inputs_str))
```

### Auto-Fill Missing Inputs
When LLM sends `action_inputs: {}` but mentions the value in its thought:

```python
_try_fill_missing_inputs(action, {}, thought="Fetching Maratha Empire overview", query):
    # For param named "query":
    # Strategy 1: "searching for X" / "looking up X" → no match
    # Strategy 2: "overview of X" / "about X"        → no match
    # Strategy 3: Title-Case noun phrases in thought  → "Maratha Empire" ✅
    return {"query": "Maratha Empire"}
```

---

## 8. Reliability & Bug Prevention

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| JSON parse crash | LLM adds markdown fences, trailing commas, curly quotes | `_safe_parse_json()` — 4 fallback strategies |
| Prompt leaking into response | AutoGen `run_stream` yields user message too | `model_client.create()` directly in `_llm_one_shot()` |
| Planner history bleed | Shared `AssistantAgent` accumulates messages | `_make_planner()` factory — fresh per request |
| ReAct empty `action_inputs` | LLM mentions value in thought but forgets JSON | `_try_fill_missing_inputs()` — 3 extraction strategies |
| ReAct infinite loop | Failed action recorded → next attempt blocked | Only record if no `"Error:"` prefix |
| Wrong DDG results | Geo-redirect / noise (e.g., HMY stock for "Maratha") | Relevance guard: discard if 0 keywords match |
| Wrong Wikipedia article | Title scorer picked institution page | Institution penalty (-4), canonical bonus (+2) |
| `datetime.utcnow()` warnings | Python 3.12 deprecation | `datetime.now(timezone.utc)` everywhere |
| `RuntimeWarning` from ddgs | Old `duckduckgo_search` package still installed | `warnings.filterwarnings("ignore", ...)` at module load |
| Bullet chars corrupt LLM JSON | `•` in weather data injected raw into prompt | `_sanitize_for_context()` replaces `•` → `-` |

---

## 9. Memory System

```
agents/memory.py
│
├── LocalStorageBackend
│     • Saves/loads ConversationSession as JSON
│     • Path: memory/conversation_logs/session_YYYYMMDD_HHMMSS_<hash>.json
│
├── ConversationMemory
│     • Manages sessions by session_id
│     • remember_interaction(user_msg, assistant_resp) → adds 2 MemoryMessages
│     • get_context(n=10) → last N messages as prompt-friendly string
│
└── SemanticMemory
      • learn(text, metadata) → stores entry with timestamp
      • search(query, n=5) → finds semantically similar past entries
      • Backed by pickle file (vector_store.pkl)
```

Each session is a JSON file:
```json
{
  "session_id": "session_20260318_113134_4a7882",
  "messages": [
    {"role": "user",      "content": "tell me about Maratha Empire", "timestamp": "..."},
    {"role": "assistant", "content": "📘 Summary: The Maratha Empire...", "timestamp": "..."}
  ]
}
```

---

## 10. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| LLM | Groq API — `openai/gpt-oss-120b` | — | All LLM inference |
| Agent Framework | AutoGen | v0.4 | `AssistantAgent`, `RoundRobinGroupChat` |
| Web Framework | FastAPI + Uvicorn | ≥0.104 | HTTP API |
| Async HTTP | aiohttp | ≥3.9 | Wikipedia, DDG Instant, external APIs |
| Web Search | ddgs | ≥9.11.4 | Real DuckDuckGo search |
| Data Validation | Pydantic | v2 | All schemas and models |
| Environment | python-dotenv | ≥1.0 | `.env` loading |
| Code Sandbox | Docker | latest | Safe code execution (optional) |
| Memory | JSON + pickle | stdlib | Local session/semantic storage |

---

## 11. How to Add a New Capability

### Step 1: Create or extend an agent

```python
# In your agent's __init__():
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

register(
    capability="translate.text",
    agent_name=self.name,
    handler=self.translate,
    description="Translate text from one language to another",
    parameters=[
        ToolParameter(name="text", type=ParameterType.STRING,
                      description="Text to translate", required=True),
        ToolParameter(name="target_lang", type=ParameterType.STRING,
                      description="Target language code (e.g. 'fr', 'es')", required=True)
    ],
    category="language",
    examples=[{"text": "Hello world", "target_lang": "fr"}]
)
```

### Step 2: Implement the handler

```python
async def translate(self, text: str, target_lang: str) -> str:
    # Your implementation here
    return f"Translated to {target_lang}: ..."
```

### Step 3: Register the agent instance in `base_agents.py`

```python
from agents.my_agent import MyAgent
my_agent = MyAgent(name="my", model_client=model_client)

AGENTS = {
    ...,
    "my": my_agent,
}
```

That's it. The Planner automatically sees `translate.text` in its next request — no prompt editing needed.

---

*For file-by-file breakdown, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)*
*For quick commands and cheat sheet, see [QUICK_REFERENCE.md](QUICK_REFERENCE.md)*
