# 🤖 JARVIS - Multi-Agent AI Assistant

> **J**ust **A** **R**ather **V**ery **I**ntelligent **S**ystem

A production-grade multi-agent AI assistant powered by **AutoGen v0.4**, **FastAPI**, and **Groq's `openai/gpt-oss-120b`**. JARVIS intelligently routes requests across a **Capability Registry** to specialized agents, automatically choosing between three execution modes — **Single**, **ReAct**, and **Parallel** — based on query complexity.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![AutoGen](https://img.shields.io/badge/AutoGen-v0.4-orange.svg)](https://microsoft.github.io/autogen/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

### 🧠 Multi-Agent Architecture
- **Planner Agent** — reads every query, picks capability + execution mode, outputs strict JSON
- **Search Agent** — real web search via `ddgs` (DuckDuckGo) + Wikipedia with LLM summarization
- **API Agent** — weather, news, stock data via external REST APIs
- **Task Agent** — persistent JSON-backed to-do list management
- **Tool Agent** — Python code execution (Docker-sandboxed, optional)

### ⚡ Three Execution Modes
| Mode | When Used | Example Query |
|------|-----------|---------------|
| **Single** | One clear capability | `"weather in Tokyo"` |
| **ReAct** | Multi-step reasoning | `"Compare Mughal and Maratha Empires"` |
| **Parallel** | Multiple independent tasks | `"Weather in Delhi AND tech news"` |

### 🔧 Core Infrastructure
- **Capability Registry** (`core/capabilities.py`) — single source of truth; agents self-register on startup
- **Typed Schemas** (`core/schemas.py`) — OpenAI/Anthropic-compatible `ToolSchema`, `ToolParameter`
- **Tracing System** (`core/tracing.py`) — visual per-request chain-of-thought logs
- **ReAct Orchestrator** (`core/orchestration.py`) — loop-prevention, stuck-detection, auto-fill missing inputs

### 💾 Memory System (Active)
- `ConversationMemory` — remembers context within a session
- `SemanticMemory` — learns from past interactions (pickle-backed)
- `LocalStorageBackend` — saves JSON logs to `memory/conversation_logs/`

### 🛡️ Reliability Built-In
- `_safe_parse_json()` — 4-strategy fallback parser handles malformed LLM JSON
- `_try_fill_missing_inputs()` — auto-extracts city/topic from LLM thought text when `action_inputs: {}`
- Duplicate action guard — prevents infinite ReAct loops
- DDG relevance guard — discards off-topic search results before they reach the LLM
- Rate limiting — 2s cooldown per client

---

## 🏗️ Architecture Overview

```
User HTTP Request  POST /api/v1/ask
        │
        ▼
┌─────────────────────┐
│   FastAPI /ask      │   backend/api.py
│   Rate limit + log  │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────┐
│   JARVISAssistant        │   agents/base_agents.py
│   process_request()      │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────┐
│   Planner (fresh LLM)│   _make_planner() → fresh per request (no history bleed)
│   Reads all tools    │   Auto-generated from Capability Registry
│   → JSON decision    │   { capability, mode, confidence, inputs }
└──────────┬───────────┘
           │
    ┌──────┴─────────────────┐
    │                        │
    ▼                        ▼
mode=single           mode=react / parallel
    │                        │
    ▼                        ▼
Capability Registry    ReActOrchestrator  /  ParallelExecutor
resolve(capability)    core/orchestration.py
    │                        │
    ▼                        ▼
Agent Handler          Capability Registry (called per ReAct step)
(search/api/task/tool)
           │
           ▼
    Memory.remember()
           │
           ▼
    JSON Response → Client
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Groq API key — free at [console.groq.com](https://console.groq.com)
- Docker Desktop *(optional — only needed for `code.execute` capability)*

### Installation

```powershell
# 1. Clone the repo
git clone https://github.com/yourusername/Multi-Agent-Assistance-.git
cd Multi-Agent-Assistance-

# 2. Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file in root
```

```env
# Required
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional — enables weather / news / stock capabilities
OPENWEATHER_API_KEY=your_openweather_key
NEWSAPI_KEY=your_newsapi_key
ALPHAVANTAGE_KEY=your_alphavantage_key
```

```powershell
# 5. Start JARVIS
python main.py

# 6. Open Swagger UI
Start-Process "http://localhost:8000/docs"
```

---

## 💬 Usage Examples

### PowerShell
```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/ask" `
  -ContentType "application/json" `
  -Body '{"message": "tell me about the Maratha Empire"}'
```

### Python
```python
import requests
resp = requests.post(
    "http://localhost:8000/api/v1/ask",
    json={"message": "Compare the Mughal and Maratha Empires"}
)
print(resp.json()["response"])
```

### cURL
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "What is quantum entanglement?"}'
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ask` | POST | Main endpoint — planner auto-routes |
| `/api/v1/ask-direct` | POST | Bypass planner, target a specific agent |
| `/api/v1/health` | GET | Health check |
| `/api/v1/agents` | GET | List all registered agents |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

---

## 🧩 Registered Capabilities

| Capability | Agent | Description |
|------------|-------|-------------|
| `search.web` | search | DDG real search + Wikipedia + LLM summary |
| `weather.read` | api | Current weather for any city |
| `news.fetch` | api | Latest news by topic |
| `stock.price` | api | Stock price lookup |
| `todo.add` | task | Add item to persistent to-do list |
| `todo.list` | task | List all to-do items |
| `todo.complete` | task | Mark item as done |
| `todo.clear` | task | Clear all items |
| `code.execute` | tool | Execute Python code in Docker sandbox |
| `wolfram.query` | tool | Wolfram Alpha math/science queries |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | `openai/gpt-oss-120b` via Groq API |
| Agent Framework | AutoGen v0.4 (`AssistantAgent`, `RoundRobinGroupChat`) |
| Web Framework | FastAPI + Uvicorn |
| Web Search | `ddgs` ≥ 9.11.4 (real DuckDuckGo search) |
| Knowledge Base | Wikipedia MediaWiki API |
| Async HTTP | `aiohttp` |
| Data Validation | Pydantic v2 |
| Code Sandbox | Docker *(optional)* |
| Memory Storage | JSON + pickle (local `memory/` directory) |

---

## 📁 Key Files at a Glance

| File | Role |
|------|------|
| `main.py` | FastAPI app, lifespan hooks, logging setup |
| `backend/api.py` | HTTP router, rate limiting, request/response models |
| `agents/base_agents.py` | `JARVISAssistant`, planner factory, model client, mode router |
| `agents/search_agent.py` | DDG search, Wikipedia, relevance guard, LLM summarization |
| `agents/api_agent.py` | Weather / news / stock capability handlers |
| `agents/task_agents.py` | To-do list (JSON-backed, 4 capabilities) |
| `agents/tool_agents.py` | Code execution, Wolfram Alpha |
| `agents/memory.py` | `ConversationMemory`, `SemanticMemory`, session management |
| `agents/api_manager.py` | Raw async HTTP calls to external APIs |
| `agents/docker_manager.py` | Docker container lifecycle for code execution |
| `core/capabilities.py` | Registry: `register()`, `resolve()`, `list_tools()` |
| `core/schemas.py` | All typed contracts: `ToolSchema`, `PlannerDecision`, `ReActPlan` |
| `core/orchestration.py` | `ReActOrchestrator`, `ParallelExecutor`, JSON repair, auto-fill |
| `core/tracing.py` | Request tracing, visual log output |

---

## 🔧 Configuration Reference

### Model & temperature (`agents/base_agents.py`)
```python
model_client = OpenAIChatCompletionClient(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    create_config={"temperature": 0.3, "max_tokens": 1000}
)
```

### Execution modes (`agents/base_agents.py`)
```python
JARVISAssistant(
    enable_react=True,
    react_confidence_threshold=0.5,   # planner confidence < this → force ReAct
    enable_parallel=True,
    parallel_timeout=30.0,
    enable_supervision=False,          # True adds a quality-review pass (slower)
    enable_memory=True,
)
```

### Server port (`main.py`)
```python
uvicorn.run("main:app", host="127.0.0.1", port=8001)
```

### Rate limit (`backend/api.py`)
```python
REQUEST_COOLDOWN = 2  # seconds between requests per client
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not found` | Create `.env` with `GROQ_API_KEY=gsk_...` in project root |
| `Port 8000 already in use` | `netstat -ano \| findstr :8000` → `Stop-Process -Id <PID> -Force` |
| `Module not found` | `pip install -r requirements.txt` |
| LLM `ConnectionError` | Start server from terminal (inherits corporate proxy env) |
| Wrong DDG results | Relevance guard discards; Wikipedia becomes sole source automatically |
| ReAct missing inputs | Auto-fill extracts value from LLM thought text (3 strategies) |
| ReAct `[LOOP PREVENTED]` | Fixed — failed actions not recorded; retries always allowed |

---

## 🚀 Extending JARVIS — Adding a New Capability

```python
# In your new agent's __init__():
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

register(
    capability="myagent.action",
    agent_name=self.name,
    handler=self.my_handler,           # async def my_handler(self, param: str) -> str
    description="What this does",
    parameters=[
        ToolParameter(name="param", type=ParameterType.STRING,
                      description="Input value", required=True)
    ],
    category="my_category"
)
```

The Planner automatically picks up the new capability at the next request — no manual prompt editing needed.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — overview, quick start, reference |
| [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) | Deep-dive: every component, data flow, design decisions |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | File-by-file responsibility map with code pointers |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Cheat sheet — commands, endpoints, capabilities |

---

## 📝 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
<b>Made with ❤️ — JARVIS Multi-Agent Assistant</b><br>
<a href="ARCHITECTURE_GUIDE.md">Architecture</a> •
<a href="PROJECT_STRUCTURE.md">Structure</a> •
<a href="http://localhost:8000/docs">Swagger UI</a>
</div>

