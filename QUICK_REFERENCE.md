# ⚡ JARVIS Quick Reference Cheat Sheet

Everything you need in one place — no reading required.

---

## 📁 File Map (Who Does What)

```
main.py                   ⭐ Start here — FastAPI app, uvicorn, logging
requirements.txt          📦 All Python dependencies
.env                      🔐 API keys (you create this)
tasks.json                💾 To-do list storage (auto-created)

backend/
  api.py                  🌐 HTTP routes, rate limiting, request/response

agents/
  base_agents.py          🧠 BRAIN — JARVISAssistant, model client, planner, mode router
  search_agent.py         🔍 DDG real search + Wikipedia + LLM summary
  api_agent.py            🌤️  Weather / news / stock capability handlers
  api_manager.py          🔌 Raw aiohttp calls to OpenWeatherMap / NewsAPI / AlphaVantage
  task_agents.py          📝 todo.add / todo.list / todo.complete / todo.clear
  tool_agents.py          🔧 code.execute (Docker) / wolfram.query
  docker_manager.py       🐳 Docker sandbox lifecycle
  memory.py               💾 ConversationMemory + SemanticMemory (session JSON logs)
  planner_agent.py        ⚠️  Legacy stub — NOT active in current routing

core/
  capabilities.py         🗂️  Capability Registry — register(), resolve(), list_tools()
  schemas.py              📐 All typed models: ToolSchema, PlannerDecision, ReActPlan...
  orchestration.py        🔄 ReActOrchestrator, ParallelExecutor, JSON repair, auto-fill
  tracing.py              📡 Per-request chain-of-thought logging

memory/
  conversation_logs/      📂 Per-session JSON logs (auto-created)
```

---

## 🚀 Start / Stop Commands

```powershell
# Install dependencies
pip install -r requirements.txt

# Start server
python main.py

# Kill server on port 8000
netstat -ano | findstr :8000   # find PID
Stop-Process -Id <PID> -Force

# Quick kill + restart
netstat -ano | findstr ":8000" | ForEach-Object {
    $p = ($_ -split '\s+')[-1]
    if ($p -match '^\d+$') { Stop-Process -Id $p -Force }
}
python main.py
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ask` | POST | Main — auto planner routing |
| `/api/v1/ask-direct` | POST | Direct to specific agent |
| `/api/v1/health` | GET | Health check |
| `/api/v1/agents` | GET | List agents |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc UI |

---

## 📬 Request Format

```json
POST /api/v1/ask
{
  "message": "your query here",
  "agent": "auto"
}
```

```json
Response:
{
  "response": "...",
  "agent_used": "multi-agent",
  "status": "success",
  "request_id": "abc123..."
}
```

---

## 💬 Test Queries by Mode

### Single Mode (fastest)
```
"What is the weather in Tokyo?"
"Add task Buy groceries"
"List my tasks"
"Latest tech news"
```

### ReAct Mode (multi-step)
```
"Compare the Mughal Empire and Maratha Empire"
"Who is Elon Musk and what companies does he run?"
"What is quantum entanglement and give me its applications"
"Research the causes of World War 2"
```

### Parallel Mode (concurrent)
```
"Get weather in Delhi and London simultaneously"
"Tell me about the Roman Empire AND the Ottoman Empire"
"Add task Call mom and also add task Buy milk"
"Get tech news and also weather in Mumbai"
```

---

## 🧩 All Capabilities

| Capability | Agent | Required Inputs | Optional |
|------------|-------|----------------|----------|
| `search.web` | search | `query: str` | `source: "auto"\|"wiki"\|"ddg"` |
| `weather.read` | api | `city: str` | — |
| `news.fetch` | api | `topic: str` | — |
| `stock.price` | api | `symbol: str` | — |
| `todo.add` | task | `description: str` | — |
| `todo.list` | task | — | — |
| `todo.complete` | task | `task_number: int` | — |
| `todo.clear` | task | — | — |
| `code.execute` | tool | `code: str` | — |
| `wolfram.query` | tool | `query: str` | — |

---

## 🔑 .env File

```env
# Required
GROQ_API_KEY=gsk_your_key_here

# Optional
OPENWEATHER_API_KEY=your_key    # https://openweathermap.org/api  (free)
NEWSAPI_KEY=your_key            # https://newsapi.org             (free)
ALPHAVANTAGE_KEY=your_key       # https://www.alphavantage.co     (free)
```

**Get Groq key (free, fast):** https://console.groq.com

---

## 🔄 Execution Mode Logic

```
Planner outputs mode field
        │
        ├── mode = "parallel"  AND parallel_capabilities present
        │         → ParallelExecutor (concurrent asyncio.gather)
        │
        ├── mode = "react"
        │   OR confidence < 0.5
        │   OR query has: compare/research/analyze/vs/step by step
        │         → ReActOrchestrator (up to 5 iterations)
        │
        └── else
                  → Single capability call (fastest)
```

---

## ⚙️ Key Config Locations

| What | Where | Code |
|------|-------|------|
| LLM model | `agents/base_agents.py` | `model="openai/gpt-oss-120b"` |
| Temperature | `agents/base_agents.py` | `"temperature": 0.3` |
| Max tokens | `agents/base_agents.py` | `"max_tokens": 1000` |
| ReAct iterations | `agents/base_agents.py` | `max_iterations=5` |
| ReAct threshold | `agents/base_agents.py` | `react_confidence_threshold=0.5` |
| Parallel timeout | `agents/base_agents.py` | `parallel_timeout=30.0` |
| Rate limit | `backend/api.py` | `REQUEST_COOLDOWN = 2` |
| Server port | `main.py` | `port=8000` |
| Tasks file | `agents/task_agents.py` | `TASKS_FILE = "tasks.json"` |
| Memory logs dir | `agents/memory.py` | `memory/conversation_logs/` |

---

## 🔍 Reading the Console Logs

```
INFO:jarvis:🚀 NEW REQUEST [rid=abc123]    ← new request, unique ID
INFO:jarvis:🧠 [PLANNER] 💭 Thinking: ...  ← planner analyzing
INFO:jarvis:🧠 [PLANNER] ✅ Decision: ...  ← capability + mode picked
INFO:jarvis:🤖 [ORCHESTRATOR] ──▶ ...     ← mode selected
INFO:jarvis:🔄 [REACT] 💭 Thinking: Iter 1 ← ReAct iteration
INFO:jarvis:🔄 [REACT] ⚡ Action: ...      ← capability being called
INFO:jarvis:🔄 [REACT] 👁️ Result: ...      ← result received
INFO:agents.search_agent:[SearchAgent] DDGS web results: 3 hits, 689 chars
INFO:agents.search_agent:[SearchAgent] Wikipedia extract chars=1205 title='...'
INFO:jarvis:   ⏱️ Completed in 4688.8ms   ← total time
INFO:backend.api:[rid=abc123] /ask success; len=1239
```

---

## 🐛 Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `GROQ_API_KEY not found` | Create `.env` in project root |
| `Port 8000 in use` | `netstat -ano \| findstr :8000` → kill PID |
| `Module not found` | `pip install -r requirements.txt` |
| `ConnectionError` on LLM | Start server from terminal (proxy env needed) |
| Empty response from search | DDG relevance guard triggered; Wikipedia used instead |
| `[LOOP PREVENTED]` in response | Fixed — only happens with truly duplicate inputs now |
| `DeprecationWarning datetime` | Fixed — all use `datetime.now(timezone.utc)` |
| `RuntimeWarning ddgs` | Fixed — suppressed at import with `warnings.filterwarnings` |

---

## ➕ Add a New Capability (3 Steps)

```python
# Step 1: Register in your agent's __init__
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

register(
    capability="my.action",
    agent_name=self.name,
    handler=self.my_handler,
    description="What it does",
    parameters=[ToolParameter(name="input", type=ParameterType.STRING,
                              description="...", required=True)],
)

# Step 2: Implement the handler
async def my_handler(self, input: str) -> str:
    return f"Result for {input}"

# Step 3: Add instance to AGENTS dict in base_agents.py
AGENTS = { ..., "my": my_agent }
```
Planner auto-discovers it. No prompt editing needed. ✅

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| Swagger UI (local) | http://localhost:8000/docs |
| Groq Console | https://console.groq.com |
| OpenWeatherMap | https://openweathermap.org/api |
| NewsAPI | https://newsapi.org |
| Alpha Vantage | https://www.alphavantage.co |
| AutoGen Docs | https://microsoft.github.io/autogen/ |
| ddgs Docs | https://pypi.org/project/ddgs/ |

---

*Full architecture: [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) | File details: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)*
