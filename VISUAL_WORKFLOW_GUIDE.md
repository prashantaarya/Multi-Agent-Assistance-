# 🎨 JARVIS Visual Workflow & Data Flow Guide

## 📌 Quick Reference Diagrams

### **1. High-Level System Overview**

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                │
│                                                                        │
│  Web Browser / Mobile App / cURL / Postman / Any HTTP Client         │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 │ HTTP POST Request
                                 │ {"message": "What's the weather?"}
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                             │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  main.py (FastAPI Application)                               │   │
│  │  • CORS Middleware                                            │   │
│  │  • Logging Configuration                                      │   │
│  │  • Error Handlers                                             │   │
│  │  • Lifespan Management                                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          API LAYER                                    │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  backend/api.py (Router)                                     │   │
│  │                                                               │   │
│  │  Endpoints:                                                   │   │
│  │  • POST /api/v1/ask           → Main entry                   │   │
│  │  • POST /api/v1/ask-direct    → Direct agent chat           │   │
│  │  • GET  /api/v1/health        → System health                │   │
│  │  • GET  /api/v1/agents        → List agents                  │   │
│  │                                                               │   │
│  │  Features:                                                    │   │
│  │  • Rate Limiting (2s cooldown)                               │   │
│  │  • Request Validation                                         │   │
│  │  • Response Formatting                                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                              │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  agents/base_agents.py (JARVISAssistant)                    │   │
│  │                                                               │   │
│  │  • Agent Registry                                             │   │
│  │  • Request Routing Logic                                      │   │
│  │  • JSON Parsing                                               │   │
│  │  • Response Aggregation                                       │   │
│  │  • Error Recovery                                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                                    │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Planner Agent (Decision Maker)                  │    │
│  │  • Analyzes user intent                                      │    │
│  │  • Routes to appropriate agent                               │    │
│  │  • Outputs JSON: {"agent": "X", "task": "Y"}               │    │
│  └────────────────┬────────────────────────────────────────────┘    │
│                   │                                                   │
│       ┌───────────┼───────────┬──────────────┬─────────────┐        │
│       │           │           │              │             │         │
│       ▼           ▼           ▼              ▼             ▼         │
│  ┌────────┐ ┌────────┐ ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Task  │ │  Tool  │ │   API    │  │  Search  │  │  Future  │   │
│  │ Agent  │ │ Agent  │ │  Agent   │  │  Agent   │  │  Agents  │   │
│  └────────┘ └────────┘ └──────────┘  └──────────┘  └──────────┘   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      INTEGRATION LAYER                                │
│                                                                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │   Groq     │  │  Docker    │  │   APIs     │  │  Search    │    │
│  │    LLM     │  │ Container  │  │ (Weather)  │  │ (DuckDo)   │    │
│  │  (Llama3)  │  │ (Python)   │  │ (News)     │  │ (Wiki)     │    │
│  │            │  │            │  │ (Stocks)   │  │            │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Request Flow (Detailed)

### **Scenario: User asks "What's the weather in Tokyo?"**

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: User Makes HTTP Request                                 │
└─────────────────────────────────────────────────────────────────┘

POST http://localhost:8000/api/v1/ask
Headers: {
    "Content-Type": "application/json"
}
Body: {
    "message": "What's the weather in Tokyo?",
    "agent": "auto"
}

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: FastAPI App Receives Request (main.py)                 │
└─────────────────────────────────────────────────────────────────┘

app.include_router(router, prefix="/api/v1")
    ↓
Routes to → backend/api.py :: ask_jarvis()

Log: "🎯 J.A.R.V.I.S is ready for requests"

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: API Router Validates & Rate Limits (backend/api.py)    │
└─────────────────────────────────────────────────────────────────┘

@router.post("/ask", response_model=AskResponse)
async def ask_jarvis(payload: AskRequest):
    
    ✓ Validate: message.strip() is not empty
    ✓ Rate Limit: Check last_request_time < 2 seconds
    
    Log: "Processing request: What's the weather in Tokyo?..."

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Call JARVIS Orchestrator (agents/base_agents.py)       │
└─────────────────────────────────────────────────────────────────┘

response = await jarvis.process_request(user_message)
    ↓
class JARVISAssistant:
    async def process_request(self, message: str):
        # Send to Planner Agent
        planner_team = RoundRobinGroupChat([self.planner], max_turns=1)

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Planner Agent Analyzes Request                         │
└─────────────────────────────────────────────────────────────────┘

Planner Agent:
    Name: "planner"
    Model: Llama 3 70B via Groq
    System Message: "Route to api/task/tool/search agents"
    
Input: "What's the weather in Tokyo?"

LLM Processing:
    ┌─────────────────────────────────────┐
    │  Groq API Call                      │
    │  • Model: llama3-70b-8192           │
    │  • Temperature: 0.7                 │
    │  • Max Tokens: 1000                 │
    └─────────────────────────────────────┘

Output (JSON):
{
    "agent": "api",
    "task": "weather: Tokyo, Japan"
}

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Parse JSON & Route to Agent                            │
└─────────────────────────────────────────────────────────────────┘

# Extract JSON from buffer
json_str = buffer.find("{") to buffer.rfind("}")
plan = json.loads(json_str)

agent_key = "api"
task = "weather: Tokyo, Japan"

# Get API Agent from registry
if agent_key == "api":
    agent = self.agents["api"]  # APIAgent instance
    
    # Check task prefix
    if task.lower().startswith("weather:"):
        city = task.split(":", 1)[1].strip()  # "Tokyo, Japan"
        return await agent.get_weather(city)

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: API Agent Processes Request (agents/api_agent.py)      │
└─────────────────────────────────────────────────────────────────┘

class APIAgent(AssistantAgent):
    async def get_weather(self, city: str):
        # Delegate to APIManager
        return await self.api.get_weather(city)

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: API Manager Makes External Call                        │
│         (agents/api_manager.py)                                 │
└─────────────────────────────────────────────────────────────────┘

class APIManager:
    async def get_weather(self, city: str):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                
External API Response:
{
    "weather": [{"description": "clear sky"}],
    "main": {
        "temp": 18.5,
        "humidity": 65
    },
    "name": "Tokyo"
}

Format Response:
return f"""
🌤️ Weather in Tokyo:
• Clear sky
• Temperature: 18.5°C
• Humidity: 65%
"""

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 9: Response Bubbles Back Up                               │
└─────────────────────────────────────────────────────────────────┘

APIManager.get_weather()
    ↓ returns string
API Agent.get_weather()
    ↓ returns string
JARVIS.process_request()
    ↓ returns string
API Router.ask_jarvis()
    ↓ wraps in AskResponse
FastAPI App
    ↓ serializes to JSON

                             ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 10: User Receives JSON Response                           │
└─────────────────────────────────────────────────────────────────┘

HTTP 200 OK
{
    "response": "🌤️ Weather in Tokyo:\n• Clear sky\n• Temperature: 18.5°C\n• Humidity: 65%",
    "agent_used": "multi-agent",
    "status": "success"
}

Total Time: ~1-2 seconds
```

---

## 🧠 Agent Decision Tree

```
                    ┌─────────────────┐
                    │  User Message   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Planner Agent   │
                    │   (Analyzer)    │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ LLM Analysis    │
                    │ (Llama 3 70B)   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┬─────────────┐
            │                │                │             │
            ▼                ▼                ▼             ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ Conversational│ │   Data      │  │   Action    │  │  Knowledge  │
    │   Question   │  │   Request   │  │   Request   │  │   Lookup    │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                 │                │
           ▼                ▼                 ▼                ▼
    ┌──────────┐    ┌──────────┐      ┌──────────┐    ┌──────────┐
    │ Answer   │    │   API    │      │   Tool   │    │  Search  │
    │ Directly │    │  Agent   │      │  Agent   │    │  Agent   │
    └──────────┘    └────┬─────┘      └────┬─────┘    └────┬─────┘
                         │                  │                │
                    ┌────┴────┐        ┌────┴────┐      ┌────┴────┐
                    │  Task   │        │ Docker  │      │DuckDuckGo│
                    │ Agent   │        │Execute  │      │Wikipedia │
                    └─────────┘        └─────────┘      └─────────┘

Examples:
─────────────────────────────────────────────────────────────────
"Tell me a joke"              → Planner answers directly
"What's the weather?"         → API Agent → OpenWeatherMap
"Remind me tomorrow"          → Task Agent → Reminders system
"Calculate 2+2"               → Tool Agent → Docker Python
"Who is Einstein?"            → Search Agent → Wikipedia/DuckDuckGo
```

---

## 📦 Component Interaction Matrix

| Component | Talks To | Data Sent | Data Received |
|-----------|----------|-----------|---------------|
| **main.py** | backend/api.py | FastAPI app instance | - |
| **backend/api.py** | JARVISAssistant | User message (str) | Response (str) |
| **JARVISAssistant** | Planner Agent | User message | JSON delegation |
| **Planner Agent** | Groq LLM | Prompt + context | JSON or text |
| **JARVISAssistant** | Specialized Agent | Task description | Task result |
| **API Agent** | APIManager | City/topic/symbol | Formatted data |
| **APIManager** | External APIs | HTTP requests | JSON responses |
| **Tool Agent** | DockerManager | Python code | Execution output |
| **DockerManager** | Docker Engine | Container specs | stdout/stderr |
| **Search Agent** | DuckDuckGo/Wiki | Search query | Raw data |
| **Search Agent** | Groq LLM | Raw data + query | Summary |

---

## 🔐 Data Flow: Environment Variables

```
┌──────────────────────────────────────────────────────────────┐
│  .env file (NOT in version control)                          │
│                                                               │
│  GROQ_API_KEY=gsk_xxx...                                     │
│  OPENWEATHER_API_KEY=abc123...                               │
│  NEWSAPI_KEY=def456...                                       │
│  ALPHAVANTAGE_KEY=ghi789...                                  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ python-dotenv loads
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  OS Environment Variables                                     │
│  os.getenv("GROQ_API_KEY")                                   │
└────────────┬─────────────────┬────────────────┬──────────────┘
             │                 │                │
             ▼                 ▼                ▼
    ┌────────────────┐  ┌────────────┐  ┌────────────┐
    │  base_agents   │  │ api_manager│  │   main.py  │
    │  (model_client)│  │ (API keys) │  │ (validation)│
    └────────────────┘  └────────────┘  └────────────┘
```

---

## ⚡ Performance Flow

```
Request Timeline:
─────────────────────────────────────────────────────────────

0ms     │ User sends HTTP request
        │
10ms    │ FastAPI receives & validates
        │ Rate limit check
        │
50ms    │ JARVIS orchestrator initializes
        │ Creates RoundRobinGroupChat
        │
150ms   │ Planner Agent → Groq LLM call
        │ (Llama 3 70B inference)
        │
200ms   │ JSON parsing & agent routing
        │
250ms   │ API Agent → External API call
        │ (OpenWeatherMap/NewsAPI/etc)
        │
800ms   │ External API responds
        │
850ms   │ Format response & return
        │
900ms   │ FastAPI serializes JSON
        │
950ms   │ User receives response
        │
─────────────────────────────────────────────────────────────
Total: ~1 second (typical)

Breakdown:
• Network latency: 100ms
• LLM inference: 150ms (Groq is FAST!)
• External API: 550ms (variable)
• Processing: 150ms
```

---

## 🎯 Agent Responsibility Matrix

| Agent | Primary Function | Input | Output | External Calls |
|-------|------------------|-------|--------|----------------|
| **Planner** | Request routing | User message | JSON delegation | Groq LLM |
| **Task** | Task management | Task description | Confirmation | None |
| **Tool** | Code execution | Python code | Execution result | Docker |
| **API** | Data fetching | Prefix:query | Formatted data | Weather/News/Stock APIs |
| **Search** | Knowledge retrieval | Search query | Summarized facts | DuckDuckGo, Wikipedia |

---

## 🔄 Async Flow Visualization

```python
# Sequential (SLOW - don't do this):
result1 = await api_call_1()  # Wait 500ms
result2 = await api_call_2()  # Wait 500ms
result3 = await api_call_3()  # Wait 500ms
# Total: 1500ms

# Parallel (FAST - what JARVIS does):
results = await asyncio.gather(
    api_call_1(),  # All run
    api_call_2(),  # at the
    api_call_3()   # same time!
)
# Total: 500ms (max of all three)
```

**Where JARVIS Uses Async:**
- ✅ Multiple API calls in APIManager
- ✅ DuckDuckGo + Wikipedia search in parallel
- ✅ Streaming responses from LLM
- ✅ Multiple concurrent user requests

---

## 📊 State Diagram

```
┌─────────────┐
│   IDLE      │ ◄──────────────────────┐
│  (Waiting)  │                        │
└──────┬──────┘                        │
       │                               │
       │ Request arrives               │
       ▼                               │
┌─────────────┐                        │
│ VALIDATING  │                        │
│  (Checks)   │                        │
└──────┬──────┘                        │
       │                               │
       │ Valid                         │
       ▼                               │
┌─────────────┐                        │
│ ROUTING     │                        │
│ (Planner)   │                        │
└──────┬──────┘                        │
       │                               │
       │ Agent selected                │
       ▼                               │
┌─────────────┐                        │
│ PROCESSING  │                        │
│ (Agent work)│                        │
└──────┬──────┘                        │
       │                               │
       │ Success/Error                 │
       ▼                               │
┌─────────────┐                        │
│ RESPONDING  │                        │
│ (Format)    │ ───────────────────────┘
└─────────────┘
```

---

## 🎭 Agent Lifecycle

```
┌───────────────────────────────────────────────────────────┐
│  Application Startup (main.py)                            │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Load Environment Variables                                │
│  • GROQ_API_KEY                                           │
│  • API keys for weather/news/stocks                        │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Create Model Client (base_agents.py)                     │
│  • Connect to Groq API                                     │
│  • Configure Llama 3 70B                                   │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Instantiate All Agents                                    │
│  • planner_agent = AssistantAgent(...)                    │
│  • task_agent = AssistantAgent(...)                       │
│  • tool_agent = AssistantAgent(...)                       │
│  • api_agent = APIAgent(...)                              │
│  • search_agent = SearchAgent(...)                        │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Register Agents in Dictionary                             │
│  AGENTS = {"planner": planner_agent, ...}                 │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Create JARVISAssistant Instance                          │
│  jarvis = JARVISAssistant()                               │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  FastAPI App Ready                                         │
│  Server listening on http://localhost:8000                 │
└───────────────────────────────────────────────────────────┘
```

---

## 🚦 Error Handling Flow

```
                    Request Arrives
                          │
                          ▼
                ┌─────────────────┐
                │  Validation     │
                └────────┬────────┘
                         │
                    ┌────┴────┐
                    │ Valid?  │
                    └────┬────┘
                    No   │   Yes
                    ┌────┴────┐
                    ▼         ▼
            ┌──────────┐  ┌──────────┐
            │ Return   │  │ Process  │
            │ 400 Error│  │ Request  │
            └──────────┘  └────┬─────┘
                               │
                          ┌────┴────┐
                          │Rate Limit│
                          └────┬────┘
                          ┌────┴────┐
                          │Exceeded?│
                          └────┬────┘
                      Yes ┌────┴────┐ No
                          ▼         ▼
                  ┌──────────┐  ┌──────────┐
                  │ Return   │  │ Continue │
                  │ 429 Error│  │ to Agent │
                  └──────────┘  └────┬─────┘
                                     │
                                ┌────┴────┐
                                │Agent Work│
                                └────┬────┘
                                ┌────┴────┐
                                │Success? │
                                └────┬────┘
                            Yes ┌────┴────┐ No
                                ▼         ▼
                        ┌──────────┐  ┌──────────┐
                        │ Return   │  │ Catch    │
                        │ Response │  │ Exception│
                        └──────────┘  └────┬─────┘
                                           │
                                      ┌────┴────┐
                                      │ Log     │
                                      │ Error   │
                                      └────┬────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │ Return       │
                                   │ 500 Error    │
                                   │ with details │
                                   └──────────────┘
```

---

## 🔍 JSON Delegation Examples

### Example 1: Weather Query
```
User Input: "What's the weather in Paris?"

Planner Output:
{
    "agent": "api",
    "task": "weather: Paris, France"
}

Routing: API Agent → APIManager.get_weather()
```

### Example 2: Search Query
```
User Input: "Who invented the telephone?"

Planner Output:
{
    "agent": "search",
    "task": "Alexander Graham Bell telephone inventor"
}

Routing: Search Agent → DuckDuckGo + Wikipedia
```

### Example 3: Task Management
```
User Input: "Remind me to call John tomorrow at 9 AM"

Planner Output:
{
    "agent": "task",
    "task": "remind me to call John tomorrow at 9 AM"
}

Routing: Task Agent → Reminders system
```

### Example 4: Code Execution
```
User Input: "Calculate the factorial of 5"

Planner Output:
{
    "agent": "tool",
    "task": "calculate factorial of 5 using Python"
}

Routing: Tool Agent → Docker Python execution
```

### Example 5: Direct Answer
```
User Input: "Tell me a joke"

Planner Output (plain text, no JSON):
"Why did the programmer quit his job? Because he didn't get arrays!"

Routing: No delegation, planner answers directly
```

---

## 💡 Tips for Understanding

1. **Follow the arrows**: Each arrow represents data flow
2. **Async = Parallel**: Multiple operations can happen simultaneously
3. **JSON is the contract**: Planner uses JSON to delegate tasks
4. **Agents are specialists**: Each agent has one job and does it well
5. **Orchestrator is the brain**: JARVISAssistant manages everything

---

**This guide complements ARCHITECTURE_GUIDE.md with visual workflows!**
