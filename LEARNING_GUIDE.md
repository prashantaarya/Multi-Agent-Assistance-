# 🎓 JARVIS Learning Guide - From Beginner to Advanced

## 🎯 Learning Path

```
Level 1: Basics          → Level 2: Intermediate → Level 3: Advanced
─────────────────────────────────────────────────────────────────────
• How to run            • Agent communication    • Add new agents
• Make API calls        • Async/await patterns   • Custom integrations
• Read logs             • Error handling         • Performance tuning
• Understand flow       • JSON delegation        • Production deployment
```

---

## 📚 Part 1: Complete Beginner's Guide

### **1.1 - Setup Your First JARVIS Instance**

**Step 1: Install Python** (if not already installed)
```powershell
# Check if Python is installed
python --version

# Should show: Python 3.11.x or newer
```

**Step 2: Clone/Download the Project**
```powershell
cd C:\Users\RYR3COB\Desktop\JAR\Multi-Agent-Assistance-
```

**Step 3: Create Virtual Environment** (Recommended)
```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Your terminal should now show (venv) prefix
```

**Step 4: Install Dependencies**
```powershell
pip install -r requirements.txt
```

**Step 5: Create `.env` File**
```powershell
# Create a file named .env in the root directory
New-Item -Path .\.env -ItemType File

# Add these lines (replace with your actual keys):
```

Contents of `.env`:
```env
# LLM Provider (Required)
GROQ_API_KEY=gsk_your_groq_api_key_here

# External APIs (Optional but recommended)
OPENWEATHER_API_KEY=your_openweather_key
NEWSAPI_KEY=your_newsapi_key
ALPHAVANTAGE_KEY=your_alphavantage_key
```

**Where to get API keys:**
- Groq: https://console.groq.com/ (FREE, required)
- OpenWeather: https://openweathermap.org/api (FREE)
- NewsAPI: https://newsapi.org/ (FREE tier available)
- Alpha Vantage: https://www.alphavantage.co/ (FREE tier available)

**Step 6: Run JARVIS**
```powershell
python main.py
```

You should see:
```
🚀 Starting J.A.R.V.I.S Multi-Agent Assistant...
✅ AutoGen v0.4 initialized
✅ Environment validated
🎯 J.A.R.V.I.S is ready for requests
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Step 7: Test It!**

Open your browser and go to: `http://localhost:8000`

You should see:
```json
{
  "message": "🤖 J.A.R.V.I.S Multi-Agent Assistant",
  "version": "2.0.0",
  "status": "online"
}
```

---

### **1.2 - Making Your First API Call**

**Option A: Using the Browser** (easiest)

1. Go to: `http://localhost:8000/docs`
2. Click on `POST /api/v1/ask`
3. Click "Try it out"
4. Enter:
```json
{
  "message": "Hello JARVIS!",
  "agent": "auto"
}
```
5. Click "Execute"

**Option B: Using PowerShell**

```powershell
$body = @{
    message = "What's the weather in Tokyo?"
    agent = "auto"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ask" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```

**Option C: Using Python**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/ask",
    json={
        "message": "Tell me a joke",
        "agent": "auto"
    }
)

print(response.json())
```

---

### **1.3 - Understanding the Response**

Every successful response looks like this:

```json
{
  "response": "The actual answer from JARVIS",
  "agent_used": "multi-agent",
  "status": "success"
}
```

**Fields explained:**
- `response`: The AI-generated answer
- `agent_used`: Which agent(s) handled your request
- `status`: "success" or "error"

---

## 🔬 Part 2: Understanding Core Concepts

### **2.1 - What is a Multi-Agent System?**

Imagine a company:
- **CEO (Planner Agent)**: Decides who should do what
- **Research Team (Search Agent)**: Finds information
- **IT Team (Tool Agent)**: Runs code and calculations
- **External Relations (API Agent)**: Gets data from outside
- **HR Team (Task Agent)**: Manages schedules and reminders

When you ask JARVIS a question, the Planner (CEO) decides which team should handle it.

### **2.2 - The Delegation Pattern**

```
You: "What's the weather in Paris?"
     ↓
Planner: "This needs external data... I'll send it to the API team"
     ↓
API Agent: "Got it! Calling OpenWeatherMap API..."
     ↓
APIManager: *Makes HTTP request*
     ↓
API Agent: "Weather retrieved! Formatting response..."
     ↓
You: "🌤️ Weather in Paris: Clear sky, 15°C"
```

### **2.3 - Why JSON for Delegation?**

JSON is structured and unambiguous:

**Bad (ambiguous text):**
```
"Send this to the API agent to get weather for Paris"
```

**Good (structured JSON):**
```json
{
  "agent": "api",
  "task": "weather: Paris, France"
}
```

The orchestrator can parse this reliably every time!

### **2.4 - Async/Await Explained**

**Synchronous (blocking) - SLOW:**
```python
# Wait for each task to finish before starting the next
result1 = fetch_weather()      # Takes 2 seconds
result2 = fetch_news()         # Takes 2 seconds
result3 = fetch_stocks()       # Takes 2 seconds
# Total: 6 seconds!
```

**Asynchronous (non-blocking) - FAST:**
```python
# Start all tasks at once
results = await asyncio.gather(
    fetch_weather(),    # All three
    fetch_news(),       # run at the
    fetch_stocks()      # same time!
)
# Total: 2 seconds! (the longest individual task)
```

---

## 🎯 Part 3: Practical Examples

### **Example 1: Weather Query (API Agent)**

**Request:**
```json
{
  "message": "What's the weather in London?",
  "agent": "auto"
}
```

**What happens:**
1. Planner receives: "What's the weather in London?"
2. Planner analyzes → Needs external data
3. Planner outputs: `{"agent":"api", "task":"weather: London"}`
4. Orchestrator routes to API Agent
5. API Agent calls OpenWeatherMap API
6. Formats and returns weather data

**Response:**
```json
{
  "response": "🌤️ Weather in London:\n• Cloudy\n• Temperature: 12°C\n• Humidity: 78%",
  "agent_used": "multi-agent",
  "status": "success"
}
```

---

### **Example 2: Knowledge Search (Search Agent)**

**Request:**
```json
{
  "message": "Who is Albert Einstein?",
  "agent": "auto"
}
```

**What happens:**
1. Planner identifies → Factual knowledge query
2. Planner outputs: `{"agent":"search", "task":"Albert Einstein biography"}`
3. Search Agent calls DuckDuckGo + Wikipedia in parallel
4. Gets raw data from both sources
5. Uses LLM to summarize the information
6. Returns natural language summary

**Response:**
```json
{
  "response": "📘 Summary:\nAlbert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity...",
  "agent_used": "multi-agent",
  "status": "success"
}
```

---

### **Example 3: Code Execution (Tool Agent)**

**Request:**
```json
{
  "message": "Calculate the factorial of 10",
  "agent": "auto"
}
```

**What happens:**
1. Planner identifies → Needs calculation
2. Planner outputs: `{"agent":"tool", "task":"calculate factorial of 10"}`
3. Tool Agent generates Python code
4. Sends code to Docker Manager
5. Docker creates isolated container
6. Executes code safely
7. Returns result

**Response:**
```json
{
  "response": "```python\n3628800\n```",
  "agent_used": "multi-agent",
  "status": "success"
}
```

---

### **Example 4: Direct Answer (Planner)**

**Request:**
```json
{
  "message": "Tell me a joke",
  "agent": "auto"
}
```

**What happens:**
1. Planner receives: "Tell me a joke"
2. Planner analyzes → Simple conversational request
3. Planner answers directly (no delegation!)
4. Returns joke immediately

**Response:**
```json
{
  "response": "Why do programmers prefer dark mode? Because light attracts bugs! 🐛😄",
  "agent_used": "multi-agent",
  "status": "success"
}
```

---

### **Example 5: Task Management (Task Agent)**

**Request:**
```json
{
  "message": "Remind me to call the dentist tomorrow at 3 PM",
  "agent": "auto"
}
```

**What happens:**
1. Planner identifies → Task/reminder request
2. Planner outputs: `{"agent":"task", "task":"remind me to call dentist tomorrow at 3 PM"}`
3. Task Agent processes the reminder
4. Stores in reminders system
5. Confirms to user

**Response:**
```json
{
  "response": "✅ Reminder set: Call the dentist tomorrow at 3:00 PM",
  "agent_used": "multi-agent",
  "status": "success"
}
```

---

## 🐛 Part 4: Troubleshooting Guide

### **Problem 1: "GROQ_API_KEY not found!"**

**Error:**
```
RuntimeError: GROQ_API_KEY not found in environment variables!
```

**Solution:**
1. Check if `.env` file exists in the root directory
2. Verify it contains: `GROQ_API_KEY=gsk_...`
3. Make sure the key is valid (get one from https://console.groq.com/)
4. Restart the application

---

### **Problem 2: "Rate limit exceeded"**

**Error:**
```json
{
  "error": "Rate limit exceeded. Wait 1.5s",
  "status": "error",
  "code": 429
}
```

**Solution:**
- This is intentional rate limiting
- Wait 2 seconds between requests
- To change cooldown, edit `REQUEST_COOLDOWN` in `backend/api.py`

---

### **Problem 3: "Module not found"**

**Error:**
```
ModuleNotFoundError: No module named 'autogen_agentchat'
```

**Solution:**
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# If still fails, try:
pip install autogen-agentchat autogen-core autogen-ext[openai]
```

---

### **Problem 4: "Port 8000 already in use"**

**Error:**
```
ERROR: [Errno 10048] Only one usage of each socket address...
```

**Solution:**
```powershell
# Option 1: Kill the process using port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force

# Option 2: Use a different port
# Edit main.py, change port=8000 to port=8001
```

---

### **Problem 5: Weather/News/Stock APIs not working**

**Error:**
```json
{
  "response": "❌ Weather error: 401 Unauthorized"
}
```

**Solution:**
1. Check if API keys are set in `.env`
2. Verify keys are valid (test on the API provider's website)
3. Check if you've exceeded free tier limits
4. Some APIs require credit card on file (even for free tier)

---

### **Problem 6: Docker not found (Tool Agent)**

**Error:**
```
docker.errors.DockerException: Error while fetching server API version
```

**Solution:**
1. Install Docker Desktop: https://www.docker.com/products/docker-desktop
2. Start Docker Desktop
3. Verify with: `docker ps`
4. Restart JARVIS

---

## 📊 Part 5: Reading Logs

### **Log Levels:**
- `INFO`: Normal operations (green flags ✅)
- `WARNING`: Something unusual but not critical (yellow ⚠️)
- `ERROR`: Something went wrong (red ❌)

### **Common Log Messages:**

**Startup:**
```
🚀 Starting J.A.R.V.I.S Multi-Agent Assistant...
✅ AutoGen v0.4 initialized
✅ Environment validated
🎯 J.A.R.V.I.S is ready for requests
```
**Meaning:** Everything started successfully

---

**Processing Request:**
```
INFO: Processing request: What's the weather in Tokyo?...
```
**Meaning:** A new request is being handled

---

**Rate Limit:**
```
WARNING: HTTP error: 429 - Rate limit exceeded
```
**Meaning:** User is sending requests too fast

---

**API Error:**
```
ERROR: Error in ask_jarvis: Request timeout
```
**Meaning:** External API took too long to respond

---

## 🚀 Part 6: Advanced Usage

### **6.1 - Direct Agent Communication**

Skip the planner and talk directly to a specific agent:

```json
{
  "message": "What's the weather in Paris?",
  "agent": "api"
}
```

Use `POST /api/v1/ask-direct` for this.

### **6.2 - Health Monitoring**

Check system health:
```
GET http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "agents_available": ["planner", "task", "tool", "api", "search"],
  "version": "2.0-v0.4"
}
```

### **6.3 - List All Agents**

```
GET http://localhost:8000/api/v1/agents
```

Response:
```json
{
  "agents": {
    "planner": {
      "name": "planner",
      "description": "You are the PlannerAgent...",
      "status": "active"
    },
    "task": {...},
    "tool": {...},
    "api": {...},
    "search": {...}
  },
  "total": 5
}
```

---

## 🎓 Part 7: Key Takeaways

### **Core Principles:**

1. **Separation of Concerns**
   - Each agent has ONE job
   - Planner only routes, never executes
   - Specialized agents do the actual work

2. **Async > Sync**
   - Always use `async/await` for I/O operations
   - Enables parallel processing
   - Better performance

3. **Error Handling**
   - Always try/except around external calls
   - Provide user-friendly error messages
   - Log errors for debugging

4. **JSON is King**
   - Structured data > Unstructured text
   - Easy to parse programmatically
   - No ambiguity

5. **Rate Limiting**
   - Protects your API from abuse
   - Prevents cost overruns
   - Ensures fair usage

---

## 🎯 Part 8: Quick Reference

### **Common Commands:**

```powershell
# Start JARVIS
python main.py

# Install dependencies
pip install -r requirements.txt

# Check if server is running
curl http://localhost:8000

# View logs
Get-Content .\jarvis.log -Wait  # Real-time log viewing

# Kill JARVIS
Ctrl + C
```

### **Common API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Welcome message |
| `/api/v1/ask` | POST | Main chat endpoint |
| `/api/v1/ask-direct` | POST | Direct agent chat |
| `/api/v1/health` | GET | System health |
| `/api/v1/agents` | GET | List all agents |
| `/docs` | GET | Interactive API docs |

### **Environment Variables:**

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | ✅ Yes | LLM inference |
| `OPENWEATHER_API_KEY` | ⚠️ Optional | Weather data |
| `NEWSAPI_KEY` | ⚠️ Optional | News articles |
| `ALPHAVANTAGE_KEY` | ⚠️ Optional | Stock quotes |

---

## 🎉 Congratulations!

You now understand:
- ✅ How to set up and run JARVIS
- ✅ How multi-agent systems work
- ✅ How requests flow through the system
- ✅ How to troubleshoot common issues
- ✅ How to make API calls
- ✅ How to read logs and debug

**Next Steps:**
1. Try adding your own agent (see ARCHITECTURE_GUIDE.md)
2. Integrate a new API
3. Customize the Planner's behavior
4. Deploy to production (cloud hosting)

**Happy coding! 🚀**
