# agents/base_agents.py

import os
import asyncio
import json
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from .api_agent import APIAgent
from .search_agent import SearchAgent

# Load environment variables
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
    json_output=False
)

# Create model client
model_client = OpenAIChatCompletionClient(
    model="llama3-70b-8192",
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
    model_info=model_info,
    create_config={
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "stop": ["TERMINATE"]
    }
)

# PlannerAgent: outputs JSON to delegate or plain text to answer
planner_agent = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message="""
You are the PlannerAgent. Your sole responsibility is to route each user request to exactly one of four agents—or answer it yourself—using this **strict** protocol:

1. **Agents & when to use them**  
   • **planner**: Handle purely conversational or opinion‑based questions (no external data needed).  
   • **task**: Handle reminders, to‑do lists, scheduling, email/calendar operations.  
   • **tool**: Handle code execution, calculations, and file/text processing.  
   • **api**: Handle structured data lookups—weather, stock quotes, simple news—using free APIs.  
   • **search**: Handle factual or reference lookups (“who is…”, “when did…”, definitions, statistics) via DuckDuckGo Instant Answer.

2. **Output format**  
   - **If you delegate**, output **ONLY** this JSON (with no extra text):  
     `{"agent":"<agent_key>","task":"<task description>"}`  
     where `<agent_key>` is exactly one of `"planner"`, `"task"`, `"tool"`, `"api"`, or `"search"`.  
   - **If you answer directly**, output **only** your natural‑language answer (no JSON).

3. **Examples**  
   - User: “Remind me to call John tomorrow at 9 AM.”  
     → `{"agent":"task","task":"remind me to call John tomorrow at 9 AM"}`  
   - User: “What’s the weather in Tokyo?”  
     → `{"agent":"api","task":"weather: Tokyo, Japan"}`  
   - User: “Who invented the telephone?”  
     → `{"agent":"search","task":"Alexander Graham Bell inventor telephone"}`  
   - User: “Tell me a joke.”  
     → _Your joke here (no JSON)_

Strictly follow these rules for every request. Never wrap JSON in markdown or quote marks. Delegation must always be a single, well‑formed JSON object. 

""".strip()
)

# Specialist agents
task_agent   = AssistantAgent(
    name="task",
    model_client=model_client,
    system_message="You are the TaskAgent. Handle reminders, to‑do lists, and scheduling with concise, actionable responses."
)
tool_agent   = AssistantAgent(
    name="tool",
    model_client=model_client,
    system_message="You are the ToolAgent. Execute Python code, perform calculations, and process files. Provide step‑by‑step outputs."
)
api_agent    = APIAgent(name="api", model_client=model_client)
search_agent = SearchAgent(name="search", model_client=model_client)

# Agent registry
AGENTS = {
    "planner": planner_agent,
    "task":    task_agent,
    "tool":    tool_agent,
    "api":     api_agent,
    "search":  search_agent
}

class JARVISAssistant:
    def __init__(self):
        self.agents  = AGENTS
        self.planner = planner_agent

    async def process_request(self, message: str) -> str:
        # 1) Ask the PlannerAgent
        planner_team = RoundRobinGroupChat([self.planner], max_turns=1)
        buffer = ""
        async for msg in planner_team.run_stream(task=message):
            if getattr(msg, "content", None):
                buffer += msg.content

        # 2) Try extracting a JSON delegation block
        json_str = None
        start = buffer.find("{")
        end   = buffer.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = buffer[start:end+1]

        if json_str:
            try:
                plan = json.loads(json_str)
                agent_key = plan.get("agent")
                task      = plan.get("task", "").strip()

                if agent_key in self.agents and agent_key != "planner":
                    agent = self.agents[agent_key]

                    # APIAgent routing by prefix
                    if agent_key == "api":
                        low = task.lower()
                        if low.startswith("weather:"):
                            city = task.split(":",1)[1].strip()
                            return await agent.get_weather(city)
                        if low.startswith("news:"):
                            topic = task.split(":",1)[1].strip()
                            return await agent.get_news(topic)
                        if low.startswith("stock:"):
                            sym = task.split(":",1)[1].strip()
                            return await agent.get_stock(sym)

                    # SearchAgent routing
                    if agent_key == "search":
                        return await agent.search(task)

                    # Generic fallback for task/tool agents
                    return await self._run_agent_task(agent, task)

            except json.JSONDecodeError:
                # Not valid JSON, treat as plain answer
                pass

        # 3) No valid delegation → return planner's response
        return buffer

    async def _run_agent_task(self, agent: AssistantAgent, task: str) -> str:
        """Run one turn with the given agent and return its response."""
        team = RoundRobinGroupChat([agent], max_turns=1)
        out = ""
        async for msg in team.run_stream(task=task):
            if getattr(msg, "content", None):
                out += msg.content
        return out or "No response"

    async def chat_direct(self, message: str, agent_name: str = "planner") -> str:
        """Bypass planner and talk directly to the specified agent."""
        if agent_name not in self.agents:
            return f"Agent '{agent_name}' not found. Available: {list(self.agents.keys())}"
        return await self._run_agent_task(self.agents[agent_name], message)


# Global instance
jarvis = JARVISAssistant()

# Convenience wrappers
async def get_assistant_response(message: str) -> str:
    return await jarvis.process_request(message)

async def run_conversation(message: str) -> str:
    return await jarvis.process_request(message)

def run_sync_conversation(message: str) -> str:
    return asyncio.run(get_assistant_response(message))
