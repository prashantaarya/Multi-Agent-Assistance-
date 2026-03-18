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

# Capability routing + typed contracts + tracing
from core.schemas import Task, PlannerDecision, Result, Artifact, ExecutionMode, CapabilityCall
from core.capabilities import resolve, list_capabilities, generate_planner_prompt, list_tools
from core.tracing import get_tracer

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
        "temperature": 0.3,   # lower temperature for routing determinism
        "max_tokens": 1000,
        "top_p": 0.9,
        "stop": ["TERMINATE"],
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# Specialist agents (each registers its own capabilities on init)
# MUST be initialized BEFORE planner to populate the tool registry
# ──────────────────────────────────────────────────────────────────────────────

task_agent   = TaskAgent(name="task", model_client=model_client)
tool_agent   = ToolAgent(name="tool", model_client=model_client)
api_agent    = APIAgent(name="api", model_client=model_client)
search_agent = SearchAgent(name="search", model_client=model_client)

# ──────────────────────────────────────────────────────────────────────────────
# Planner (brain): delegates by CAPABILITY with strict JSON
# Auto-generates tool documentation from registry (Industry Best Practice)
# ──────────────────────────────────────────────────────────────────────────────

PLANNER_PROMPT_V = "planner@v4.0"

def _build_planner_system_message() -> str:
    """
    Build planner system message with AUTO-GENERATED tool documentation.
    This ensures the planner always has up-to-date tool information.
    Includes execution mode selection (single vs react vs parallel).
    """
    # Get auto-generated tool documentation from registry
    tools_doc = generate_planner_prompt()
    
    return f"""
[{PLANNER_PROMPT_V}]
You are the Planner (JARVIS's brain). Analyze the user's request and output STRICT JSON.

## OUTPUT SCHEMA
{{
  "capability": "<primary_capability_name>",
  "inputs": {{ <required_parameters> }},
  "confidence": <0.0 to 1.0>,
  "fallback": "<alternative_capability | null>",
  "mode": "single | react | parallel",
  "reasoning": "<brief explanation of your decision>",
  "parallel_capabilities": [  // ONLY for mode="parallel"
    {{"capability": "<cap1>", "inputs": {{...}}, "label": "<friendly_name>"}},
    {{"capability": "<cap2>", "inputs": {{...}}, "label": "<friendly_name>"}}
  ]
}}

## MODE SELECTION RULES (CRITICAL)

### "single" (Default)
Use for straightforward requests with ONE clear capability.
Examples: "weather in Delhi", "add task", "run code"

### "parallel" (Multiple Independent Tasks)
Use when user wants MULTIPLE INDEPENDENT things done at once:
- Keywords: "and", "also", "both", "as well as", "along with"
- Pattern: "Get X and Y", "Show me A and B"
- The tasks must be INDEPENDENT (not dependent on each other)

### "react" (Complex Reasoning)
Use for tasks requiring STEP-BY-STEP reasoning where results depend on each other:
- Comparisons requiring analysis: "which is better"
- Research tasks: "investigate", "analyze", "explain in detail"
- Tasks where step 2 depends on step 1's result

## {tools_doc}

## ROUTING RULES (STRICT)
1. FACTUAL QUESTIONS: who/what/when/where/why/how -> "search.web"
2. TIME-SENSITIVE: "today", "latest", "current", "now" -> appropriate API or search
3. TASK MANAGEMENT: add/list/complete/clear tasks -> "todo.*" capabilities
4. CODE EXECUTION: run Python code -> "code.execute"
5. WEATHER/NEWS/STOCKS: Use corresponding API capabilities
6. SMALL TALK: greetings, opinions -> respond directly (NO JSON)

## EXAMPLES - SINGLE MODE

User: "what's the weather in Delhi?"
{{"capability":"weather.read","inputs":{{"city":"Delhi"}},"confidence":0.92,"fallback":"search.web","mode":"single","reasoning":"Simple weather lookup"}}

User: "add task Buy groceries"
{{"capability":"todo.add","inputs":{{"description":"Buy groceries"}},"confidence":0.95,"fallback":"todo.list","mode":"single","reasoning":"Direct task addition"}}

## EXAMPLES - PARALLEL MODE (Independent tasks)

User: "get weather in Delhi and latest tech news"
{{"capability":"weather.read","inputs":{{"city":"Delhi"}},"confidence":0.9,"fallback":null,"mode":"parallel","reasoning":"Two independent lookups - weather AND news","parallel_capabilities":[{{"capability":"weather.read","inputs":{{"city":"Delhi"}},"label":"Delhi Weather"}},{{"capability":"news.fetch","inputs":{{"topic":"technology"}},"label":"Tech News"}}]}}

User: "show me both Mumbai weather and stock market update"
{{"capability":"weather.read","inputs":{{"city":"Mumbai"}},"confidence":0.88,"fallback":null,"mode":"parallel","reasoning":"User wants two independent pieces of info","parallel_capabilities":[{{"capability":"weather.read","inputs":{{"city":"Mumbai"}},"label":"Mumbai Weather"}},{{"capability":"stock.price","inputs":{{"symbol":"NIFTY"}},"label":"Stock Update"}}]}}

User: "add task call mom and also add task buy milk"
{{"capability":"todo.add","inputs":{{"description":"call mom"}},"confidence":0.9,"fallback":null,"mode":"parallel","reasoning":"Two independent task additions","parallel_capabilities":[{{"capability":"todo.add","inputs":{{"description":"call mom"}},"label":"Task 1"}},{{"capability":"todo.add","inputs":{{"description":"buy milk"}},"label":"Task 2"}}]}}

## EXAMPLES - REACT MODE (Step-by-step reasoning)

User: "compare weather in Delhi vs Mumbai and recommend which is better for travel"
{{"capability":"weather.read","inputs":{{"city":"Delhi"}},"confidence":0.7,"fallback":"search.web","mode":"react","reasoning":"Comparison needs both data points then analysis"}}

User: "research quantum computing and explain its applications"
{{"capability":"search.web","inputs":{{"query":"quantum computing explanation"}},"confidence":0.65,"fallback":null,"mode":"react","reasoning":"Research requires multiple searches and synthesis"}}

## SMALL TALK (NO JSON)

User: "Hi, how are you?"
Hello! I'm JARVIS, your AI assistant. How can I help you today?

IMPORTANT: Output ONLY valid JSON for capability routing. NO markdown, NO extra text.
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
            # Safety net: if factual or time-sensitive -> force search.web
            if _looks_factual_question(message) or _looks_time_sensitive(message):
                with tracer.span("override.search_web"):
                    resolved = resolve("search.web")
                    if resolved:
                        agent_name, handler = resolved
                        try:
                            return await handler(query=message)
                        except Exception:
                            # If search fails, fall back to planner’s text
                            return buffer
                    return buffer  # capability not registered
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

        # Confidence/time gate: if planner wants to answer directly with low confidence, force search
        if decision.capability == "planner" and (decision.confidence is None or decision.confidence < 0.7):
            with tracer.span("override.planner_low_conf_to_search", confidence=decision.confidence):
                resolved = resolve("search.web")
                if resolved:
                    _, handler = resolved
                    try:
                        return await handler(query=message)
                    except Exception:
                        pass  # if search fails, continue normal flow (rare)

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
