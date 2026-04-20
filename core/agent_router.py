# core/agent_router.py
"""
True Agent Architecture Router

Instead of:  Planner → Python function (old)
Now:         Planner → Domain Agent (LLM) → Tools (new)

Each domain agent is an LLM-powered agent with its own:
- System prompt (domain expertise)
- Tool set (only tools relevant to its domain)
- ReAct reasoning loop (multi-step tool calls)

The router maps domain names to agent instances and handles execution.

Architecture:
    ┌──────────┐
    │  USER    │
    └────┬─────┘
         │
    ┌────▼─────┐
    │ PLANNER  │  "This is a gmail task about replying to urgent emails"
    │  (LLM)   │  → agent: "gmail", task: "check inbox and reply to urgent"
    └────┬─────┘
         │
    ┌────▼──────────────┐
    │   AGENT ROUTER    │  Routes to the right domain agent
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │   GMAIL AGENT     │  LLM thinks: "I need to:
    │   (LLM + Tools)   │   1. Check inbox → gmail_check_inbox()
    │                    │   2. Find urgent → analyze results
    │                    │   3. Read it → gmail_read_email()
    │                    │   4. Draft reply → gmail_draft_reply()"
    └───────────────────┘
"""

import json
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat

from core.tracing import get_tracer

logger = logging.getLogger("jarvis.router")


# ──────────────────────────────────────────────────────────────────────────────
# Domain Agent Definition
# ──────────────────────────────────────────────────────────────────────────────

class DomainAgent:
    """
    An LLM-powered domain agent that can reason and call multiple tools.
    
    Each domain agent:
    - Has a specialized system prompt
    - Has access to a set of tools (functions)
    - Uses a ReAct loop to decide which tools to call and in what order
    - Returns a final synthesized answer
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: Dict[str, Dict[str, Any]],
        model_client,
        max_steps: int = 6,
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools  # {tool_name: {"function": callable, "schema": dict}}
        self.model_client = model_client
        self.max_steps = max_steps
    
    def _build_tools_doc(self) -> str:
        """Build tools documentation for the agent's prompt"""
        lines = []
        for name, tool_info in self.tools.items():
            schema = tool_info["schema"]
            params = schema.get("parameters", {}).get("properties", {})
            required = schema.get("parameters", {}).get("required", [])
            
            param_parts = []
            for pname, pinfo in params.items():
                req = " (REQUIRED)" if pname in required else " (optional)"
                param_parts.append(f"    - {pname}: {pinfo.get('description', '')}{req}")
            
            param_str = "\n".join(param_parts) if param_parts else "    (no parameters)"
            lines.append(f"  {name}: {schema.get('description', '')}\n{param_str}")
        
        return "\n\n".join(lines)
    
    def _build_agent_prompt(self, task: str) -> str:
        """Build the full prompt for the agent including tools and task"""
        tools_doc = self._build_tools_doc()
        
        return f"""{self.system_prompt}

## YOUR TOOLS
{tools_doc}

## HOW TO RESPOND
You must respond in strict JSON format for EACH step:

To CALL A TOOL:
{{
  "thought": "Why I need to call this tool",
  "tool": "tool_name",
  "tool_inputs": {{"param1": "value1", "param2": "value2"}},
  "is_final": false
}}

To GIVE FINAL ANSWER (after tool calls or if you can answer directly):
{{
  "thought": "Summarizing what I found",
  "tool": null,
  "tool_inputs": {{}},
  "is_final": true,
  "answer": "Your complete, helpful answer to the user"
}}

## RULES
1. Maximum {self.max_steps} steps - be efficient
2. ⚠️ CRITICAL: NEVER make up or hallucinate data - ALWAYS call your tools to retrieve real information
3. For data retrieval tasks (list, search, check, get), you MUST call the appropriate tool first
4. Call tools one at a time, wait for results before next step
5. ALWAYS provide a final answer (is_final: true) at the end
6. Use the tool results to build a helpful, natural response
7. For simple tasks (single tool call), call the tool then give final answer
8. For complex tasks, chain multiple tool calls then synthesize
7. Output ONLY valid JSON per step. No markdown, no extra text.
"""

    async def execute(self, task: str, request_id: str = None):
        """
        Execute a task using the agent's ReAct reasoning loop.
        
        The agent thinks about what tool to call, calls it, observes the result,
        and repeats until it has enough info to give a final answer.
        
        Returns: str OR dict with {"response": str, "data": dict}
        """
        tracer = get_tracer(request_id)
        
        context = f"USER TASK: {task}\n\nBegin by analyzing what you need to do:"
        action_history = []
        structured_data = None  # Track structured data from tool calls
        
        with tracer.span(f"agent.{self.name}.execute", task=task[:80]):
            # DEBUG: Verify model client
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[{self.name}] Model client: {type(self.model_client).__name__ if self.model_client else 'NONE'}")
            logger.info(f"[{self.name}] Max steps: {self.max_steps}")
            logger.info(f"[{self.name}] Available tools: {list(self.tools.keys())}")
            
            for step in range(1, self.max_steps + 1):
                tracer.thought(self.name, f"Step {step}/{self.max_steps}")
                
                # DEBUG: Log the context being sent
                logger.info(f"[{self.name}] Step {step} context length: {len(context)} chars")
                logger.info(f"[{self.name}] Step {step} context preview:\n{context[:300]}")
                
                # Build agent with current context
                agent_prompt = self._build_agent_prompt(task)
                
                # DEBUG: Log the prompt being sent
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[{self.name}] Building agent with prompt length: {len(agent_prompt)} chars")
                
                # Create agent using EXACT same pattern as working planner
                agent = AssistantAgent(
                    name=f"{self.name}_agent",
                    model_client=self.model_client,
                    system_message=agent_prompt,
                )
                
                # Call agent using EXACT same pattern as planner
                logger.info(f"[{self.name}] Calling agent with context: {context[:100]}...")
                team = RoundRobinGroupChat([agent], max_turns=1)
                buffer = ""
                
                async for msg in team.run_stream(task=context):
                    msg_content = getattr(msg, "content", None)
                    if msg_content:
                        buffer += msg_content
                        logger.info(f"[{self.name}] Received content chunk, total length now: {len(buffer)}")
                
                logger.info(f"[{self.name}] Final buffer length: {len(buffer)} chars")
                if len(buffer) > 0:
                    logger.info(f"[{self.name}] Buffer preview: {buffer[:200]}")
                else:
                    logger.error(f"[{self.name}] ❌ EMPTY BUFFER - No response from model!")
                    logger.error(f"[{self.name}] Agent name: {agent.name}")
                    logger.error(f"[{self.name}] Model client type: {type(self.model_client)}")
                    logger.error(f"[{self.name}] System message length: {len(agent_prompt)}")
                    logger.error(f"[{self.name}] Context sent: {context}")
                logger.info(f"[{self.name}] LLM raw response (first 800 chars):\n{buffer[:800]}")
                
                # Parse JSON response
                step_data = self._parse_response(buffer)
                if not step_data:
                    tracer.thought(self.name, f"Could not parse response, synthesizing from history")
                    logger.warning(f"[{self.name}] Failed to parse LLM response. Full output:\n{buffer}")
                    return self._synthesize_from_history(task, action_history)
                
                thought = step_data.get("thought", "")
                tracer.decision(self.name, f"[STEP {step}] {thought[:100]}")
                
                # Check if final answer
                if step_data.get("is_final"):
                    answer = step_data.get("answer", "")
                    if answer:
                        tracer.observation(self.name, f"Final answer ready ({len(answer)} chars)")
                        # Return structured format if we have data, otherwise just text
                        if structured_data:
                            return {"response": answer, "data": structured_data}
                        return answer
                    if structured_data:
                        fallback = self._synthesize_from_history(task, action_history)
                        return {"response": fallback, "data": structured_data}
                    return self._synthesize_from_history(task, action_history)
                
                # Execute tool call
                tool_name = step_data.get("tool")
                tool_inputs = step_data.get("tool_inputs", {})
                
                if tool_name and tool_name in self.tools:
                    with tracer.span(f"agent.{self.name}.tool", tool=tool_name):
                        tracer.action(self.name, f"Calling {tool_name}")
                        try:
                            result = self.tools[tool_name]["function"](**tool_inputs)
                            # Handle both sync and async results
                            if hasattr(result, '__await__'):
                                result = await result
                            
                            # Handle structured responses from tools
                            if isinstance(result, dict) and "response" in result:
                                # Tool returned {"response": str, "data": dict}
                                result_str = result["response"]
                                if "data" in result and result["data"] is not None:
                                    structured_data = result["data"]  # Store for final return
                            else:
                                # Legacy: tool returned just a string
                                result_str = json.dumps(result, indent=2, default=str) if isinstance(result, (dict, list)) else str(result)
                            
                            tracer.observation(self.name, f"Got result ({len(result_str)} chars)")
                        except Exception as e:
                            result_str = f"ERROR: {e}"
                            tracer.thought(self.name, f"Tool error: {e}")
                    
                    action_history.append({
                        "step": step,
                        "tool": tool_name,
                        "inputs": tool_inputs,
                        "result": result_str[:2000],  # Truncate long results
                    })
                    
                    # Build context for next iteration
                    context = f"USER TASK: {task}\n\n"
                    context += "PREVIOUS STEPS:\n"
                    for ah in action_history:
                        context += f"\nStep {ah['step']}: Called {ah['tool']}\n"
                        context += f"Result: {ah['result'][:500]}\n"
                    context += f"\nContinue (step {step + 1}/{self.max_steps}). Decide next tool or give final answer:"
                
                elif tool_name:
                    # Tool not found
                    context += f"\n\nError: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}\nTry again:"
            
            # Max steps reached - synthesize from what we have
            tracer.thought(self.name, "Max steps reached, synthesizing answer")
            fallback = self._synthesize_from_history(task, action_history)
            if structured_data:
                return {"response": fallback, "data": structured_data}
            return fallback
    
    def _parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from agent response"""
        if not raw:
            return None
        
        text = raw.strip()
        # Strip markdown fences
        for fence in ["```json", "```JSON", "```"]:
            if fence in text:
                text = text.split(fence, 1)[-1]
                text = text.rsplit("```", 1)[0]
                text = text.strip()
                break
        
        # Find JSON block
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return None
        
        candidate = text[s:e + 1]
        
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        
        # Fix common issues
        import re
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
    
    def _synthesize_from_history(self, task: str, history: List[Dict]) -> str:
        """Create an answer from collected tool results when LLM fails to synthesize"""
        if not history:
            return f"I attempted to handle your request but couldn't gather enough information. Could you rephrase it?"
        
        parts = [f"Here's what I found for: {task}\n"]
        for item in history:
            if "ERROR" not in item.get("result", ""):
                try:
                    result = json.loads(item["result"])
                    if isinstance(result, dict) and result.get("success"):
                        # Format nicely based on the tool
                        if "emails" in result:
                            parts.append(f"📧 **Emails** ({result.get('total', 0)} found):")
                            for email in result.get("emails", [])[:5]:
                                parts.append(f"  • From: {email.get('from', 'Unknown')} — {email.get('subject', 'No subject')}")
                        elif "email" in result:
                            email = result["email"]
                            parts.append(f"📖 **Email**: {email.get('subject', '')}")
                            parts.append(f"  From: {email.get('from', '')}")
                            parts.append(f"  {email.get('body', '')[:300]}")
                        elif "draft_id" in result:
                            parts.append(f"✏️ **Draft created**: {result.get('draft_id', '')}")
                            preview = result.get("preview", {})
                            parts.append(f"  To: {preview.get('to', '')}")
                            parts.append(f"  Subject: {preview.get('subject', '')}")
                        elif "message" in result:
                            parts.append(f"✅ {result['message']}")
                        else:
                            parts.append(json.dumps(result, indent=2))
                except (json.JSONDecodeError, TypeError):
                    parts.append(str(item["result"])[:300])
        
        return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Agent Registry & Router
# ──────────────────────────────────────────────────────────────────────────────

class AgentRouter:
    """
    Routes planner decisions to the correct domain agent.
    
    The planner says "use the gmail agent" → router finds gmail agent → 
    gmail agent thinks and calls its own tools.
    """
    
    def __init__(self):
        self._agents: Dict[str, DomainAgent] = {}
    
    def register(self, agent: DomainAgent):
        """Register a domain agent"""
        self._agents[agent.name] = agent
        logger.info(f"Registered domain agent: {agent.name} ({len(agent.tools)} tools)")
    
    def get_agent(self, name: str) -> Optional[DomainAgent]:
        """Get agent by name"""
        return self._agents.get(name)
    
    def list_agents(self) -> List[Dict[str, str]]:
        """List all registered agents with descriptions"""
        return [
            {
                "name": a.name,
                "description": a.description,
                "tools": list(a.tools.keys()),
                "tool_count": len(a.tools),
            }
            for a in self._agents.values()
        ]
    
    def get_agent_descriptions(self) -> str:
        """
        Get high-level agent descriptions for the planner prompt.

        SECURITY / TOKEN BEST PRACTICE:
        - Planner only sees agent name, domain description, and tool count.
        - Full tool schemas are NEVER exposed to the planner.
        - Each agent receives only its own tool schemas at execution time.
        This limits blast radius, reduces token usage, and prevents prompt
        injection through tool schema contents.
        """
        lines = []
        for agent in self._agents.values():
            tool_count = len(agent.tools)
            tool_summary = f"{tool_count} tool{'s' if tool_count != 1 else ''}"
            lines.append(f'- "{agent.name}": {agent.description} ({tool_summary})')
        return "\n".join(lines)
    
    async def route(self, agent_name: str, task: str, request_id: str = None) -> str:
        """Route a task to a domain agent"""
        agent = self._agents.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found. Available: {list(self._agents.keys())}"
        
        tracer = get_tracer(request_id)
        tracer.route("router", f"Routing to {agent_name} agent")
        
        return await agent.execute(task, request_id)


# ──────────────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────────────

_router: Optional[AgentRouter] = None

def get_router() -> AgentRouter:
    """Get the global agent router"""
    global _router
    if _router is None:
        _router = AgentRouter()
    return _router
