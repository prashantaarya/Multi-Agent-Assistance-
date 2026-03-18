# core/orchestration.py
"""
Advanced Agent Communication Patterns (Industry Best Practice)

This module implements three key patterns:
1. ReAct Loop - Multi-step reasoning (Observe → Think → Act → Repeat)
2. Collaborative Agents - Agents can invoke other agents
3. Supervisor Pattern - Quality control on agent outputs

These patterns enable JARVIS to handle complex, multi-step tasks.
"""

import json
import logging
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat

from core.schemas import (
    ReActPlan, ReActStep, ThoughtType,
    AgentMessage, SupervisorReview, WorkflowState,
    PlannerDecision
)
from core.capabilities import resolve, list_tools, get_tool
from core.tracing import get_tracer

logger = logging.getLogger("jarvis.orchestration")


# ──────────────────────────────────────────────────────────────────────────────
# ReAct Loop Implementation (Improved v2.0)
# Multi-step reasoning: Observe -> Think -> Act -> Repeat
# With loop prevention, stuck detection, and better prompting
# ──────────────────────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """
You are a ReAct (Reasoning + Acting) agent. You solve complex problems by thinking step-by-step.

## OUTPUT FORMAT (STRICT JSON)
{{
  "thought_type": "observe|think|act|reflect|complete",
  "thought": "<your reasoning - be specific and concise>",
  "action": "<capability_name or null>",
  "action_inputs": {{ "<param_name>": "<actual_value>", ... }},
  "is_final": true/false,
  "final_answer": "<comprehensive answer if is_final=true, else null>",
  "progress_summary": "<brief summary of what you've learned so far>"
}}

## THOUGHT TYPES
1. OBSERVE: Analyze the query and identify what information you need
2. THINK: Plan your approach - what capability should you use?
3. ACT: Execute a capability (MUST set action AND action_inputs with ALL required parameters)
4. REFLECT: Evaluate the result - do you have enough information?
5. COMPLETE: Synthesize all observations into a final answer

## AVAILABLE CAPABILITIES (with required parameters)
{capabilities}

## ⚠️ CRITICAL: ACTION_INPUTS ARE MANDATORY ⚠️
When thought_type="act", you MUST:
1. Set "action" to a valid capability name from the list above
2. Set "action_inputs" with ALL required parameters filled with ACTUAL values

{capability_examples}

❌ NEVER DO THIS: {{"action": "some.capability", "action_inputs": {{}}}}
❌ NEVER DO THIS: {{"action": "some.capability", "action_inputs": {{"param": "<value>"}}}}
✅ ALWAYS DO THIS: {{"action": "some.capability", "action_inputs": {{"param": "actual_value"}}}}

## ERROR RECOVERY
If you see "Error: Missing required inputs for X: [param]":
- You FORGOT to include the required parameter
- On your NEXT step, call the SAME capability but WITH the correct inputs
- Extract the parameter value from the original query

## CRITICAL RULES
1. Maximum {max_iterations} steps - be efficient!
2. Current step: {current_step}/{max_iterations}
3. NEVER repeat the same action with the SAME inputs (loop prevention)
4. After getting data, move to COMPLETE - don't keep fetching
5. If an action fails twice, move to COMPLETE with partial data
6. Always provide progress_summary to track what you've learned

## LOOP PREVENTION
Previous actions taken: {previous_actions}
DO NOT repeat any of these exact action+input combinations.
If you need to retry a failed action, you MUST change the inputs.

## EXAMPLE WORKFLOWS

### Example 1: Single data fetch
Query: "What's the weather in Tokyo?"
Step 1: {{"thought_type":"observe","thought":"User wants Tokyo weather","action":null,"action_inputs":{{}},"is_final":false,"final_answer":null,"progress_summary":"Need Tokyo weather"}}
Step 2: {{"thought_type":"act","thought":"Fetching Tokyo weather","action":"weather.read","action_inputs":{{"city":"Tokyo"}},"is_final":false,"final_answer":null,"progress_summary":"Fetching weather"}}
Step 3: {{"thought_type":"complete","thought":"Got Tokyo weather data","action":null,"action_inputs":{{}},"is_final":true,"final_answer":"The weather in Tokyo is [data from observation]","progress_summary":"Done"}}

### Example 2: Comparison (multiple fetches)
Query: "Compare news about AI vs blockchain"
Step 1: {{"thought_type":"observe","thought":"Need news on two topics for comparison","action":null,"action_inputs":{{}},"is_final":false,"final_answer":null,"progress_summary":"Need AI and blockchain news"}}
Step 2: {{"thought_type":"act","thought":"Getting AI news first","action":"news.fetch","action_inputs":{{"topic":"AI"}},"is_final":false,"final_answer":null,"progress_summary":"Fetching AI news"}}
Step 3: {{"thought_type":"act","thought":"Now getting blockchain news","action":"news.fetch","action_inputs":{{"topic":"blockchain"}},"is_final":false,"final_answer":null,"progress_summary":"Have AI news, fetching blockchain"}}
Step 4: {{"thought_type":"complete","thought":"Have both topics","action":null,"action_inputs":{{}},"is_final":true,"final_answer":"AI News: [data]\\n\\nBlockchain News: [data]\\n\\nComparison: [analysis]","progress_summary":"Comparison complete"}}

### Example 3: Search query
Query: "Who invented Python?"
Step 1: {{"thought_type":"act","thought":"Searching for Python inventor","action":"search.web","action_inputs":{{"query":"who invented Python programming language"}},"is_final":false,"final_answer":null,"progress_summary":"Searching"}}
Step 2: {{"thought_type":"complete","thought":"Found answer","action":null,"action_inputs":{{}},"is_final":true,"final_answer":"Python was created by Guido van Rossum...","progress_summary":"Done"}}

IMPORTANT: Output ONLY valid JSON. No markdown, no extra text, no code fences.
""".strip()


class ReActOrchestrator:
    """
    Implements the ReAct (Reasoning + Acting) pattern for multi-step reasoning.
    
    Improvements in v2.0:
    - Loop prevention: Tracks previous actions to avoid repetition
    - Stuck detection: Detects when agent is making no progress
    - Better prompting: Clearer examples and rules
    - Progress tracking: Each step includes progress summary
    
    Flow:
    1. User query -> Planner decides if ReAct is needed
    2. If complex -> ReAct loop begins
    3. Each iteration: Think -> Act -> Observe -> Reflect
    4. Loop prevention checks before each action
    5. Continue until final answer or max iterations
    """
    
    def __init__(self, model_client, max_iterations: int = 5):
        self.model_client = model_client
        self.max_iterations = max_iterations
        self._react_agent = None
        # Track executed actions for loop prevention
        self._action_history: List[Tuple[str, str]] = []
    
    def _get_capabilities_doc(self) -> str:
        """Get formatted capabilities list for the prompt"""
        tools = list_tools()
        lines = []
        for tool in tools:
            if tool.parameters:
                params = ", ".join(f"{p.name}: {p.type.value}" for p in tool.parameters)
                required = [p.name for p in tool.parameters if p.required]
                req_str = f" [REQUIRED: {', '.join(required)}]" if required else ""
            else:
                params = ""
                req_str = ""
            lines.append(f"- {tool.name}({params}): {tool.description}{req_str}")
        return "\n".join(lines)
    
    def _get_capability_examples(self) -> str:
        """Generate dynamic examples showing correct action_inputs for each capability"""
        tools = list_tools()
        examples = []
        for tool in tools:
            if tool.parameters:
                required = [p for p in tool.parameters if p.required]
                if required:
                    # Build example inputs with placeholder values
                    example_inputs = {}
                    for p in required:
                        if p.type.value == "string":
                            example_inputs[p.name] = f"<{p.name}_value>"
                        elif p.type.value == "integer":
                            example_inputs[p.name] = 123
                        elif p.type.value == "boolean":
                            example_inputs[p.name] = True
                        else:
                            example_inputs[p.name] = f"<{p.name}>"
                    
                    inputs_json = json.dumps(example_inputs).replace('"<', '"ACTUAL_').replace('>"', '"')
                    examples.append(f"- {tool.name} → action_inputs: {inputs_json}")
        
        if not examples:
            return "All capabilities accept empty inputs."
        return "Required inputs per capability:\n" + "\n".join(examples)
    
    def _format_action_history(self) -> str:
        """Format previous actions for loop prevention"""
        if not self._action_history:
            return "None yet"
        return ", ".join(f"{action}({inputs})" for action, inputs in self._action_history[-5:])
    
    def _is_duplicate_action(self, action: str, inputs: Dict[str, Any]) -> bool:
        """Check if this action+inputs combination was already executed"""
        inputs_str = json.dumps(inputs, sort_keys=True)
        return (action, inputs_str) in self._action_history
    
    def _record_action(self, action: str, inputs: Dict[str, Any]):
        """Record an action to prevent loops"""
        inputs_str = json.dumps(inputs, sort_keys=True)
        self._action_history.append((action, inputs_str))
    
    def _try_fill_missing_inputs(
        self,
        action: str,
        action_inputs: Dict[str, Any],
        thought: str,
        original_query: str
    ) -> Dict[str, Any]:
        """
        Auto-fill missing required inputs by extracting values from the LLM's
        own thought text and the original query.

        Solves the common LLM failure where it correctly identifies an action
        (e.g. weather.read for Mumbai) but sends empty action_inputs: {}.
        The city/topic is already present in the thought — we just extract it.
        """
        import re

        schema = get_tool(action)
        if not schema or not schema.parameters:
            return action_inputs

        filled = dict(action_inputs)
        required_missing = [
            p for p in schema.parameters
            if p.required and p.name not in filled
        ]
        if not required_missing:
            return filled  # nothing to fix

        search_text = f"{thought} {original_query}"

        # Build set of values already used for this action (avoid re-fetching)
        used_values: Dict[str, set] = {}
        for act, inp_str in self._action_history:
            if act == action:
                try:
                    for k, v in json.loads(inp_str).items():
                        used_values.setdefault(k, set()).add(str(v).lower())
                except Exception:
                    pass

        for param in required_missing:
            pname = param.name.lower()
            already_used = used_values.get(pname, set())

            if pname == "city":
                # Find Title-Case proper nouns (city names) in thought + query
                common_words = {
                    "the", "we", "you", "he", "she", "it", "they", "a", "an",
                    "user", "fetching", "getting", "current", "weather",
                    "compare", "comparing", "error", "action", "query",
                    "need", "want", "with", "from", "suggest", "better",
                    "picnic", "trip", "visit", "travel", "retrieve", "fetch",
                    "obtain", "collect", "gather", "and", "for", "to",
                }
                candidates = re.findall(r'\b([A-Z][a-zA-Z]{2,})\b', search_text)
                for c in candidates:
                    if c.lower() not in common_words and c.lower() not in already_used:
                        filled[pname] = c
                        logger.info(f"ReAct auto-fill: '{pname}' = '{c}' extracted from context")
                        break

            elif pname in ("query", "search", "keyword"):
                val = None

                # Strategy 1: explicit "searching for X" / "looking up X" / "finding X"
                m = re.search(
                    r'(?:search(?:ing)?\s+for|look(?:ing)?\s+up|find(?:ing)?)\s+(.+?)'
                    r'(?:\s+to\b|\s+from\b|\s*$)',
                    thought, re.IGNORECASE
                )
                if m:
                    val = m.group(1).strip().rstrip(".")

                # Strategy 2: "overview of X" / "about X" / "regarding X" / "on X"
                if not val:
                    m = re.search(
                        r'(?:overview\s+of|about|regarding|on|for|fetch(?:ing)?\s+(?:info|data|details)?\s*(?:on|about|for)?)'
                        r'\s+([A-Za-z][A-Za-z0-9 ]{2,50}?)'
                        r'(?:\s+to\b|\s+in\b|\s+from\b|\.|,|\s*$)',
                        thought, re.IGNORECASE
                    )
                    if m:
                        val = m.group(1).strip().rstrip(".")

                # Strategy 3 (fallback): grab any Title-Case multi-word noun phrase
                # e.g. "Maratha Empire", "Ottoman Empire", "World War II"
                if not val:
                    # Match 1-4 consecutive Title-Case words (capital first letter)
                    noun_phrases = re.findall(
                        r'\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]*|[IVX]+)){0,3})\b',
                        thought
                    )
                    skip = {"fetching", "gathering", "retrieving", "getting",
                            "overview", "concise", "historical", "political",
                            "economic", "military", "cultural", "the", "a", "an",
                            "error", "action", "user", "react", "plan"}
                    for phrase in noun_phrases:
                        if phrase.lower() not in skip and phrase.lower() not in already_used:
                            val = phrase
                            break

                if val and val.lower() not in already_used:
                    filled[pname] = val
                    logger.info(f"ReAct auto-fill: '{pname}' = '{val}' extracted from thought")

            elif pname == "topic":
                m = re.search(
                    r'(?:about|for|on|regarding)\s+([A-Za-z][A-Za-z0-9 ]{2,30}?)'
                    r'(?:\s+(?:news|data|info)|\s*$|\.)',
                    thought, re.IGNORECASE
                )
                if m:
                    val = m.group(1).strip()
                    if val and val.lower() not in already_used:
                        filled[pname] = val
                        logger.info(f"ReAct auto-fill: '{pname}' = '{val}' extracted from thought")

        if filled != action_inputs:
            logger.info(f"ReAct auto-fill for '{action}': {action_inputs} → {filled}")

        return filled

    def _detect_stuck(self, plan: ReActPlan) -> bool:
        """Detect if the agent is stuck (making no progress)"""
        if len(plan.steps) < 2:
            return False
        
        # Check for repeated thoughts (sign of being stuck)
        recent_thoughts = [s.thought.lower()[:50] for s in plan.steps[-3:]]
        if len(recent_thoughts) >= 2 and len(set(recent_thoughts)) == 1:
            return True
        
        # Check for no observations after multiple actions
        recent_steps = plan.steps[-3:]
        actions_without_observations = sum(
            1 for s in recent_steps 
            if s.action and (not s.observation or "Error" in str(s.observation))
        )
        if actions_without_observations >= 2:
            return True
        
        return False
    
    def _create_react_agent(self, current_step: int = 1) -> AssistantAgent:
        """Create the ReAct reasoning agent with current context"""
        system_message = REACT_SYSTEM_PROMPT.format(
            capabilities=self._get_capabilities_doc(),
            capability_examples=self._get_capability_examples(),
            max_iterations=self.max_iterations,
            current_step=current_step,
            previous_actions=self._format_action_history()
        )
        return AssistantAgent(
            name="react_reasoner",
            model_client=self.model_client,
            system_message=system_message
        )
    
    async def execute(self, query: str, request_id: str = None) -> str:
        """
        Execute a ReAct loop for the given query.
        
        Returns the final answer after multi-step reasoning.
        """
        tracer = get_tracer(request_id)
        plan = ReActPlan(original_query=query, max_iterations=self.max_iterations)
        
        # Reset action history for new query
        self._action_history = []
        
        with tracer.span("react.start", query=query[:60]):
            tracer.thought("react", f"Starting multi-step reasoning for: {query[:80]}...")
            context = f"Query: {query}\n\nBegin your reasoning with OBSERVE:"
            
            while plan.current_iteration < plan.max_iterations and not plan.is_complete:
                iteration = plan.current_iteration + 1
                
                # Recreate agent with updated context (loop prevention info)
                self._react_agent = self._create_react_agent(current_step=iteration)
                
                with tracer.span(f"react.iteration_{iteration}"):
                    tracer.thought("react", f"Iteration {iteration}/{self.max_iterations}")
                    
                    # Check for stuck state
                    if self._detect_stuck(plan):
                        tracer.thought("react", "Stuck detected - forcing completion")
                        plan.final_answer = self._synthesize_answer(plan)
                        plan.is_complete = True
                        break
                    
                    # Get next step from ReAct agent
                    step_response = await self._get_next_step(context, plan)
                    
                    if not step_response:
                        tracer.thought("react", "No valid response - synthesizing from collected data")
                        # Don't break immediately - synthesize what we have so far
                        if plan.steps:
                            plan.final_answer = self._synthesize_answer(plan)
                            plan.is_complete = True
                        break
                    
                    thought_type_str = step_response.get("thought_type", "think")
                    tracer.decision("react", f"[{thought_type_str.upper()}] {step_response.get('thought', '')[:100]}")
                    
                    # Check if we're done
                    if step_response.get("is_final"):
                        plan.is_complete = True
                        plan.final_answer = step_response.get("final_answer", "")
                        plan.add_step(
                            thought_type=ThoughtType.COMPLETE,
                            thought=step_response.get("thought", "Task complete"),
                            observation=plan.final_answer
                        )
                        tracer.observation("react", f"Final answer ready ({len(plan.final_answer)} chars)")
                        break
                    
                    # Execute action if specified
                    observation = ""
                    action = step_response.get("action")
                    action_inputs = step_response.get("action_inputs", {})
                    
                    if action:
                        # Auto-fill missing required inputs from the LLM's own thought
                        # and the original query. Fixes the case where the model says
                        # "Fetching Mumbai weather" in thought but sends action_inputs: {}
                        action_inputs = self._try_fill_missing_inputs(
                            action, action_inputs,
                            step_response.get("thought", ""),
                            query
                        )
                        # Check for duplicate action (loop prevention)
                        if self._is_duplicate_action(action, action_inputs):
                            observation = f"[LOOP PREVENTED] This action was already executed. Try a different approach or provide final answer."
                            tracer.thought("react", f"Loop prevented: {action}")
                        else:
                            with tracer.span("react.action", action=action):
                                tracer.action("react", f"Executing {action}")
                                observation = await self._execute_action(action, action_inputs)
                                # Only record successful actions — don't block retries
                                # on calls that failed due to missing/empty inputs
                                if not observation.startswith("Error:"):
                                    self._record_action(action, action_inputs)
                                tracer.observation("react", f"Got: {observation[:100]}...")
                    
                    # Sanitize observation BEFORE storing in plan
                    # This ensures plan.get_context() returns clean text (no bullets/curly quotes)
                    # that won't corrupt the LLM's next JSON output
                    clean_observation = self._sanitize_for_context(observation)
                    
                    # Record step (stored with sanitized observation)
                    thought_type = ThoughtType(thought_type_str) if thought_type_str in [t.value for t in ThoughtType] else ThoughtType.THINK
                    plan.add_step(
                        thought_type=thought_type,
                        thought=step_response.get("thought", ""),
                        action=action,
                        action_inputs=action_inputs,
                        observation=clean_observation
                    )
                    
                    # Build context for next iteration
                    # plan.get_context() already includes the clean observation above
                    progress = self._sanitize_for_context(step_response.get("progress_summary", ""))
                    context = plan.get_context() + f"\n\nProgress so far: {progress}\n\nContinue reasoning (step {iteration + 1}/{self.max_iterations}):"
            
            # If we hit max iterations without completing
            if not plan.is_complete:
                tracer.thought("react", f"Max iterations ({self.max_iterations}) reached - synthesizing answer")
                plan.final_answer = self._synthesize_answer(plan)
                plan.is_complete = True
        
        return plan.final_answer or "Unable to complete the task."
    
    @staticmethod
    def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """
        Robustly parse JSON from LLM output.
        Handles common LLM output issues:
        - Extra text before/after JSON
        - Markdown code fences (```json ... ```)
        - Single quotes instead of double quotes
        - Trailing commas
        - Special characters (bullets •, arrows) inside string values
        """
        if not raw:
            return None
        
        # Strip markdown code fences
        text = raw.strip()
        for fence in ["```json", "```JSON", "```"]:
            if fence in text:
                text = text.split(fence, 1)[-1]
                text = text.rsplit("```", 1)[0]
                text = text.strip()
                break
        
        # Find the outermost { ... } block
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return None
        
        candidate = text[s:e + 1]
        
        # Attempt 1: direct parse
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: fix trailing commas before } or ]
        import re as _re
        cleaned = _re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Attempt 3: replace smart/curly quotes
        cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Attempt 4: extract only safe known fields with regex fallback
        try:
            result: Dict[str, Any] = {}
            for field, pattern in [
                ("thought_type", r'"thought_type"\s*:\s*"([^"]+)"'),
                ("thought",      r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"'),
                ("action",       r'"action"\s*:\s*"([^"]+)"'),
                ("is_final",     r'"is_final"\s*:\s*(true|false)'),
                ("final_answer", r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"'),
            ]:
                m = _re.search(pattern, cleaned, _re.DOTALL)
                if m:
                    val = m.group(1)
                    if field == "is_final":
                        result[field] = val == "true"
                    else:
                        result[field] = val
            if "thought_type" in result or "thought" in result:
                result.setdefault("thought_type", "think")
                result.setdefault("is_final", False)
                result.setdefault("action", None)
                result.setdefault("action_inputs", {})
                result.setdefault("final_answer", None)
                result.setdefault("progress_summary", "")
                return result
        except Exception:
            pass
        
        logger.debug(f"JSON parse failed for: {candidate[:200]}")
        return None

    @staticmethod
    def _sanitize_for_context(text: str) -> str:
        """
        Sanitize observation text before injecting into LLM prompt.
        Removes/replaces characters that corrupt the JSON the LLM writes next.
        """
        # Replace bullet chars with plain dashes
        text = text.replace("\u2022", "-").replace("\u2023", "-")
        # Replace curly quotes
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        # Remove null bytes
        text = text.replace("\x00", "")
        return text

    async def _get_next_step(self, context: str, plan: ReActPlan) -> Optional[Dict[str, Any]]:
        """Get the next reasoning step from the ReAct agent with robust JSON parsing"""
        try:
            team = RoundRobinGroupChat([self._react_agent], max_turns=1)
            buffer = ""
            async for msg in team.run_stream(task=context):
                if getattr(msg, "content", None):
                    buffer += msg.content

            result = self._safe_parse_json(buffer)
            if result is None:
                logger.warning(f"ReAct: JSON parse failed. Raw output (first 300 chars): {buffer[:300]}")
            return result
        except Exception as e:
            logger.error(f"ReAct step failed: {e}")
            return None
    
    async def _execute_action(self, action: str, inputs: Dict[str, Any]) -> str:
        """Execute a capability and return the observation"""
        try:
            resolved = resolve(action)
            if not resolved:
                return f"Error: Capability '{action}' not found. Available: search.web, weather.read, news.fetch, etc."
            
            _, handler = resolved
            
            # Validate required inputs before calling - gives LLM a clear retry hint
            schema = get_tool(action)
            if schema and schema.parameters:
                required_params = [p.name for p in schema.parameters if p.required]
                missing = [p for p in required_params if p not in inputs]
                if missing:
                    hint = ", ".join(f'"{p}": "<value>"' for p in required_params)
                    return (
                        f"Error: Missing required inputs for '{action}': {missing}. "
                        f"Retry with action_inputs: {{{hint}}}"
                    )
            
            result = await (handler(**inputs) if inputs else handler())
            return str(result)
        except Exception as e:
            return f"Error executing {action}: {str(e)}"
    
    def _synthesize_answer(self, plan: ReActPlan) -> str:
        """
        Synthesize a final answer from all collected observations.
        Produces a combined, readable response instead of raw data dumps.
        """
        good_obs = [
            s.observation for s in plan.steps
            if s.observation
            and "Error" not in str(s.observation)
            and "LOOP PREVENTED" not in str(s.observation)
        ]
        
        if not good_obs:
            thoughts = [s.thought for s in plan.steps if s.thought]
            return f"Based on my analysis:\n\n{thoughts[-1]}" if thoughts else "I was unable to find sufficient information."
        
        # Combine all observations with clear labels
        parts = []
        action_steps = [
            s for s in plan.steps
            if s.observation and "Error" not in str(s.observation) and s.action
        ]
        
        for step in action_steps:
            label = step.action.replace(".", " ").title() if step.action else "Result"
            safe = self._sanitize_for_context(step.observation)
            parts.append(f"**{label}**\n{safe}")
        
        combined = "\n\n".join(parts) if parts else self._sanitize_for_context("\n\n".join(good_obs))
        
        # Add the original query as framing
        return f"Here is what I found for: *{plan.original_query}*\n\n{combined}"


# ──────────────────────────────────────────────────────────────────────────────
# Collaborative Agent Communication
# Agents can invoke other agents for sub-tasks
# ──────────────────────────────────────────────────────────────────────────────

class AgentCollaborator:
    """
    Enables agents to communicate and delegate tasks to each other.
    
    Pattern:
    - Agent A receives a complex task
    - Agent A identifies it needs help from Agent B
    - Agent A sends a message to Agent B
    - Agent B processes and returns result
    - Agent A incorporates result into its response
    """
    
    def __init__(self, agents: Dict[str, AssistantAgent]):
        self.agents = agents
        self.message_history: List[AgentMessage] = []
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        message_type: str = "request",
        context: Dict[str, Any] = None,
        request_id: str = None
    ) -> AgentMessage:
        """
        Send a message from one agent to another.
        
        Returns the response message.
        """
        tracer = get_tracer(request_id)
        
        if to_agent not in self.agents:
            raise ValueError(f"Agent '{to_agent}' not found")
        
        # Create request message
        request = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            context=context or {}
        )
        self.message_history.append(request)
        
        with tracer.span("agent.collaborate", from_a=from_agent, to_a=to_agent):
            # Execute the target agent
            target_agent = self.agents[to_agent]
            team = RoundRobinGroupChat([target_agent], max_turns=1)
            
            # Format message with context
            formatted_content = f"[From {from_agent}]: {content}"
            if context:
                formatted_content += f"\n[Context]: {json.dumps(context)}"
            
            buffer = ""
            async for msg in team.run_stream(task=formatted_content):
                if getattr(msg, "content", None):
                    buffer += msg.content
            
            # Create response message
            response = AgentMessage(
                from_agent=to_agent,
                to_agent=from_agent,
                message_type="response",
                content=buffer,
                parent_message_id=request.id
            )
            self.message_history.append(response)
            
            return response
    
    async def delegate_to_capability(
        self,
        from_agent: str,
        capability: str,
        inputs: Dict[str, Any],
        request_id: str = None
    ) -> str:
        """
        Delegate a task to a capability (not a specific agent).
        
        This allows agents to use capabilities without knowing which agent owns them.
        """
        tracer = get_tracer(request_id)
        
        with tracer.span("agent.delegate", from_a=from_agent, capability=capability):
            resolved = resolve(capability)
            if not resolved:
                return f"Error: Capability '{capability}' not found"
            
            agent_name, handler = resolved
            
            # Log the delegation
            delegation = AgentMessage(
                from_agent=from_agent,
                to_agent=agent_name,
                message_type="delegate",
                content=f"Execute {capability}",
                context={"capability": capability, "inputs": inputs}
            )
            self.message_history.append(delegation)
            
            # Execute
            try:
                result = await (handler(**inputs) if inputs else handler())
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
    
    def get_conversation_history(self, agent_name: str = None) -> List[AgentMessage]:
        """Get message history, optionally filtered by agent"""
        if agent_name:
            return [m for m in self.message_history 
                    if m.from_agent == agent_name or m.to_agent == agent_name]
        return self.message_history


# ──────────────────────────────────────────────────────────────────────────────
# Supervisor Pattern
# Quality control and review of agent outputs
# ──────────────────────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """
You are a Supervisor agent. Your job is to review outputs from other agents for quality.

Evaluate the response based on:
1. ACCURACY - Is the information correct?
2. COMPLETENESS - Does it fully answer the question?
3. CLARITY - Is it clear and well-organized?
4. RELEVANCE - Is it relevant to the query?

Output STRICT JSON:
{
  "review_status": "approved|needs_revision|rejected",
  "quality_score": <0.0 to 1.0>,
  "feedback": "<specific feedback if not approved>",
  "revised_output": "<improved version if needs_revision, else null>"
}

If the response is good, set review_status="approved" and quality_score >= 0.8.
If it needs improvement, set review_status="needs_revision" and provide revised_output.
If it's completely wrong, set review_status="rejected" with feedback.
""".strip()


class SupervisorOrchestrator:
    """
    Implements the Supervisor pattern for quality control.
    
    Flow:
    1. Agent produces output
    2. Supervisor reviews the output
    3. If approved → return output
    4. If needs_revision → return revised output
    5. If rejected → retry with different approach or return error
    """
    
    def __init__(self, model_client, enable_review: bool = True):
        self.model_client = model_client
        self.enable_review = enable_review
        self._supervisor_agent = None
        self.review_history: List[SupervisorReview] = []
    
    def _create_supervisor_agent(self) -> AssistantAgent:
        """Create the supervisor agent"""
        return AssistantAgent(
            name="supervisor",
            model_client=self.model_client,
            system_message=SUPERVISOR_SYSTEM_PROMPT
        )
    
    async def review_output(
        self,
        task_id: str,
        agent_name: str,
        original_query: str,
        agent_output: str,
        request_id: str = None
    ) -> SupervisorReview:
        """
        Review an agent's output for quality.
        
        Returns a SupervisorReview with status and optional revision.
        """
        if not self.enable_review:
            # Return auto-approved if review is disabled
            return SupervisorReview(
                task_id=task_id,
                agent_name=agent_name,
                original_output=agent_output,
                review_status="approved",
                quality_score=1.0
            )
        
        tracer = get_tracer(request_id)
        
        if not self._supervisor_agent:
            self._supervisor_agent = self._create_supervisor_agent()
        
        with tracer.span("supervisor.review", agent=agent_name):
            review_prompt = f"""
Review this response:

ORIGINAL QUERY: {original_query}

AGENT ({agent_name}) RESPONSE:
{agent_output}

Evaluate and provide your review in JSON format.
"""
            
            team = RoundRobinGroupChat([self._supervisor_agent], max_turns=1)
            buffer = ""
            async for msg in team.run_stream(task=review_prompt):
                if getattr(msg, "content", None):
                    buffer += msg.content
            
            # Parse review JSON
            try:
                s, e = buffer.find("{"), buffer.rfind("}")
                if s != -1 and e > s:
                    review_data = json.loads(buffer[s:e+1])
                    review = SupervisorReview(
                        task_id=task_id,
                        agent_name=agent_name,
                        original_output=agent_output,
                        review_status=review_data.get("review_status", "approved"),
                        feedback=review_data.get("feedback"),
                        revised_output=review_data.get("revised_output"),
                        quality_score=float(review_data.get("quality_score", 0.8))
                    )
                else:
                    # Default to approved if parsing fails
                    review = SupervisorReview(
                        task_id=task_id,
                        agent_name=agent_name,
                        original_output=agent_output,
                        review_status="approved",
                        quality_score=0.7,
                        feedback="Review parsing failed, auto-approved"
                    )
            except Exception as e:
                logger.error(f"Supervisor review failed: {e}")
                review = SupervisorReview(
                    task_id=task_id,
                    agent_name=agent_name,
                    original_output=agent_output,
                    review_status="approved",
                    quality_score=0.7,
                    feedback=f"Review error: {str(e)}"
                )
            
            self.review_history.append(review)
            return review
    
    def get_final_output(self, review: SupervisorReview) -> str:
        """Get the final output based on review status"""
        if review.review_status == "approved":
            return review.original_output
        elif review.review_status == "needs_revision" and review.revised_output:
            return review.revised_output
        else:
            return f"⚠️ {review.original_output}\n\n[Supervisor Note: {review.feedback}]"


# ──────────────────────────────────────────────────────────────────────────────
# Parallel Executor (Multiple capabilities at once)
# ──────────────────────────────────────────────────────────────────────────────

import asyncio
import time

from core.schemas import CapabilityCall, ParallelResult


class ParallelExecutor:
    """
    Executes multiple capabilities in parallel using asyncio.gather().
    
    Benefits:
    - Faster response for multi-capability queries
    - Better user experience (don't wait for sequential execution)
    - Efficient use of async I/O
    
    Example:
        User: "Get weather in Delhi and latest tech news"
        
        Sequential: weather(2s) -> news(2s) = 4 seconds total
        Parallel:   weather(2s) + news(2s) = 2 seconds total (50% faster!)
    
    Usage:
        executor = ParallelExecutor()
        results = await executor.execute([
            CapabilityCall(capability="weather.read", inputs={"city": "Delhi"}),
            CapabilityCall(capability="news.fetch", inputs={"topic": "tech"})
        ])
    """
    
    def __init__(self, timeout: float = 30.0):
        """
        Args:
            timeout: Maximum time to wait for all parallel tasks (seconds)
        """
        self.timeout = timeout
    
    async def execute(
        self,
        capabilities: List[CapabilityCall],
        request_id: str = None
    ) -> List[ParallelResult]:
        """
        Execute multiple capabilities in parallel.
        
        Args:
            capabilities: List of capabilities to execute
            request_id: Optional request ID for tracing
        
        Returns:
            List of ParallelResult objects with outputs or errors
        """
        tracer = get_tracer(request_id)
        
        if not capabilities:
            return []
        
        with tracer.span("parallel.execute", count=len(capabilities)):
            tracer.thought("parallel", f"Executing {len(capabilities)} capabilities in parallel")
            
            # Create tasks for each capability
            tasks = []
            for cap in capabilities:
                task = self._execute_single(cap, tracer)
                tasks.append(task)
            
            # Execute all tasks in parallel with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                tracer.thought("parallel", f"Timeout after {self.timeout}s")
                # Return timeout errors for incomplete tasks
                results = [
                    ParallelResult(
                        capability=cap.capability,
                        label=cap.label,
                        status="error",
                        error=f"Timeout after {self.timeout}s"
                    )
                    for cap in capabilities
                ]
                return results
            
            # Process results
            final_results = []
            for i, result in enumerate(results):
                cap = capabilities[i]
                if isinstance(result, Exception):
                    final_results.append(ParallelResult(
                        capability=cap.capability,
                        label=cap.label,
                        status="error",
                        error=str(result)
                    ))
                elif isinstance(result, ParallelResult):
                    final_results.append(result)
                else:
                    # Unexpected result type
                    final_results.append(ParallelResult(
                        capability=cap.capability,
                        label=cap.label,
                        status="ok",
                        output=str(result)
                    ))
            
            # Log summary
            successes = sum(1 for r in final_results if r.status == "ok")
            tracer.observation("parallel", f"Completed: {successes}/{len(capabilities)} successful")
            
            return final_results
    
    async def _execute_single(
        self,
        cap: CapabilityCall,
        tracer
    ) -> ParallelResult:
        """Execute a single capability and measure time"""
        start_time = time.time()
        
        try:
            with tracer.span(f"parallel.{cap.capability}", inputs=str(cap.inputs)[:50]):
                resolved = resolve(cap.capability)
                
                if not resolved:
                    return ParallelResult(
                        capability=cap.capability,
                        label=cap.label,
                        status="error",
                        error=f"Capability '{cap.capability}' not found",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                
                _, handler = resolved
                result = await (handler(**cap.inputs) if cap.inputs else handler())
                
                return ParallelResult(
                    capability=cap.capability,
                    label=cap.label,
                    status="ok",
                    output=str(result),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
        
        except Exception as e:
            return ParallelResult(
                capability=cap.capability,
                label=cap.label,
                status="error",
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def format_results(
        self,
        results: List[ParallelResult],
        original_query: str = ""
    ) -> str:
        """
        Format parallel results into a readable response.
        
        Args:
            results: List of ParallelResult objects
            original_query: Original user query for context
        
        Returns:
            Formatted string combining all results
        """
        if not results:
            return "No results"
        
        # Single result - return directly
        if len(results) == 1:
            r = results[0]
            if r.status == "ok":
                return r.output
            else:
                return f"❌ Error: {r.error}"
        
        # Multiple results - combine with labels
        parts = []
        total_time = 0
        
        for r in results:
            total_time += r.execution_time_ms
            label = r.label or r.capability.replace(".", " ").title()
            
            if r.status == "ok":
                parts.append(f"## {label}\n{r.output}")
            else:
                parts.append(f"## {label}\n❌ Error: {r.error}")
        
        # Add timing info
        max_time = max(r.execution_time_ms for r in results)
        parts.append(f"\n---\n*Executed {len(results)} queries in parallel ({max_time:.0f}ms)*")
        
        return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Workflow Orchestrator (Combines all patterns)
# ──────────────────────────────────────────────────────────────────────────────

class WorkflowOrchestrator:
    """
    High-level orchestrator that combines:
    - ReAct for complex reasoning
    - Collaboration for multi-agent tasks
    - Supervision for quality control
    
    Automatically selects the right pattern based on task complexity.
    """
    
    def __init__(
        self,
        model_client,
        agents: Dict[str, AssistantAgent],
        enable_react: bool = True,
        enable_collaboration: bool = True,
        enable_supervision: bool = True,
        react_threshold: float = 0.6
    ):
        self.model_client = model_client
        self.agents = agents
        self.enable_react = enable_react
        self.enable_collaboration = enable_collaboration
        self.enable_supervision = enable_supervision
        self.react_threshold = react_threshold
        
        # Initialize orchestrators
        self.react = ReActOrchestrator(model_client) if enable_react else None
        self.collaborator = AgentCollaborator(agents) if enable_collaboration else None
        self.supervisor = SupervisorOrchestrator(model_client, enable_supervision)
    
    def _is_complex_query(self, query: str, decision: PlannerDecision = None) -> bool:
        """
        Determine if a query needs ReAct multi-step reasoning.
        
        Complex queries typically:
        - Require multiple pieces of information
        - Have multiple sub-questions
        - Need comparison or analysis
        - Low confidence from planner
        """
        # Check confidence
        if decision and decision.confidence < self.react_threshold:
            return True
        
        # Check for complexity indicators
        complexity_indicators = [
            " and ", " then ", " after ",  # Multi-step
            "compare", "difference", "versus", "vs",  # Comparison
            "analyze", "explain why", "how does",  # Analysis
            "step by step", "detailed",  # Explicit complexity
            "multiple", "several", "all the"  # Multiple items
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in complexity_indicators)
    
    async def execute(
        self,
        query: str,
        decision: PlannerDecision = None,
        request_id: str = None
    ) -> str:
        """
        Execute a query using the appropriate pattern(s).
        
        1. Check if ReAct is needed for complex queries
        2. Execute using single-hop or ReAct
        3. Apply supervision if enabled
        4. Return final output
        """
        tracer = get_tracer(request_id)
        state = WorkflowState(original_task=query)
        
        with tracer.span("workflow.execute", query=query[:60]):
            # Determine execution strategy
            use_react = self.enable_react and self._is_complex_query(query, decision)
            
            if use_react:
                # Use ReAct for complex reasoning
                with tracer.span("workflow.react"):
                    output = await self.react.execute(query, request_id)
                    state.agent_outputs["react"] = output
            elif decision:
                # Use single-hop execution
                with tracer.span("workflow.single_hop"):
                    resolved = resolve(decision.capability)
                    if resolved:
                        _, handler = resolved
                        output = await (handler(**decision.inputs) if decision.inputs else handler())
                        state.agent_outputs[decision.capability] = str(output)
                    else:
                        output = "Capability not found"
                        state.errors.append(f"Capability {decision.capability} not resolved")
            else:
                output = "No execution strategy available"
                state.errors.append("No decision provided")
            
            # Apply supervision if enabled
            if self.enable_supervision and output:
                with tracer.span("workflow.supervise"):
                    review = await self.supervisor.review_output(
                        task_id=state.workflow_id,
                        agent_name="workflow",
                        original_query=query,
                        agent_output=output,
                        request_id=request_id
                    )
                    output = self.supervisor.get_final_output(review)
            
            state.final_output = output
            state.is_complete = True
            state.completed_at = datetime.utcnow()
            
            return output
