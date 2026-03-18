# core/schemas.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal, Callable
from uuid import uuid4
from datetime import datetime
from enum import Enum

def new_id() -> str:
    return uuid4().hex


# ──────────────────────────────────────────────────────────────────────────────
# Industry-Standard Tool Schema (OpenAI/Anthropic Compatible)
# ──────────────────────────────────────────────────────────────────────────────

class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Single parameter definition for a tool"""
    name: str
    type: ParameterType = ParameterType.STRING
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None  # For constrained values
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format (OpenAI style)"""
        schema = {
            "type": self.type.value,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


class ToolSchema(BaseModel):
    """
    Industry-standard tool definition compatible with:
    - OpenAI Function Calling
    - Anthropic Tool Use
    - LangChain Tools
    - AutoGen Tools
    """
    name: str                           # e.g., "todo.add"
    description: str                    # Human-readable description
    parameters: List[ToolParameter] = Field(default_factory=list)
    agent_name: str                     # Owning agent
    handler: Optional[Callable] = None  # The actual async function
    category: Optional[str] = None      # e.g., "productivity", "search", "api"
    examples: List[Dict[str, Any]] = Field(default_factory=list)  # Usage examples
    
    class Config:
        arbitrary_types_allowed = True  # Allow Callable type
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def to_planner_prompt(self) -> str:
        """Generate a concise prompt line for the planner"""
        if not self.parameters:
            return f'- "{self.name}": {self.description} | inputs: {{}}'
        
        inputs_example = {p.name: f"<{p.type.value}>" for p in self.parameters}
        return f'- "{self.name}": {self.description} | inputs: {inputs_example}'

class Artifact(BaseModel):
    kind: str
    data: Dict[str, Any]

class Task(BaseModel):
    id: str = Field(default_factory=new_id)
    user_id: Optional[str] = None
    message: str
    # Optional: if planner already knows the capability, it can set it
    capability: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ExecutionMode(str, Enum):
    """Execution mode selected by planner"""
    SINGLE = "single"      # Single capability call (default)
    REACT = "react"        # Multi-step reasoning for complex queries
    PARALLEL = "parallel"  # Multiple capabilities in parallel


class CapabilityCall(BaseModel):
    """Single capability to execute (used in parallel mode)"""
    capability: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[str] = None  # Optional label for referencing in response


class PlannerDecision(BaseModel):
    capability: str  # Primary capability (used in single/react mode)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.7
    fallback: Optional[str] = None  # another capability or "planner"
    mode: ExecutionMode = ExecutionMode.SINGLE  # Execution mode
    reasoning: Optional[str] = None  # Brief reasoning for the decision
    
    # For PARALLEL mode: list of capabilities to execute simultaneously
    parallel_capabilities: List[CapabilityCall] = Field(default_factory=list)
    
    # Internal pointer to logs/thoughts if you want (not exposed)
    notes_ref: Optional[str] = None


class ParallelResult(BaseModel):
    """Result from parallel execution"""
    capability: str
    label: Optional[str] = None
    status: Literal["ok", "error"] = "ok"
    output: str = ""
    error: Optional[str] = None
    execution_time_ms: float = 0


class Result(BaseModel):
    task_id: str
    status: Literal["ok", "error"] = "ok"
    output: str = ""
    artifacts: List[Artifact] = Field(default_factory=list)
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    finished_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# Advanced Agent Communication Patterns (Industry Best Practice)
# ──────────────────────────────────────────────────────────────────────────────

class ThoughtType(str, Enum):
    """Types of thoughts in ReAct loop"""
    OBSERVE = "observe"      # Gathering information
    THINK = "think"          # Reasoning about the situation
    ACT = "act"              # Taking an action
    REFLECT = "reflect"      # Evaluating the result
    COMPLETE = "complete"    # Task is done


class ReActStep(BaseModel):
    """Single step in a ReAct reasoning loop"""
    step_number: int
    thought_type: ThoughtType
    thought: str                          # The reasoning/observation
    action: Optional[str] = None          # Capability to invoke (if ACT)
    action_inputs: Dict[str, Any] = Field(default_factory=dict)
    observation: Optional[str] = None     # Result of the action
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReActPlan(BaseModel):
    """Complete ReAct execution plan"""
    task_id: str = Field(default_factory=new_id)
    original_query: str
    steps: List[ReActStep] = Field(default_factory=list)
    max_iterations: int = 5
    current_iteration: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None
    
    def add_step(self, thought_type: ThoughtType, thought: str, 
                 action: str = None, action_inputs: dict = None, 
                 observation: str = None) -> ReActStep:
        """Add a step to the plan"""
        step = ReActStep(
            step_number=len(self.steps) + 1,
            thought_type=thought_type,
            thought=thought,
            action=action,
            action_inputs=action_inputs or {},
            observation=observation
        )
        self.steps.append(step)
        self.current_iteration += 1
        return step
    
    def get_context(self) -> str:
        """Get formatted context of all steps for LLM"""
        if not self.steps:
            return "No steps taken yet."
        
        lines = [f"Original Query: {self.original_query}\n"]
        for step in self.steps:
            lines.append(f"Step {step.step_number} [{step.thought_type.value.upper()}]:")
            lines.append(f"  Thought: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}")
                if step.action_inputs:
                    lines.append(f"  Inputs: {step.action_inputs}")
            if step.observation:
                lines.append(f"  Observation: {step.observation[:500]}...")
        return "\n".join(lines)


class AgentMessage(BaseModel):
    """Message passed between agents for collaborative communication"""
    id: str = Field(default_factory=new_id)
    from_agent: str
    to_agent: str
    message_type: Literal["request", "response", "delegate", "feedback"] = "request"
    content: str
    context: Dict[str, Any] = Field(default_factory=dict)
    parent_message_id: Optional[str] = None  # For threading
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SupervisorReview(BaseModel):
    """Supervisor review of an agent's output"""
    task_id: str
    agent_name: str
    original_output: str
    review_status: Literal["approved", "needs_revision", "rejected"] = "approved"
    feedback: Optional[str] = None
    revised_output: Optional[str] = None
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    review_timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorkflowState(BaseModel):
    """State for complex multi-agent workflows"""
    workflow_id: str = Field(default_factory=new_id)
    original_task: str
    current_agent: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    pending_steps: List[str] = Field(default_factory=list)
    agent_outputs: Dict[str, str] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    is_complete: bool = False
    final_output: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
