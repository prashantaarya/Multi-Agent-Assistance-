# core/capabilities.py
from typing import Callable, Dict, Tuple, Optional, List, Any
from core.schemas import ToolSchema, ToolParameter, ParameterType

# ──────────────────────────────────────────────────────────────────────────────
# Industry-Standard Tool Registry with Schemas
# ──────────────────────────────────────────────────────────────────────────────

# capability_name -> ToolSchema (includes agent_name and handler)
_REGISTRY: Dict[str, ToolSchema] = {}


def register(
    capability: str,
    agent_name: str,
    handler: Callable,
    description: str = "",
    parameters: List[ToolParameter] = None,
    category: str = None,
    examples: List[Dict[str, Any]] = None
) -> None:
    """
    Register a capability with full schema (OpenAI/Anthropic compatible).
    
    Example:
        register(
            capability="todo.add",
            agent_name="task",
            handler=task_agent._cap_add,
            description="Add a new task to the to-do list",
            parameters=[
                ToolParameter(name="description", type=ParameterType.STRING, 
                             description="The task description", required=True)
            ],
            category="productivity",
            examples=[{"description": "Buy groceries"}]
        )
    """
    tool = ToolSchema(
        name=capability,
        agent_name=agent_name,
        handler=handler,
        description=description or f"Execute {capability}",
        parameters=parameters or [],
        category=category,
        examples=examples or []
    )
    _REGISTRY[capability] = tool


def register_tool(tool: ToolSchema) -> None:
    """Register a pre-built ToolSchema directly"""
    _REGISTRY[tool.name] = tool


def resolve(capability: str) -> Optional[Tuple[str, Callable]]:
    """Resolve capability to (agent_name, handler) - backward compatible"""
    tool = _REGISTRY.get(capability)
    if tool and tool.handler:
        return (tool.agent_name, tool.handler)
    return None


def get_tool(capability: str) -> Optional[ToolSchema]:
    """Get the full ToolSchema for a capability"""
    return _REGISTRY.get(capability)


def list_capabilities() -> Dict[str, Tuple[str, Callable]]:
    """Backward compatible: Returns {capability: (agent_name, handler)}"""
    return {
        name: (tool.agent_name, tool.handler)
        for name, tool in _REGISTRY.items()
        if tool.handler
    }


def list_tools() -> List[ToolSchema]:
    """Get all registered tools with full schemas"""
    return list(_REGISTRY.values())


def get_tools_by_agent(agent_name: str) -> List[ToolSchema]:
    """Get all tools owned by a specific agent"""
    return [t for t in _REGISTRY.values() if t.agent_name == agent_name]


def get_tools_by_category(category: str) -> List[ToolSchema]:
    """Get all tools in a category"""
    return [t for t in _REGISTRY.values() if t.category == category]


def generate_openai_tools() -> List[Dict[str, Any]]:
    """Generate OpenAI-compatible tool definitions for all registered tools"""
    return [tool.to_openai_format() for tool in _REGISTRY.values()]


def generate_anthropic_tools() -> List[Dict[str, Any]]:
    """Generate Anthropic-compatible tool definitions for all registered tools"""
    return [tool.to_anthropic_format() for tool in _REGISTRY.values()]


def generate_planner_prompt() -> str:
    """
    Generate the capabilities section for the planner's system prompt.
    Auto-updates as new tools are registered.
    """
    lines = ["Available capabilities:"]
    
    # Group by category
    by_category: Dict[str, List[ToolSchema]] = {}
    for tool in _REGISTRY.values():
        cat = tool.category or "general"
        by_category.setdefault(cat, []).append(tool)
    
    for category, tools in sorted(by_category.items()):
        lines.append(f"\n## {category.upper()}")
        for tool in tools:
            lines.append(tool.to_planner_prompt())
    
    return "\n".join(lines)
