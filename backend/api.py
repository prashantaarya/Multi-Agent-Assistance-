# backend/api.py
"""
JARVIS API Backend with Structured Error Handling

Features:
- RESTful endpoints for all JARVIS capabilities
- Structured error responses with error codes
- Rate limiting and correlation IDs
- OpenAPI documentation
"""

import time
import asyncio
import logging
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents.base_agents import jarvis
from core.capabilities import (
    list_tools, 
    generate_openai_tools, 
    generate_anthropic_tools,
    get_tools_by_agent,
    get_tools_by_category
)
from core.errors import JARVISError, ErrorHandler
from agents.memory import get_memory, MemoryMessage

# -----------------------------------------------------------------------------
# Router & logging
# -----------------------------------------------------------------------------
router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Correlation ID helpers (minimal & self-contained)
# -----------------------------------------------------------------------------
REQUEST_ID_HEADER = "X-Request-ID"

def _new_request_id() -> str:
    return uuid.uuid4().hex

def _get_request_id(req: Optional[Request]) -> str:
    try:
        hdr = req.headers.get(REQUEST_ID_HEADER) if req else None
        return hdr.strip() if hdr else _new_request_id()
    except Exception:
        return _new_request_id()

# -----------------------------------------------------------------------------
# Simple rate limiting (per-process; keep as-is)
# -----------------------------------------------------------------------------
REQUEST_COOLDOWN = 2  # seconds
_last_request_time = {}

def apply_rate_limit(client_id: str = "default") -> None:
    """Apply best-effort rate limiting."""
    now = time.time()
    if client_id in _last_request_time:
        delta = now - _last_request_time[client_id]
        if delta < REQUEST_COOLDOWN:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Wait {REQUEST_COOLDOWN - delta:.1f}s",
            )
    _last_request_time[client_id] = now

# -----------------------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------------------
class AskRequest(BaseModel):
    message: str
    agent: str = "auto"  # "auto", "planner", "task", "tool", "api", "search"

class AskResponse(BaseModel):
    response: str
    agent_used: str
    status: str = "success"
    request_id: str

class HealthResponse(BaseModel):
    status: str
    agents_available: List[str]
    version: str
    request_id: str

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.post("/ask", response_model=AskResponse)
async def ask_jarvis(payload: AskRequest, request: Request):
    """
    Main endpoint for JARVIS interactions (auto or directed agent selection).
    Adds a per-request correlation ID and logs with it.
    """
    rid = _get_request_id(request)
    try:
        user_message = (payload.message or "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        apply_rate_limit()

        logger.info(f"[rid={rid}] /ask received; agent={payload.agent}; msg='{user_message[:120]}'")

        if payload.agent == "auto":
            response_text = await jarvis.process_request(user_message, request_id=rid)
            agent_used = "multi-agent"
        else:
            response_text = await jarvis.chat_direct(user_message, payload.agent, request_id=rid)
            agent_used = payload.agent

        # ✅ Log a preview of the final answer to help you see the outcome
        preview = response_text.replace("\n", " ")[:200]
        logger.info(f"[rid={rid}] /ask answer: {preview}")

        logger.info(f"[rid={rid}] /ask success; agent_used={agent_used}; len={len(response_text)}")
        return AskResponse(response=response_text, agent_used=agent_used, status="success", request_id=rid)

    except asyncio.TimeoutError:
        logger.warning(f"[rid={rid}] /ask timeout")
        raise HTTPException(status_code=408, detail="Request timeout")
    except HTTPException:
        logger.warning(f"[rid={rid}] /ask client error")
        raise
    except Exception as e:
        logger.error(f"[rid={rid}] /ask error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.post("/ask-direct", response_model=AskResponse)
async def ask_direct(payload: AskRequest, request: Request):
    """
    Direct chat with a specific agent (bypasses planner).
    """
    rid = _get_request_id(request)
    try:
        user_message = (payload.message or "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        agent_name = payload.agent if payload.agent != "auto" else "planner"
        apply_rate_limit()

        logger.info(f"[rid={rid}] /ask-direct; agent={agent_name}; msg='{user_message[:120]}'")

        response_text = await jarvis.chat_direct(user_message, agent_name, request_id=rid)

        # ✅ Log the answer preview here too
        preview = response_text.replace("\n", " ")[:200]
        logger.info(f"[rid={rid}] /ask-direct answer: {preview}")

        logger.info(f"[rid={rid}] /ask-direct success; agent_used={agent_name}; len={len(response_text)}")
        return AskResponse(response=response_text, agent_used=agent_name, status="success", request_id=rid)

    except Exception as e:
        logger.error(f"[rid={rid}] /ask-direct error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check; includes correlation ID so you can trace checks in logs.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /health")
    return HealthResponse(
        status="healthy",
        agents_available=list(jarvis.agents.keys()),
        version="2.0-v0.4",
        request_id=rid,
    )

@router.get("/agents")
async def list_agents(request: Request):
    """
    List available agents (with brief description). Adds correlation ID to logs.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /agents")
    return {
        "request_id": rid,
        "agents": {
            name: {
                "name": name,
                "description": (
                    agent.system_message[:100] + "..."
                    if len(getattr(agent, "system_message", "") or "") > 100
                    else (getattr(agent, "system_message", "") or "")
                ),
                "status": "active",
            }
            for name, agent in jarvis.agents.items()
        },
        "total": len(jarvis.agents),
    }


@router.get("/tools")
async def list_all_tools(request: Request):
    """
    List all registered tools with their schemas (Industry-standard format).
    Useful for debugging and understanding available capabilities.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /tools")
    
    tools = list_tools()
    return {
        "request_id": rid,
        "total": len(tools),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "agent": t.agent_name,
                "category": t.category,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type.value,
                        "description": p.description,
                        "required": p.required,
                        "default": p.default,
                        "enum": p.enum
                    }
                    for p in t.parameters
                ],
                "examples": t.examples
            }
            for t in tools
        ]
    }


@router.get("/tools/openai")
async def get_openai_tools(request: Request):
    """
    Get tools in OpenAI function-calling format.
    Can be used directly with OpenAI API or compatible providers.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /tools/openai")
    return {
        "request_id": rid,
        "format": "openai",
        "tools": generate_openai_tools()
    }


@router.get("/tools/anthropic")
async def get_anthropic_tools(request: Request):
    """
    Get tools in Anthropic tool-use format.
    Can be used directly with Anthropic Claude API.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /tools/anthropic")
    return {
        "request_id": rid,
        "format": "anthropic",
        "tools": generate_anthropic_tools()
    }


@router.get("/tools/agent/{agent_name}")
async def get_tools_for_agent(agent_name: str, request: Request):
    """
    Get all tools registered by a specific agent.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /tools/agent/{agent_name}")
    
    tools = get_tools_by_agent(agent_name)
    if not tools:
        raise HTTPException(status_code=404, detail=f"No tools found for agent '{agent_name}'")
    
    return {
        "request_id": rid,
        "agent": agent_name,
        "total": len(tools),
        "tools": [t.to_openai_format() for t in tools]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Advanced Orchestration Endpoints
# ──────────────────────────────────────────────────────────────────────────────

class ReactRequest(BaseModel):
    """Request for ReAct multi-step reasoning"""
    message: str
    max_iterations: int = 5


class CollaborateRequest(BaseModel):
    """Request for agent-to-agent collaboration"""
    from_agent: str
    to_agent: str
    message: str


class ParallelCapability(BaseModel):
    """Single capability for parallel execution"""
    capability: str
    inputs: Dict[str, Any] = {}
    label: Optional[str] = None


class ParallelRequest(BaseModel):
    """Request for parallel execution of multiple capabilities"""
    capabilities: List[ParallelCapability]


@router.post("/ask-react")
async def ask_react(payload: ReactRequest, request: Request):
    """
    Force ReAct (Reasoning + Acting) multi-step mode for complex queries.
    
    This mode uses iterative reasoning:
    1. OBSERVE: Analyze available information
    2. THINK: Reason about the approach
    3. ACT: Execute a capability
    4. REFLECT: Evaluate the result
    5. Repeat until answer is found or max iterations
    
    Best for:
    - Complex research questions
    - Multi-part queries
    - Questions requiring analysis and comparison
    """
    rid = _get_request_id(request)
    try:
        user_message = (payload.message or "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        apply_rate_limit()
        logger.info(f"[rid={rid}] /ask-react; msg='{user_message[:120]}'")

        response_text = await jarvis.process_request_react(user_message, request_id=rid)

        preview = response_text.replace("\n", " ")[:200]
        logger.info(f"[rid={rid}] /ask-react answer: {preview}")

        return AskResponse(
            response=response_text, 
            agent_used="react-reasoner", 
            status="success", 
            request_id=rid
        )

    except Exception as e:
        logger.error(f"[rid={rid}] /ask-react error: {e}")
        raise HTTPException(status_code=500, detail=f"ReAct error: {str(e)}")


@router.post("/ask-parallel")
async def ask_parallel(payload: ParallelRequest, request: Request):
    """
    Execute multiple capabilities in parallel for faster responses.
    
    This mode runs all specified capabilities simultaneously using asyncio.gather().
    Much faster than sequential execution for independent tasks.
    
    Example request:
    ```json
    {
        "capabilities": [
            {"capability": "weather.read", "inputs": {"city": "Delhi"}, "label": "Delhi Weather"},
            {"capability": "news.fetch", "inputs": {"topic": "tech"}, "label": "Tech News"}
        ]
    }
    ```
    
    Benefits:
    - 2 tasks that take 2s each = 2s total (not 4s)
    - Perfect for dashboard-style queries
    - Each result is labeled for easy identification
    """
    rid = _get_request_id(request)
    try:
        if not payload.capabilities:
            raise HTTPException(status_code=400, detail="At least one capability is required")

        apply_rate_limit()
        
        cap_names = [c.capability for c in payload.capabilities]
        logger.info(f"[rid={rid}] /ask-parallel; capabilities={cap_names}")

        # Convert to dict format expected by jarvis
        caps = [
            {
                "capability": c.capability,
                "inputs": c.inputs,
                "label": c.label
            }
            for c in payload.capabilities
        ]

        result = await jarvis.process_request_parallel(caps, request_id=rid)

        logger.info(f"[rid={rid}] /ask-parallel completed; {len(payload.capabilities)} capabilities")

        return {
            "request_id": rid,
            "status": "success",
            "mode": "parallel",
            "results": result.get("results", []),
            "formatted_response": result.get("formatted", "")
        }

    except Exception as e:
        logger.error(f"[rid={rid}] /ask-parallel error: {e}")
        raise HTTPException(status_code=500, detail=f"Parallel execution error: {str(e)}")


@router.post("/collaborate")
async def agent_collaborate(payload: CollaborateRequest, request: Request):
    """
    Enable agent-to-agent communication for collaborative tasks.
    
    One agent can request help from another agent.
    Useful for complex workflows where multiple specialists are needed.
    """
    rid = _get_request_id(request)
    try:
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        logger.info(f"[rid={rid}] /collaborate; {payload.from_agent} -> {payload.to_agent}")

        response_text = await jarvis.collaborate(
            from_agent=payload.from_agent,
            to_agent=payload.to_agent,
            message=payload.message,
            request_id=rid
        )

        return {
            "request_id": rid,
            "from_agent": payload.from_agent,
            "to_agent": payload.to_agent,
            "response": response_text,
            "status": "success"
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[rid={rid}] /collaborate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestration/status")
async def orchestration_status(request: Request):
    """
    Get the current orchestration configuration and capabilities.
    Shows which advanced patterns are enabled.
    """
    rid = _get_request_id(request)
    return {
        "request_id": rid,
        "patterns": {
            "single_hop": {
                "enabled": True,
                "description": "Planner delegates to one agent (default mode)"
            },
            "parallel": {
                "enabled": jarvis.enable_parallel,
                "description": "Execute multiple capabilities simultaneously",
                "endpoint": "/ask-parallel",
                "trigger": "User requests multiple independent tasks (e.g., 'get X and Y')"
            },
            "react": {
                "enabled": jarvis.enable_react,
                "description": "Multi-step reasoning (Observe -> Think -> Act -> Reflect)",
                "max_iterations": 5,
                "trigger": "Complex queries or low planner confidence"
            },
            "collaboration": {
                "enabled": jarvis.enable_collaboration,
                "description": "Agents can invoke other agents"
            },
            "supervision": {
                "enabled": jarvis.enable_supervision,
                "description": "Quality control review of agent outputs"
            }
        },
        "agents_available": list(jarvis.agents.keys()),
        "react_confidence_threshold": jarvis.react_confidence_threshold
    }


@router.post("/test")
async def test_connection(request: Request):
    """
    Quick connectivity test through the full stack; includes request_id.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /test starting")
    try:
        test_message = "Hello JARVIS, are you working?"
        response = await jarvis.process_request(test_message, request_id=rid)
        preview = response.replace("\n", " ")[:200]
        logger.info(f"[rid={rid}] /test answer: {preview}")
        logger.info(f"[rid={rid}] /test success; len={len(response)}")
        return {
            "request_id": rid,
            "status": "success",
            "test_message": test_message,
            "response": response,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"[rid={rid}] /test error: {e}")
        return {
            "request_id": rid,
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Memory Endpoints (Phase 3: Second Brain)
# ──────────────────────────────────────────────────────────────────────────────

class MemoryLearnRequest(BaseModel):
    """Request to teach JARVIS something"""
    content: str
    source: str = "knowledge"  # "knowledge", "user_fact", "task"


class MemorySearchRequest(BaseModel):
    """Request to search memories"""
    query: str
    top_k: int = 5
    source_filter: Optional[str] = None


class SessionRequest(BaseModel):
    """Request for session operations"""
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@router.get("/memory/stats")
async def memory_stats(request: Request):
    """
    Get memory system statistics.
    Shows conversation sessions and semantic memory entries.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /memory/stats")
    
    memory = get_memory()
    return {
        "request_id": rid,
        **memory.stats()
    }


@router.get("/memory/sessions")
async def list_sessions(request: Request, limit: int = 20):
    """
    List recent conversation sessions.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /memory/sessions")
    
    memory = get_memory()
    return {
        "request_id": rid,
        "sessions": memory.conversation.list_sessions(limit=limit)
    }


@router.post("/memory/sessions/create")
async def create_session(payload: SessionRequest, request: Request):
    """
    Create a new conversation session.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /memory/sessions/create")
    
    memory = get_memory()
    session = memory.conversation.create_session(user_id=payload.user_id)
    
    return {
        "request_id": rid,
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "status": "created"
    }


@router.get("/memory/sessions/{session_id}")
async def get_session(session_id: str, request: Request, limit: int = 20):
    """
    Get messages from a specific session.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /memory/sessions/{session_id}")
    
    memory = get_memory()
    session = memory.conversation.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    messages = session.get_recent(limit)
    return {
        "request_id": rid,
        "session_id": session_id,
        "message_count": len(session.messages),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "capability": m.capability,
                "confidence": m.confidence
            }
            for m in messages
        ]
    }


@router.delete("/memory/sessions/{session_id}")
async def clear_session(session_id: str, request: Request):
    """
    Clear all messages from a session.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /memory/sessions/{session_id} DELETE")
    
    memory = get_memory()
    memory.conversation.clear_session(session_id)
    
    return {
        "request_id": rid,
        "session_id": session_id,
        "status": "cleared"
    }


@router.post("/memory/learn")
async def learn(payload: MemoryLearnRequest, request: Request):
    """
    Teach JARVIS something new (add to long-term memory).
    
    Examples:
    - {"content": "The user prefers Python over JavaScript", "source": "user_fact"}
    - {"content": "API rate limit for weather.com is 100 req/min", "source": "knowledge"}
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /memory/learn; source={payload.source}")
    
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    memory = get_memory()
    entry = memory.semantic.add(payload.content, source=payload.source)
    
    return {
        "request_id": rid,
        "status": "learned",
        "entry_id": entry.id,
        "source": payload.source
    }


@router.post("/memory/search")
async def search_memory(payload: MemorySearchRequest, request: Request):
    """
    Search long-term memory for relevant information.
    
    Uses semantic similarity to find related memories.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /memory/search; query='{payload.query[:60]}'")
    
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    memory = get_memory()
    results = memory.semantic.search(
        query=payload.query,
        top_k=payload.top_k,
        source_filter=payload.source_filter
    )
    
    return {
        "request_id": rid,
        "query": payload.query,
        "results": [
            {
                "content": r.content,
                "score": round(r.score, 4),
                "source": r.source,
                "timestamp": r.timestamp.isoformat()
            }
            for r in results
        ]
    }


@router.get("/memory/context")
async def get_context(request: Request, query: str = "", session_id: Optional[str] = None, max_messages: int = 5):
    """
    Get combined context for a query.
    
    Returns relevant conversation history and semantic memories.
    Useful for understanding what JARVIS "knows" about a topic.
    """
    rid = _get_request_id(request)
    logger.info(f"[rid={rid}] /memory/context; query='{query[:60]}'")
    
    memory = get_memory()
    context = memory.get_context_for_prompt(
        query=query,
        session_id=session_id,
        max_conversation_messages=max_messages
    )
    
    return {
        "request_id": rid,
        "query": query,
        "context": context,
        "context_length": len(context)
    }


@router.delete("/memory/semantic")
async def clear_semantic_memory(request: Request):
    """
    Clear all semantic (long-term) memory.
    USE WITH CAUTION - this deletes all learned information.
    """
    rid = _get_request_id(request)
    logger.warning(f"[rid={rid}] /memory/semantic DELETE - clearing all semantic memory")
    
    memory = get_memory()
    memory.semantic.clear()
    
    return {
        "request_id": rid,
        "status": "cleared",
        "warning": "All semantic memories have been deleted"
    }