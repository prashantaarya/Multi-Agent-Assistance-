# backend/api.py
import time
import asyncio
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.base_agents import jarvis

# Router Setup
router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Rate Limiting
REQUEST_COOLDOWN = 2  # seconds
last_request_time = {}

# Request/Response Models
class AskRequest(BaseModel):
    message: str
    agent: str = "auto"  # auto, planner, task, tool, api

class AskResponse(BaseModel):
    response: str
    agent_used: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    agents_available: list
    version: str

# Helper Functions
def apply_rate_limit(client_id: str = "default") -> None:
    """Apply rate limiting"""
    current_time = time.time()
    
    if client_id in last_request_time:
        time_diff = current_time - last_request_time[client_id]
        if time_diff < REQUEST_COOLDOWN:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Wait {REQUEST_COOLDOWN - time_diff:.1f}s"
            )
    
    last_request_time[client_id] = current_time

# API Endpoints
@router.post("/ask", response_model=AskResponse)
async def ask_jarvis(payload: AskRequest):
    """Main endpoint for JARVIS interactions"""
    try:
        # Validate input
        user_message = payload.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Rate limiting
        apply_rate_limit()
        
        logger.info(f"Processing request: {user_message[:100]}...")
        
        # Process request
        if payload.agent == "auto":
            response = await jarvis.process_request(user_message)
            agent_used = "multi-agent"
        else:
            response = await jarvis.chat_direct(user_message, payload.agent)
            agent_used = payload.agent
        
        return AskResponse(
            response=response,
            agent_used=agent_used
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Error in ask_jarvis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.post("/ask-direct", response_model=AskResponse)
async def ask_direct(payload: AskRequest):
    """Direct chat with specific agent"""
    try:
        user_message = payload.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        agent_name = payload.agent
        if agent_name == "auto":
            agent_name = "planner"
        
        apply_rate_limit()
        
        response = await jarvis.chat_direct(user_message, agent_name)
        
        return AskResponse(
            response=response,
            agent_used=agent_name
        )
        
    except Exception as e:
        logger.error(f"Error in ask_direct: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agents_available=list(jarvis.agents.keys()),
        version="2.0-v0.4"
    )

@router.get("/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": {
            name: {
                "name": name,
                "description": agent.system_message[:100] + "..." if len(agent.system_message) > 100 else agent.system_message,
                "status": "active"
            }
            for name, agent in jarvis.agents.items()
        },
        "total": len(jarvis.agents)
    }

# Test endpoint for development
@router.post("/test")
async def test_connection():
    """Test JARVIS connection"""
    try:
        test_message = "Hello JARVIS, are you working?"
        response = await jarvis.process_request(test_message)
        return {
            "status": "success",
            "test_message": test_message,
            "response": response,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": time.time()
        }