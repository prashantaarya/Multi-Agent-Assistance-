# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.api import router
import uvicorn
import logging
import sys
import os
from contextlib import asynccontextmanager

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jarvis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting J.A.R.V.I.S Multi-Agent Assistant...")
    logger.info("✅ AutoGen v0.4 initialized")
    
    # Verify environment
    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY not found!")
        raise RuntimeError("GROQ_API_KEY required")
    
    logger.info("✅ Environment validated")
    logger.info("🎯 J.A.R.V.I.S is ready for requests")
    
    yield
    
    # Shutdown  
    logger.info("🛑 Shutting down J.A.R.V.I.S...")
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="J.A.R.V.I.S Multi-Agent Assistant",
    description="Advanced AI Assistant powered by AutoGen v0.4 + FastAPI",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router, prefix="/api/v1", tags=["JARVIS"])

# Root endpoint
@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "🤖 J.A.R.V.I.S Multi-Agent Assistant",
        "version": "2.0.0",
        "autogen": "v0.4",
        "status": "online",
        "endpoints": {
            "ask": "/api/v1/ask",
            "ask_direct": "/api/v1/ask-direct", 
            "health": "/api/v1/health",
            "agents": "/api/v1/agents",
            "docs": "/docs"
        },
        "description": "Your personal AI assistant with specialized agents"
    }

@app.get("/status")
async def system_status():
    """System status check"""
    return {
        "jarvis": "online",
        "autogen_version": "v0.4",
        "agents": ["planner", "task", "tool", "api"],
        "features": [
            "Multi-agent delegation",
            "Async processing",
            "Rate limiting",
            "Error handling"
        ]
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status": "error",
            "code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "status": "error"
        }
    )

def main():
    """Run the application"""
    try:
        logger.info("🚀 Launching J.A.R.V.I.S server...")
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("👋 J.A.R.V.I.S stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()