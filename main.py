# main.py
import logging
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.api import router

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("jarvis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Quiet noisy libraries
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("autogen_core.events").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Optional: quiet uvicorn access logs if desired
# logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Lifespan hooks
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting J.A.R.V.I.S Multi-Agent Assistant...")
    logger.info("✅ AutoGen v0.4 initialized")

    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY not found!")
        raise RuntimeError("GROQ_API_KEY required")

    logger.info("✅ Environment validated")
    logger.info("🎯 J.A.R.V.I.S is ready for requests")

    yield

    # Shutdown
    logger.info("🛑 Shutting down J.A.R.V.I.S...")
    logger.info("✅ Shutdown complete")

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="J.A.R.V.I.S Multi-Agent Assistant",
    description="Advanced AI Assistant powered by AutoGen v0.4 + FastAPI",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1", tags=["JARVIS"])

# -----------------------------------------------------------------------------
# Basic endpoints & exception handlers
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
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
            "docs": "/docs",
        },
        "description": "Your personal AI assistant with specialized agents",
    }

@app.get("/status")
async def system_status():
    return {
        "jarvis": "online",
        "autogen_version": "v0.4",
        "agents": ["planner", "task", "tool", "api", "search"],
        "features": [
            "Multi-agent delegation",
            "Async processing",
            "Rate limiting",
            "Correlation IDs & tracing",
            "Error handling",
        ],
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status": "error",
            "code": exc.status_code,
        },
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "status": "error",
        },
    )

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    try:
        logger.info("🚀 Launching J.A.R.V.I.S server...")
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info",
            # access_log=False,  # enable to quiet HTTP access logs
        )
    except KeyboardInterrupt:
        logger.info("👋 J.A.R.V.I.S stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
