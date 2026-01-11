"""
FastAPI Main Application.

Entry point for the chatbot API server.

Run with: uvicorn server.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import config
from .routers import chat, tools, graph

# Create FastAPI app
app = FastAPI(
    title="Chatbot API",
    description="LangGraph-based chatbot API with text and voice support",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(tools.router, prefix="/tools", tags=["tools"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "voice": "/chat/voice",
            "stream": "/chat/stream",
            "tools": "/tools",
            "graph": "/graph",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )
