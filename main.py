"""
main.py
FastAPI application entry point for the SHL Assessment Recommender API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from models import ChatRequest, ChatResponse, HealthResponse, Recommendation
from llm.client import generate_response
from llm.prompts import build_search_query
from retrieval.embedder import load_index, search
from config import MAX_TURNS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Eagerly loads the search index on startup to optimize request latency.
    """
    print("Starting SHL Recommender API...")
    print("Loading search index...")
    load_index()
    print("Server startup complete. Ready to handle requests.")
    yield
    print("Server shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment selection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for deployment validation.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles conversational recommendations by combining RAG retrieval and LLM generation.
    """
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    
    if len(messages) > MAX_TURNS:
        return ChatResponse(
            reply=(
                "We've reached the maximum conversation length. "
                "Based on our discussion, here is my final recommendation. "
                "Please start a new conversation for additional needs."
            ),
            recommendations=[],
            end_of_conversation=True,
        )
    
    search_query = build_search_query(messages)
    
    if search_query.strip():
        retrieved_items = search(search_query)
    else:
        retrieved_items = []
    
    response_dict = generate_response(
        messages=messages,
        retrieved_items=retrieved_items,
    )
    
    recommendations = [
        Recommendation(
            name=rec["name"],
            url=rec["url"],
            test_type=rec["test_type"],
        )
        for rec in response_dict.get("recommendations", [])
    ]
    
    return ChatResponse(
        reply=response_dict.get("reply", ""),
        recommendations=recommendations,
        end_of_conversation=response_dict.get("end_of_conversation", False),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
