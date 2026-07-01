"""
main.py — The FastAPI application entry point.

This is the file that RUNS the web server.
It defines:
  - GET /health  → readiness check
  - POST /chat   → the conversational agent endpoint

To run locally:
    uvicorn main:app --reload --port 8000

Then test at:
    http://localhost:8000/health
    http://localhost:8000/docs   ← auto-generated API docs (SwaggerUI)
"""

from fastapi import FastAPI, HTTPException    # FastAPI = framework, HTTPException = error responses
from fastapi.middleware.cors import CORSMiddleware  # CORS = allows browser requests from any origin
from contextlib import asynccontextmanager   # for startup/shutdown events

from models import ChatRequest, ChatResponse, HealthResponse, Recommendation
from llm.client import generate_response
from llm.prompts import build_search_query
from retrieval.embedder import load_index, search
from config import MAX_TURNS


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE when the server starts (before any requests).
    
    Why load the FAISS index on startup?
    - Loading takes ~2 seconds (reads from disk, initializes).
    - If we loaded it inside /chat, every request would be slow.
    - By loading once at startup, every request is fast.
    
    This is called "eager initialization" — load expensive resources early.
    """
    print("Starting SHL Recommender API...")
    print("Loading FAISS index...")
    load_index()    # load the vector index from disk into memory
    print("Ready! Server is live.")
    
    yield   # this is where the server runs (handles all requests)
    
    # Code after yield runs on shutdown (cleanup)
    print("Server shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────

# Create the FastAPI app
# lifespan = our startup function defined above
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment selection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
# CORS = Cross-Origin Resource Sharing
# Without this, browsers would block requests from other domains.
# allow_origins=["*"] = accept requests from ANY domain (needed for SHL's evaluator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # all origins allowed
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, OPTIONS, etc.
    allow_headers=["*"],          # all headers allowed
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    GET /health — Server readiness check.
    
    SHL's evaluator calls this before starting evaluation.
    It also uses this to wake up the server (Render free tier sleeps).
    
    Must return: {"status": "ok"} with HTTP 200.
    That's it. Simple. But critical.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat — The main conversational agent endpoint.
    
    What happens here (step by step):
    
    1. RECEIVE: FastAPI receives the request body
       Pydantic automatically validates it matches ChatRequest schema.
       If invalid → 422 error returned automatically (we don't write that code)
    
    2. VALIDATE: Check conversation hasn't exceeded turn limit
    
    3. SEARCH: Convert conversation to a search query,
       search FAISS for relevant catalog items (RAG retrieval step)
    
    4. GENERATE: Send conversation + retrieved items to Gemini,
       get back a JSON response
    
    5. RETURN: Return the response as ChatResponse
       Pydantic validates the response shape too.
    
    The whole thing must complete in under 30 seconds (SHL requirement).
    """
    
    # Step 1: Get the messages from the request
    # request.messages is a list of Message objects (validated by Pydantic)
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    
    # Step 2: Check turn limit
    # SHL caps conversations at 8 turns (user + assistant combined)
    if len(messages) > MAX_TURNS:
        # We've hit the limit — return a graceful closing message
        return ChatResponse(
            reply=(
                "We've reached the maximum conversation length. "
                "Based on our discussion, here is my final recommendation. "
                "Please start a new conversation for additional needs."
            ),
            recommendations=[],
            end_of_conversation=True,
        )
    
    # Step 3: Build search query and retrieve relevant catalog items
    # This is the RETRIEVAL step of RAG
    search_query = build_search_query(messages)
    
    if search_query.strip():
        # Search FAISS for the most relevant assessments
        retrieved_items = search(search_query)
    else:
        # No meaningful query yet (e.g., first message is just "hello")
        retrieved_items = []
    
    # Step 4: Generate response using Gemini (the GENERATION step of RAG)
    response_dict = generate_response(
        messages=messages,
        retrieved_items=retrieved_items,
    )
    
    # Step 5: Convert recommendations dicts to Recommendation objects
    # (Pydantic needs proper objects, not plain dicts)
    recommendations = [
        Recommendation(
            name=rec["name"],
            url=rec["url"],
            test_type=rec["test_type"],
        )
        for rec in response_dict.get("recommendations", [])
    ]
    
    # Step 6: Return the response
    # FastAPI will serialize this to JSON automatically
    return ChatResponse(
        reply=response_dict.get("reply", ""),
        recommendations=recommendations,
        end_of_conversation=response_dict.get("end_of_conversation", False),
    )


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    This allows running directly with: python main.py
    But the recommended way is: uvicorn main:app --reload
    
    --reload = restart server automatically when you save a file (dev only)
    """
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
