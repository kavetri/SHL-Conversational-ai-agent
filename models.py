"""
models.py — Pydantic data models (the "shapes" of our API request and response).

Why Pydantic models?
- They define EXACTLY what data must look like when it comes in and goes out.
- FastAPI uses them to automatically validate every request.
- If a request is missing a field or has the wrong type → FastAPI rejects it
  with a clear error BEFORE our code even runs.

Think of these as contracts:
  "I promise that every /chat request will look like ChatRequest."
  "I promise that every /chat response will look like ChatResponse."
"""

from pydantic import BaseModel      # BaseModel = base class for all Pydantic models
from typing import List, Optional   # List = typed list, Optional = field can be None


# ── Request Models ────────────────────────────────────────────────────────────

class Message(BaseModel):
    """
    Represents one message in the conversation.
    
    role: who sent this message — either "user" or "assistant"
    content: the actual text of the message
    
    Example:
        {"role": "user", "content": "I need a Java developer assessment"}
        {"role": "assistant", "content": "Sure! What seniority level?"}
    """
    role: str        # "user" or "assistant"
    content: str     # the message text


class ChatRequest(BaseModel):
    """
    The full body of a POST /chat request.
    
    messages: the COMPLETE conversation history so far.
    
    Why the full history every time?
    The API is STATELESS — our server stores nothing between calls.
    Every call must carry everything the agent needs to understand context.
    
    Example request body:
    {
        "messages": [
            {"role": "user", "content": "Hiring a Java developer"},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user", "content": "Mid-level, 4 years experience"}
        ]
    }
    """
    messages: List[Message]    # list of Message objects (defined above)


# ── Response Models ───────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    """
    One assessment recommendation.
    
    name: the exact name of the SHL assessment
    url: the real URL from the SHL catalog (NEVER hallucinated)
    test_type: the type code(s) — "K", "P", "A", "S", "B", "C", "D"
               Can be multiple like "K,S" for combined types
    
    Example:
        {
            "name": "Core Java (Advanced Level) (New)",
            "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
            "test_type": "K"
        }
    """
    name: str         # assessment name
    url: str          # real SHL catalog URL
    test_type: str    # type code(s)


class ChatResponse(BaseModel):
    """
    The full body of the POST /chat response.
    
    reply: the agent's natural language reply text
    
    recommendations: 
        - EMPTY LIST [] when agent is still clarifying / refusing
        - LIST OF 1-10 items when agent has committed to a shortlist
        - ⚠️ NEVER null — always an empty list or a populated list
    
    end_of_conversation:
        - false: conversation is still ongoing
        - true: agent considers the task complete (user confirmed shortlist)
    
    This schema is NON-NEGOTIABLE per SHL's assignment.
    Any deviation breaks the automated evaluator.
    """
    reply: str                              # agent's text reply
    recommendations: List[Recommendation]   # [] or 1-10 items
    end_of_conversation: bool               # true only when task is done


# ── Health Check Response ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """
    Response for GET /health endpoint.
    SHL evaluator pings this to check if our service is alive.
    Must return {"status": "ok"} with HTTP 200.
    """
    status: str    # always "ok"
