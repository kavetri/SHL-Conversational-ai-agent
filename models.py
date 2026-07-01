"""
models.py
Pydantic schemas defining the API request and response models.
"""

from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    """
    Represents an individual message in the conversation history.
    """
    role: str      # "user" or "assistant"
    content: str   # The message content


class ChatRequest(BaseModel):
    """
    Input schema for the POST /chat endpoint.
    """
    messages: List[Message]


class Recommendation(BaseModel):
    """
    Represents a recommended assessment from the catalog.
    """
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """
    Response schema for the POST /chat endpoint.
    """
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    """
    Response schema for the GET /health endpoint.
    """
    status: str
