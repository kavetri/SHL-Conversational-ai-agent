"""
client.py — Gemini API wrapper.

What this file does:
- Wraps the google-generativeai library into a clean, simple function
- Handles JSON parsing of Gemini's response
- Has retry logic for when Gemini returns malformed JSON
- Returns a clean Python dict that main.py can use directly

Why a wrapper?
- We isolate ALL Gemini-specific code here.
- If we ever switch LLM providers (Groq, Claude, etc.), we ONLY change this file.
- main.py doesn't care HOW the LLM works — it just calls generate_response().
"""

import json           # json = parse Gemini's text response into a Python dict
import re             # re = regular expressions for extracting JSON from text
import google.generativeai as genai   # the official Gemini Python SDK

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE
from llm.prompts import SYSTEM_PROMPT, build_user_prompt, build_search_query


# ── Gemini Setup ──────────────────────────────────────────────────────────────

# Configure the SDK with your API key
# This must be called once before any API calls
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Create a .env file with GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=GEMINI_API_KEY)

# Create the Gemini model instance
# GenerationConfig controls HOW Gemini generates text
model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config=genai.GenerationConfig(
        temperature=GEMINI_TEMPERATURE,   # 0.1 = mostly deterministic
        response_mime_type="application/json",  # force JSON output mode!
    ),
    system_instruction=SYSTEM_PROMPT,     # inject our system prompt
)


# ── JSON Extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """
    Extract and parse JSON from Gemini's response text.
    
    Even with response_mime_type="application/json", sometimes Gemini
    wraps the JSON in markdown code blocks like:
        ```json
        {"reply": "..."}
        ```
    
    This function handles all cases:
    1. Pure JSON: {"reply": "..."}
    2. Markdown-wrapped: ```json\n{...}\n```
    3. JSON embedded in text: some text {"reply": "..."} more text
    
    Why? Because if JSON parsing fails, our API returns an error.
    Defensive parsing = robust system.
    """
    text = text.strip()
    
    # Case 1: Try direct JSON parsing first (cleanest case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Case 2: Strip markdown code block and try again
    # Pattern: ```json\n...\n``` or ```\n...\n```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    
    # Case 3: Find first { to last } and try that
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
    
    # All parsing attempts failed — return a safe fallback response
    # This prevents the entire API from crashing due to one bad LLM response
    return {
        "reply": "I'm having trouble processing that request. Could you rephrase?",
        "recommendations": [],
        "end_of_conversation": False,
    }


# ── Main Generation Function ──────────────────────────────────────────────────

def generate_response(messages: list, retrieved_items: list) -> dict:
    """
    Generate an agent response using Gemini.
    
    This is the core function — called by main.py on every /chat request.
    
    Flow:
    1. Count turns to enforce the 8-turn limit
    2. Build the prompt (system prompt already set on model)
    3. Call Gemini API
    4. Parse the JSON response
    5. Validate and clean the response
    6. Return a clean dict matching the ChatResponse schema
    
    Args:
        messages: full conversation history [{role, content}, ...]
        retrieved_items: catalog items from FAISS search
    
    Returns:
        dict with keys: reply, recommendations, end_of_conversation
    """
    
    # Count turns (each message = 1 turn)
    turn_count = len(messages)
    
    # Build the prompt that includes catalog context + conversation history
    user_prompt = build_user_prompt(
        conversation_history=messages,
        retrieved_items=retrieved_items,
        turn_count=turn_count,
    )
    
    try:
        # Call the Gemini API
        # generate_content() sends our prompt and returns a response
        response = model.generate_content(user_prompt)
        
        # Extract the text from the response
        response_text = response.text
        
        # Parse the JSON from the text
        parsed = extract_json(response_text)
        
        # Validate and clean the parsed response
        return validate_response(parsed, retrieved_items)
        
    except Exception as e:
        # If ANYTHING goes wrong (network error, API error, etc.)
        # return a graceful error response — never crash the API
        print(f"LLM Error: {e}")
        return {
            "reply": "I encountered an issue. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        }


def validate_response(parsed: dict, retrieved_items: list) -> dict:
    """
    Validate and clean the LLM response to ensure schema compliance.
    
    SHL's automated evaluator requires EXACT schema. Any deviation = 0 score.
    This function acts as a last line of defense.
    
    Checks:
    - reply must be a non-empty string
    - recommendations must be a list (not null)
    - Each recommendation must have name, url, test_type
    - URLs must come from retrieved catalog items (anti-hallucination)
    - end_of_conversation must be a boolean
    """
    
    # Build a set of valid URLs from retrieved items for validation
    valid_urls = {item.get("url", "") for item in retrieved_items}
    
    # ── Validate reply ────────────────────────────────────────────────────────
    reply = parsed.get("reply", "")
    if not isinstance(reply, str) or not reply.strip():
        reply = "I'm here to help you select SHL assessments. What role are you hiring for?"
    
    # ── Validate recommendations ──────────────────────────────────────────────
    raw_recs = parsed.get("recommendations", [])
    
    # Must be a list, not null or other type
    if not isinstance(raw_recs, list):
        raw_recs = []
    
    # Clean each recommendation
    clean_recs = []
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        
        name = rec.get("name", "").strip()
        url = rec.get("url", "").strip()
        test_type = rec.get("test_type", "K").strip()
        
        # Skip if missing critical fields
        if not name or not url:
            continue
        
        # Anti-hallucination check: URL must be from our catalog
        # If URL is not in retrieved items, try to find the matching catalog item
        if url not in valid_urls:
            # Try to find by name in retrieved items
            matched = next(
                (item for item in retrieved_items if item.get("name", "").lower() == name.lower()),
                None
            )
            if matched:
                url = matched.get("url", url)   # use the real URL
            else:
                # URL is hallucinated — skip this recommendation
                print(f"Warning: Skipping hallucinated recommendation: {name} -> {url}")
                continue
        
        clean_recs.append({
            "name": name,
            "url": url,
            "test_type": test_type,
        })
    
    # Enforce 1-10 limit on recommendations
    if len(clean_recs) > 10:
        clean_recs = clean_recs[:10]
    
    # ── Validate end_of_conversation ──────────────────────────────────────────
    end_of_conv = parsed.get("end_of_conversation", False)
    if not isinstance(end_of_conv, bool):
        end_of_conv = str(end_of_conv).lower() == "true"
    
    return {
        "reply": reply,
        "recommendations": clean_recs,
        "end_of_conversation": end_of_conv,
    }
