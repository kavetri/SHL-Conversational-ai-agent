"""
client.py
Gemini API client interface with defensive output parsing and validation.
"""

import json
import re
import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE
from llm.prompts import SYSTEM_PROMPT, build_user_prompt

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured in the environment.")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config=genai.GenerationConfig(
        temperature=GEMINI_TEMPERATURE,
        response_mime_type="application/json",
    ),
    system_instruction=SYSTEM_PROMPT,
)


def extract_json(text: str) -> dict:
    """
    Extracts and parses JSON content from the raw LLM response text.
    Handles potential markdown code wrapping block configurations defensively.
    """
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
            
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
            
    return {
        "reply": "I'm having trouble processing that request. Could you rephrase?",
        "recommendations": [],
        "end_of_conversation": False,
    }


def generate_response(messages: list, retrieved_items: list) -> dict:
    """
    Constructs the prompt context and queries the Gemini LLM endpoint.
    """
    user_prompt = build_user_prompt(
        conversation_history=messages,
        retrieved_items=retrieved_items,
        turn_count=len(messages),
    )
    
    try:
        response = model.generate_content(user_prompt)
        parsed = extract_json(response.text)
        return validate_response(parsed, retrieved_items)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {
            "reply": "I encountered an issue. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        }


def validate_response(parsed: dict, retrieved_items: list) -> dict:
    """
    Validates and standardizes the parsed JSON model schema response.
    Performs real-time link validation to filter out hallucinated URLs.
    """
    valid_urls = {item.get("url", "") for item in retrieved_items}
    
    reply = parsed.get("reply", "")
    if not isinstance(reply, str) or not reply.strip():
        reply = "I'm here to help you select SHL assessments. What role are you hiring for?"
        
    raw_recs = parsed.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []
        
    clean_recs = []
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
            
        name = rec.get("name", "").strip()
        url = rec.get("url", "").strip()
        test_type = rec.get("test_type", "K").strip()
        
        if not name or not url:
            continue
            
        # Anti-hallucination validation against the retrieved database subset
        if url not in valid_urls:
            matched = next(
                (item for item in retrieved_items if item.get("name", "").lower() == name.lower()),
                None
            )
            if matched:
                url = matched.get("url", url)
            else:
                continue
                
        clean_recs.append({
            "name": name,
            "url": url,
            "test_type": test_type,
        })
        
    if len(clean_recs) > 10:
        clean_recs = clean_recs[:10]
        
    end_of_conv = parsed.get("end_of_conversation", False)
    if not isinstance(end_of_conv, bool):
        end_of_conv = str(end_of_conv).lower() == "true"
        
    return {
        "reply": reply,
        "recommendations": clean_recs,
        "end_of_conversation": end_of_conv,
    }
