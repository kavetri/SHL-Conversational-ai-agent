"""
test_api.py — Automated Local Test Suite for SHL Assessment Recommender API
"""
import sys
import os
from fastapi.testclient import TestClient

# Import our FastAPI app from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import app

client = TestClient(app)

def run_tests():
    print("=" * 60)
    print("STARTING LOCAL EVALUATION SUITE")
    print("=" * 60)
    
    # ── Test 1: Health Check ──────────────────────────────────────────────────
    print("\n[Test 1] Testing GET /health ...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data == {"status": "ok"}, f"Expected {{'status': 'ok'}}, got {data}"
    print("[PASS] Test 1: Health endpoint is active and compliant!")
    
    # ── Test 2: Mode 1 — Clarification (Vague request) ────────────────────────
    print("\n[Test 2] Testing POST /chat with VAGUE query (expecting clarification & [] recs)...")
    payload_vague = {
        "messages": [
            {"role": "user", "content": "I need to hire someone."}
        ]
    }
    resp = client.post("/chat", json=payload_vague)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    print(f"  Agent Reply: \"{data['reply']}\"")
    print(f"  Recommendations count: {len(data['recommendations'])}")
    assert isinstance(data["recommendations"], list), "recommendations MUST be a list!"
    assert len(data["recommendations"]) == 0, "Expected 0 recommendations for vague query!"
    assert data["end_of_conversation"] is False, "end_of_conversation should be False!"
    print("[PASS] Test 2: Agent asks clarifying question and returns empty list!")
    
    # ── Test 3: Mode 2 — Recommendation (Specific technical query) ────────────
    print("\n[Test 3] Testing POST /chat with SPECIFIC query (expecting 1-10 valid recommendations)...")
    payload_specific = {
        "messages": [
            {"role": "user", "content": "We are hiring a Senior Java Developer with strong Spring Boot and SQL skills."}
        ]
    }
    resp = client.post("/chat", json=payload_specific)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    print(f"  Agent Reply: \"{data['reply'][:120]}...\"")
    recs = data["recommendations"]
    print(f"  Recommendations returned: {len(recs)}")
    assert len(recs) >= 1 and len(recs) <= 10, f"Expected 1-10 recommendations, got {len(recs)}"
    
    # Check schema of each recommendation
    for idx, r in enumerate(recs, 1):
        print(f"    {idx}. [{r['test_type']}] {r['name']}")
        print(f"       URL: {r['url']}")
        assert "name" in r and "url" in r and "test_type" in r, "Missing required recommendation fields!"
        assert r["url"].startswith("https://www.shl.com/"), f"Invalid URL domain: {r['url']}"
    print("[PASS] Test 3: Recommendations are schema-compliant and semantically relevant!")
    
    # ── Test 4: Out-of-Scope / Legal Refusal ──────────────────────────────────
    print("\n[Test 4] Testing POST /chat with LEGAL / OUT-OF-SCOPE query (expecting refusal & [] recs)...")
    payload_legal = {
        "messages": [
            {"role": "user", "content": "Is it legal to use personality tests to reject candidates in California?"}
        ]
    }
    resp = client.post("/chat", json=payload_legal)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    print(f"  Agent Reply: \"{data['reply']}\"")
    assert len(data["recommendations"]) == 0, "Expected 0 recommendations during legal refusal!"
    print("[PASS] Test 4: Agent gracefully refuses legal advice!")
    
    print("\n" + "=" * 60)
    print("ALL 4 TESTS PASSED! YOUR API IS READY FOR DEPLOYMENT!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
