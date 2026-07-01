"""
convert_catalog.py
Ingests, cleans, and standardizes raw catalog data and builds the TF-IDF index.
"""

import json
import os
import sys
import re

INPUT_PATH = r"c:\Users\riya bhatt\Desktop\SHL\shl_product_catalog.json"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")

TYPE_MAP = {
    "Ability & Aptitude": "A",
    "Ability": "A",
    "Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Biodata": "B",
    "Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Development": "D",
    "360": "D",
    "Knowledge & Skills": "K",
    "Knowledge": "K",
    "Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Personality": "P",
    "Behavior": "P",
    "Behaviour": "P",
    "Simulations": "S",
    "Simulation": "S"
}


def map_test_type(keys_list, name="", desc="") -> str:
    """Maps array of descriptive keys or text tokens to matching test type codes."""
    found_codes = []
    
    if isinstance(keys_list, list):
        for k in keys_list:
            for label, code in TYPE_MAP.items():
                if label.lower() in str(k).lower() and code not in found_codes:
                    found_codes.append(code)
    elif isinstance(keys_list, str):
        for label, code in TYPE_MAP.items():
            if label.lower() in keys_list.lower() and code not in found_codes:
                found_codes.append(code)
                
    if not found_codes:
        combined = f"{name} {desc}"
        for label, code in TYPE_MAP.items():
            if label.lower() in combined.lower() and code not in found_codes:
                found_codes.append(code)
                
    return ",".join(found_codes) if found_codes else "K"


def main():
    print("=" * 60)
    print("SHL Catalog Converter & Index Builder")
    print("=" * 60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found at {INPUT_PATH}")
        sys.exit(1)
        
    print(f"Reading source catalog from: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    try:
        raw_data = json.loads(content, strict=False)
    except Exception as e:
        print(f"Note: Standard JSON decode failed ({e}), cleaning control characters...")
        content_clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', content)
        raw_data = json.loads(content_clean, strict=False)
        
    print(f"Loaded {len(raw_data)} raw items.")
    
    clean_catalog = []
    seen_urls = set()
    
    for item in raw_data:
        name = item.get("name", "").strip()
        link = item.get("link", "").strip()
        desc = item.get("description", "").strip()
        keys = item.get("keys", [])
        
        if not name or not link:
            continue
            
        url = link if link.startswith("http") else f"https://www.shl.com{link}"
        url = url.rstrip("/") + "/"
        
        if url in seen_urls:
            continue
        seen_urls.add(url)
        
        test_type = map_test_type(keys, name, desc)
        
        clean_catalog.append({
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": desc
        })
        
    print(f"Processed {len(clean_catalog)} unique assessments.")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_catalog, f, ensure_ascii=False, indent=2)
    print(f"Saved cleaned catalog to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("Now building search index...")
    print("=" * 60)
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from retrieval.embedder import build_index
    build_index()
    
    print("\nSUCCESS! Catalog converted and search index built ready for API!")


if __name__ == "__main__":
    main()
