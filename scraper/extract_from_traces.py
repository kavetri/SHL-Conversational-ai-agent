"""
extract_from_traces.py - Fixed version
Correctly parses markdown tables from the SHL conversation traces.
"""
import json, os, re
from pathlib import Path

TRACES_DIR = r"c:\Users\riya bhatt\Desktop\SHL\GenAI_SampleConversations"
OUTPUT = "scraper/catalog.json"

def parse_trace(filepath):
    text = Path(filepath).read_text(encoding="utf-8")
    assessments = []
    seen_urls = set()

    # Each line that has a product URL also has the name in the same row
    # Format: | # | Name | Type | Keys | Duration | Languages | URL |
    # OR sometimes the URL is inline as <url> or [text](url)
    
    for line in text.split("\n"):
        if "|" not in line:
            continue

        # Find a SHL product catalog URL in this line
        url_match = re.search(
            r'https://www\.shl\.com/products/product-catalog/view/([\w-]+)/?',
            line
        )
        if not url_match:
            continue

        slug = url_match.group(1)
        full_url = f"https://www.shl.com/products/product-catalog/view/{slug}/"

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # Split row into cells and clean them
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c and c != "-" * len(c)]

        # Name is usually the 2nd cell (after the # index)
        # Try to find the real name - it shouldn't be just a number
        name = ""
        for cell in cells:
            # Skip pure numbers, dashes, URLs, type codes
            if re.match(r'^\d+$', cell): continue
            if re.match(r'^[-|]+$', cell): continue
            if cell.startswith("http"): continue
            if re.match(r'^[A-Z,]+$', cell) and len(cell) <= 10: continue
            if re.match(r'^\d+ minutes?$', cell, re.I): continue
            if cell.startswith("<http"): continue
            if "minutes" in cell.lower(): continue
            if len(cell) < 3: continue
            name = cell
            break

        if not name:
            # Try to derive name from URL slug
            name = slug.replace("-", " ").replace("new", "(New)").title()

        # Detect test type from the whole line
        type_code = "K"
        type_patterns = [
            (r'\|\s*((?:[A-Z],?)+)\s*\|.*Knowledge', "K"),
            (r'Personality', "P"),
            (r'Ability', "A"),
            (r'Simulation', "S"),
            (r'Biodata', "B"),
            (r'Competenc', "C"),
            (r'Development', "D"),
        ]
        
        # Look for explicit type cell like | K | or | A,S | or | P |
        type_cell = re.search(r'\|\s*([A-Z](?:,[A-Z])*)\s*\|', line)
        if type_cell:
            candidate = type_cell.group(1)
            # Make sure it's a real type code (not part of a name)
            valid_codes = set("AKPSBCDE")
            if all(c in valid_codes or c == "," for c in candidate):
                type_code = candidate

        # Override based on Keywords content
        if "Personality" in line and "Behavior" in line:
            if "," not in type_code:
                type_code = "P"
        if "Knowledge & Skills" in line:
            if "Simulation" in line:
                type_code = "K,S"
            elif "," not in type_code:
                type_code = "K"
        if "Ability & Aptitude" in line:
            type_code = "A" if "Simulation" not in line else "A,S"
        if "Biodata" in line and "Situational" in line:
            type_code = "B"
        if "Competenc" in line and "Knowledge" in line:
            type_code = "C,K"
        if "Development" in line and "360" in line:
            type_code = "D"

        assessments.append({
            "name": name,
            "url": full_url,
            "test_type": type_code,
            "description": "",
        })

    return assessments

# Run
all_items = []
seen_urls = set()

traces = sorted(Path(TRACES_DIR).glob("*.md"))
print(f"Processing {len(traces)} trace files...\n")

for trace in traces:
    items = parse_trace(trace)
    new_count = 0
    for item in items:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            all_items.append(item)
            new_count += 1
    print(f"  {trace.name}: {new_count} new assessments")

print(f"\nTotal unique assessments from traces: {len(all_items)}")
print("\nFull list:")
for item in all_items:
    print(f"  [{item['test_type']:5}] {item['name']}")
    print(f"          {item['url']}")

# Save
os.makedirs("scraper", exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(all_items, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUTPUT}")
