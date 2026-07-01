"""
Final scraper - uses correct URL + handles cookie + waits properly
"""
from playwright.sync_api import sync_playwright
import json, time, os, re
from bs4 import BeautifulSoup

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")

# The correct catalog URL (from the conversation traces URLs pattern)
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

TYPE_MAP = {
    "Ability": "A", "Aptitude": "A",
    "Assessment Exercise": "E",
    "Biodata": "B", "Situational": "B",
    "Competenc": "C",
    "Development": "D", "360": "D",
    "Knowledge": "K", "Skills": "K",
    "Personality": "P", "Behavior": "P", "Behaviour": "P",
    "Simulation": "S",
}

def get_type(text):
    found = []
    for kw, code in TYPE_MAP.items():
        if kw.lower() in text.lower() and code not in found:
            found.append(code)
    return ",".join(found) if found else "K"

def extract_assessments(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen = set()

    # Find all anchor tags linking to product catalog view pages
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match both URL patterns seen in the traces
        if "/product-catalog/view/" in href or "/products/product-catalog/view/" in href:
            # Build full URL
            if href.startswith("http"):
                full_url = href
            else:
                full_url = "https://www.shl.com" + href

            # Normalize URL
            full_url = full_url.rstrip("/") + "/"

            if full_url in seen:
                continue
            seen.add(full_url)

            name = a.get_text(strip=True)
            if not name or len(name) < 3:
                continue

            # Get surrounding context for type detection
            parent = a.find_parent("tr") or a.find_parent("li") or a.find_parent("div")
            ctx = parent.get_text() if parent else a.get_text()
            test_type = get_type(ctx)

            results.append({
                "name": name,
                "url": full_url,
                "test_type": test_type,
                "description": "",
            })

    return results

def run():
    print("="*60)
    print("SHL Catalog Scraper v3")
    print("="*60)

    all_items = []
    seen_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Page 1 - accept cookies, get to catalog
        print(f"\nLoading catalog page...")
        page.goto(CATALOG_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(4)

        # Accept cookie banner
        for sel in [
            "button:has-text('OK')",
            "button:has-text('Accept All')",
            "button:has-text('Accept')",
            "#onetrust-accept-btn-handler",
        ]:
            try:
                page.click(sel, timeout=3000)
                print(f"Cookie dismissed: {sel}")
                time.sleep(3)
                break
            except: continue

        # Click "Individual Test Solutions" filter if visible
        for sel in [
            "text=Individual Test Solutions",
            "a:has-text('Individual Test')",
            "label:has-text('Individual')",
            "input[value='1']",
        ]:
            try:
                page.click(sel, timeout=4000)
                print(f"Clicked filter: {sel}")
                time.sleep(3)
                break
            except: continue

        # Try to set items per page to maximum
        for sel in ["select[name*='per'], select[name*='count'], select.per-page"]:
            try:
                page.select_option(sel, index=-1)  # last option = maximum
                time.sleep(2)
                break
            except: continue

        # Scrape all pages via pagination
        page_num = 1
        while page_num <= 50:  # safety limit
            print(f"\nPage {page_num}...")
            time.sleep(2)

            html = page.content()
            items = extract_assessments(html)

            new = 0
            for item in items:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    all_items.append(item)
                    new += 1

            print(f"  Found {len(items)} items, {new} new. Total: {len(all_items)}")

            # Debug: save first page HTML
            if page_num == 1:
                with open("scraper/debug_final.html", "w", encoding="utf-8") as f:
                    f.write(html)
                print("  Saved first page HTML to debug_final.html")

                # Print some links for debugging
                from bs4 import BeautifulSoup as BS
                s = BS(html, "html.parser")
                sample_links = [a["href"] for a in s.find_all("a", href=True)][:20]
                print(f"  Sample links: {sample_links}")

            if new == 0 and page_num > 1:
                print("  No new items — done.")
                break

            # Try to click "Next" button
            next_clicked = False
            for sel in [
                "a[rel='next']",
                "button:has-text('Next')",
                "a:has-text('Next')",
                ".pagination__next",
                "[aria-label='Next page']",
                "[aria-label='Next']",
            ]:
                try:
                    btn = page.query_selector(sel)
                    if btn and btn.is_visible():
                        btn.click()
                        next_clicked = True
                        print(f"  Clicked next: {sel}")
                        time.sleep(3)
                        break
                except: continue

            if not next_clicked:
                print("  No next button found — done.")
                break

            page_num += 1

        browser.close()

    print(f"\n{'='*60}")
    print(f"Total: {len(all_items)} assessments")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nSample:")
    for item in all_items[:5]:
        print(f"  [{item['test_type']}] {item['name']}")
        print(f"       {item['url']}")

if __name__ == "__main__":
    run()
