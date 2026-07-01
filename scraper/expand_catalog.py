"""
expand_catalog.py
Expands the seed catalog by:
1. Visiting each known assessment's page to confirm data
2. Finding additional assessments via SHL's catalog API/search
3. Scraping catalog pages by directly injecting cookie acceptance
"""
import json, os, re, time, requests
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

SEED_FILE = "scraper/catalog.json"
OUTPUT_FILE = "scraper/catalog.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TYPE_MAP = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P", "Personality & Behaviour": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

def get_type_from_text(text):
    codes = []
    for label, code in TYPE_MAP.items():
        if label.lower() in text.lower() and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"

def scrape_detail_page(url, session):
    """Visit an individual product page and extract description + correct type."""
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None, None
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract description from product page
        desc = ""
        for sel in [
            ".product-description", ".product__description",
            ".hero__description", "[class*='description']",
            "main p", ".content p",
        ]:
            elem = soup.select_one(sel)
            if elem:
                desc = elem.get_text(strip=True)[:500]
                break

        # Extract type from page
        page_text = soup.get_text()
        test_type = get_type_from_text(page_text)

        return desc, test_type
    except Exception as e:
        return None, None

def scrape_catalog_with_playwright():
    """Use Playwright to get catalog items, with proper cookie + wait handling."""
    items = []
    seen_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )

        # Pre-set cookies to bypass consent (common approach)
        context.add_cookies([
            {
                "name": "CookieConsent",
                "value": "true",
                "domain": ".shl.com",
                "path": "/",
            },
            {
                "name": "cookie_consent",
                "value": "1",
                "domain": ".shl.com",
                "path": "/",
            },
        ])

        page = context.new_page()

        # Try multiple catalog URL variations
        catalog_urls = [
            "https://www.shl.com/solutions/products/product-catalog/?type=1",
            "https://www.shl.com/products/product-catalog/?type=1",
            "https://www.shl.com/solutions/products/product-catalog/",
        ]

        for url in catalog_urls:
            print(f"  Trying: {url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
            except:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)
                except:
                    continue

            # Extra wait
            time.sleep(5)

            # Dismiss any remaining cookie banner
            for sel in ["button:has-text('OK')", "button:has-text('Accept')",
                        "button:has-text('Accept All')", "#onetrust-accept-btn-handler"]:
                try:
                    page.click(sel, timeout=2000)
                    time.sleep(2)
                    break
                except: pass

            # Scroll and wait for content
            for _ in range(3):
                page.evaluate("window.scrollBy(0, 500)")
                time.sleep(1)

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Try to find product links
            links = [a for a in soup.find_all("a", href=True)
                    if "/product-catalog/view/" in a.get("href", "")]

            print(f"  Found {len(links)} product links")

            if links:
                for a in links:
                    href = a["href"]
                    full_url = "https://www.shl.com" + href if href.startswith("/") else href
                    full_url = full_url.rstrip("/") + "/"
                    if full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)
                    name = a.get_text(strip=True)
                    if name and len(name) > 3:
                        items.append({
                            "name": name,
                            "url": full_url,
                            "test_type": "K",
                            "description": "",
                        })
                break

            # Save HTML for inspection
            with open(f"scraper/debug_{url.split('/')[-1] or 'root'}.html", "w", encoding="utf-8") as f:
                f.write(html)

        browser.close()

    return items

def main():
    # Load seed catalog
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    existing_urls = {item["url"] for item in catalog}
    print(f"Seed catalog: {len(catalog)} items")

    # Step 1: Try to get more items from catalog pages
    print("\nStep 1: Scraping catalog pages with Playwright...")
    new_items = scrape_catalog_with_playwright()
    print(f"Got {len(new_items)} items from catalog pages")

    added = 0
    for item in new_items:
        if item["url"] not in existing_urls:
            existing_urls.add(item["url"])
            catalog.append(item)
            added += 1
    print(f"Added {added} new items. Total: {len(catalog)}")

    # Step 2: Enrich existing items with descriptions from their detail pages
    print("\nStep 2: Enriching items with descriptions...")
    session = requests.Session()
    session.headers.update(HEADERS)

    enriched = 0
    for i, item in enumerate(catalog):
        if item.get("description"):
            continue  # already has description
        
        print(f"  [{i+1}/{len(catalog)}] {item['name'][:50]}...")
        desc, test_type = scrape_detail_page(item["url"], session)
        
        if desc:
            item["description"] = desc
            enriched += 1
        if test_type and test_type != "K":
            item["test_type"] = test_type
        
        time.sleep(0.5)  # polite delay

    print(f"Enriched {enriched} items with descriptions")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"\nFinal catalog: {len(catalog)} items")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
