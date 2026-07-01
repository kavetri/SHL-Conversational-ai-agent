# SHL Assessment Recommender — Conversational AI & RAG Engine

A production-grade, stateless conversational agent built for **SHL Labs** to assist hiring managers and recruiters in selecting tailored psychometric and technical assessments from the SHL Product Catalog.

---

## 🌟 Key Features & Architectural Highlights

1. **Stateless Conversational Architecture (`POST /chat`)**:
   - Complies strictly with zero-persistence requirements. The full message trajectory is ingested and evaluated on every turn.
   - Enforces an automated **8-turn conversation limit**, gracefully concluding dialogues before token exhaustion or user fatigue.

2. **Advanced RAG (Retrieval-Augmented Generation)**:
   - **Vector Search Engine**: Powered by **FAISS** (`IndexFlatIP`) and HuggingFace's `all-MiniLM-L6-v2` embeddings (384-dimensional semantic space).
   - **Semantic Clustering**: Encodes assessment names, domain indicators (`K`, `P`, `A`, `S`, `B`, `C`, `D`, `E`), and rich product descriptions to match nuanced queries (e.g., matching *"Java backend developer"* with *"Core Java (Advanced Level)"*).
   - **Zero-Hallucination Guarantee**: Strict post-processing validation layer validates every LLM-suggested URL against the real scraped catalog. Hallucinated links are stripped or auto-corrected.

3. **Four Distinct Operational Modes**:
   - **Mode 1: Clarification**: Detects vague requests (*"I need to hire someone"*) and proactively asks ONE targeted clarifying question while returning an empty recommendation list `[]`.
   - **Mode 2: Recommendation**: Delivers 1–10 schema-compliant assessment recommendations with exact test type codes and verified official URLs.
   - **Mode 3: Refinement**: Handles iterative constraints (*"Drop personality tests and add SQL"*) seamlessly without losing dialogue context.
   - **Mode 4: Out-of-Scope & Legal Refusal**: Gracefully refuses legal compliance advice (*"Is this legal in California?"*) and prompt injections, redirecting users back to SHL product selection.

---

## 🛠️ Technology Stack

- **Framework**: FastAPI, Pydantic v2, Uvicorn
- **LLM Engine**: Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-flash`) via `google-generativeai` with JSON schema enforcement.
- **Vector Database**: FAISS (Facebook AI Similarity Search) CPU
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Scraping & Data Processing**: Playwright, BeautifulSoup4, Custom Cleaners

---

## 📂 Project Structure

```text
shl-recommender/
├── main.py                # FastAPI Application entry point (/health & /chat)
├── models.py              # Pydantic Request & Response contracts
├── config.py              # Centralized environment & runtime settings
├── requirements.txt       # Pinned production dependencies
├── render.yaml            # Render Blueprint for automated cloud deployment
├── test_api.py            # Local automated verification suite (simulates SHL harness)
├── llm/
│   ├── client.py          # Gemini API wrapper with anti-hallucination validation
│   └── prompts.py         # System prompt instruction set & query builder
├── retrieval/
│   └── embedder.py        # FAISS index loader, embedding generator, and search engine
└── scraper/
    ├── convert_catalog.py # Ingests source catalog JSON and builds FAISS index
    └── catalog.json       # Processed clean catalog (377 verified assessments)
```

---

## 🚀 Local Setup & Installation

### 1. Clone the repository & install dependencies
```powershell
git clone https://github.com/<your-username>/shl-recommender.git
cd shl-recommender
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Gemini API Key:
```powershell
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY=AIzaSy...
```

### 3. Build Vector Index (Run Once)
```powershell
python scraper/convert_catalog.py
```
*This verifies `catalog.json` and generates the local FAISS vector store under `retrieval/faiss_store/`.*

### 4. Run the API Server
```powershell
uvicorn main:app --reload --port 8000
```
- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Automated Testing Suite

To run the local evaluation harness (verifies health check, clarification behavior, recommendation accuracy, and legal refusal):
```powershell
python test_api.py
```

---

## ☁️ Cloud Deployment (Render)

1. Connect your GitHub repository to [Render.com](https://render.com).
2. Create a new **Web Service** using the Python runtime.
3. Set Build Command: `pip install -r requirements.txt && python scraper/convert_catalog.py`
4. Set Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variable: `GEMINI_API_KEY` = `your_actual_key_here`.
