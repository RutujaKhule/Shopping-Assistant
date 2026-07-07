# 🛍️ Multimodal AI Shopping Assistant (RAG)

Upload a photo of any product and get real-time prices across Indian
retailers, an AI buying recommendation, a customer review summary, similar
products, and a follow-up chatbot — all grounded in live web search. No
static datasets, no hardcoded product data.

LIVE DEMO - https://shopping-assistant-6wdreks83r45dcktpxdswb.streamlit.app/
## How it works

```
Product Image
     │
     ▼
Gemini Vision  ──►  Product identified (name, brand, category, model, color)
     │
     ▼
Tavily Search  ──►  Specs · Prices (Amazon/Flipkart/Croma/Reliance Digital/
     │                Vijay Sales) · Reviews · Similar products — all live
     ▼
Chunk + Embed  ──►  Gemini text-embedding-004
     │
     ▼
FAISS (in-memory, per-session)
     │
     ▼
LangChain Retrieval  ──►  Gemini Flash  ──►  Grounded answers,
                                              recommendations, summaries
```

## Project structure

```
shopping_ai/
├── app.py                  # Streamlit entrypoint — orchestrates everything
├── requirements.txt
├── .env                    # API keys (not committed)
├── services/
│   ├── gemini.py           # Gemini Vision + Flash wrapper
│   ├── search.py           # Tavily real-time search
│   ├── embeddings.py       # Chunking + Gemini embeddings
│   ├── comparison.py       # Price table, best-deal highlighting, buy recommendation
│   └── reviews.py          # AI review summarization
├── rag/
│   ├── vector_store.py     # Temporary in-memory FAISS index
│   └── retrieval.py        # RAG pipeline + chat memory
└── ui/
    └── components.py       # Streamlit UI components
```

## Setup

1. **Clone / open the project folder**, then create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API keys** to `.env`:
   ```
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
   - Gemini key: https://aistudio.google.com/app/apikey
   - Tavily key: https://app.tavily.com/

4. **Run locally:**
   ```bash
   streamlit run app.py
   ```
   The app opens at `http://localhost:8501`.

## Using the app

- Upload a product photo in the sidebar.
- Wait for identification, pricing, and review summarization to complete.
- Ask questions in the chat box, or use the quick-action buttons:
  - "Should I buy this?"
  - "Summarize customer reviews"
  - "What are the pros and cons?"
  - "Show cheaper alternatives"
  - "Which website offers the best deal?"
- For product comparisons, just ask naturally: *"Compare this with the
  Samsung Galaxy S24"* — the app detects the named product, researches it
  live, and returns a side-by-side comparison.
- For budget alternatives, ask: *"Show laptops under 60000"* — the app
  parses the price cap and category, searches live, and answers from that
  fresh data.
- Click **"🔄 Start Over"** in the sidebar to clear everything and upload a
  new product.

## Notes on design choices

- **No static data anywhere.** Every spec, price, and review comes from a
  live Tavily search triggered by the identified product.
- **FAISS is in-memory and per-session only** — it's rebuilt every time a
  new image is uploaded, never persisted to disk.
- **Gemini does double duty**: besides vision + answer generation, it's
  also used to parse retailers' unstructured search snippets into clean
  structured pricing data, since there's no product database to query.

## Deployment

### Streamlit Community Cloud (simplest)
1. Push this project to a public or private GitHub repo (make sure `.env`
   is in `.gitignore` and NOT committed).
2. Go to https://share.streamlit.io → "New app" → point it at your repo,
   branch, and `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_key"
   TAVILY_API_KEY = "your_key"
   ```
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

### Render
1. Push to GitHub.
2. On https://render.com, create a **New Web Service** from your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add `GOOGLE_API_KEY` and `TAVILY_API_KEY` under **Environment**.
6. Deploy.

### Railway
1. Push to GitHub, then "New Project" → "Deploy from GitHub repo" on
   https://railway.app.
2. Railway auto-detects Python; set the start command to:
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Add `GOOGLE_API_KEY` and `TAVILY_API_KEY` in the **Variables** tab.
4. Deploy — Railway assigns a public URL automatically.

### AWS Lambda (backend API only)
Streamlit itself isn't a great fit for Lambda (it's a long-running server,
not a request/response function). If you want an AWS Lambda deployment,
the practical path is to split the backend logic out:
1. Wrap `services/` and `rag/` in a small FastAPI app (e.g. one `/ask`
   endpoint that runs the pipeline in `app.py` minus the Streamlit calls).
2. Package with **Mangum** (`pip install mangum`) to adapt FastAPI to
   Lambda's handler signature.
3. Deploy via AWS SAM, the Serverless Framework, or the AWS CLI
   (`aws lambda create-function`), with `GOOGLE_API_KEY` and
   `TAVILY_API_KEY` set as Lambda environment variables.
4. Put the existing Streamlit `app.py` behind a separate lightweight host
   (e.g. Streamlit Cloud or an EC2/container instance) and have it call
   your Lambda API instead of the services directly, if you want the
   frontend and backend fully decoupled.
5. Note: FAISS's in-memory index won't persist between cold Lambda
   invocations — each request would need to rebuild its index from a
   fresh search, which is consistent with this project's "no static data,
   always live" design anyway.

## Troubleshooting

- **"GOOGLE_API_KEY is missing or unset"** — check `.env` is in the project
  root and the key doesn't still say `your_google_gemini_api_key_here`.
- **Empty price table** — Tavily may not have found a retailer listing;
  this is expected for very new, obscure, or regional-only products.
- **Slow responses** — real-time web search + multiple Gemini calls take a
  few seconds per product load; this is inherent to a fully live pipeline
  with no cached/static data.
