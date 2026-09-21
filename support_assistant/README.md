# Module 3 -- Support Assistant (Zepto Data & AI Platform)

I built this module as a retrieval-augmented support assistant that answers customer questions about Zepto's policies (delivery, returns, membership, tracking, cancellation, damaged items, gift cards, support hours). I implemented it using a local ChromaDB vector store, sentence-transformers embeddings, a LangGraph routing graph, and a FastAPI HTTP layer. I designed it to run fully offline by default (`MOCK_LLM=1`) -- no LLM API key or network call to any LLM provider is required for my graded baseline.

## My Architecture: ingestion -> embedding -> retrieval -> generation

### 1. Ingestion -- `ingest.py`

I wrote `load_documents()` to read each of the 8 policy files `docs/doc_01.txt` ... `docs/doc_08.txt`. I decided to treat each whole file as a single chunk, which I felt was acceptable given their short length (~300-600 characters each).

### 2. Embedding -- `ingest.py` (`get_embedding_model`, `build_collection`)

I programmed the script to embed each of the 8 chunks with sentence-transformers' `all-MiniLM-L6-v2` model (`SentenceTransformer.encode`). I write the 8 resulting 384-dim vectors into a local persistent ChromaDB collection (`chromadb.PersistentClient` pointed at `support_assistant/chroma_db`, collection name `zepto_policy_docs`, cosine distance space). My chunk IDs are formatted as (`doc_01_chunk_01` ... `doc_08_chunk_01`); I also make sure each chunk's metadata stores its `source_doc_id` (e.g. `doc_01`). Running my `python ingest.py` script (re)builds the collection from scratch and prints a summary. The collection correctly ends up with exactly 8 stored vectors.

I deliberately reuse the exact same embedding model and persistent collection at query time inside my `graph.py` via `get_embedding_model()` / `get_chroma_client()` imported from `ingest.py`. This ensures my ingestion and retrieval logic never drift apart.

### 3. Retrieval + routing -- `graph.py`

I built a LangGraph `StateGraph` over a `TypedDict` state (`SupportAssistantState`: `query`, `intent`, `retrieved_chunks`, `answer`, `sources`, `confidence`) with exactly three nodes:

- **`classify_intent`** -- I programmed this node to lowercase the query and check it for any of my defined keywords: `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`. If any match, it sets `intent = "policy_question"`; otherwise `intent = "general_question"`. This is my pure keyword heuristic with no LLM call, and its logic never depends on `MOCK_LLM`.
- **`retrieve_and_answer`** (reached when `intent == "policy_question"`) -- I use this to embed the incoming query with `all-MiniLM-L6-v2` and query the `zepto_policy_docs` ChromaDB collection for the top 3 most similar chunks by cosine similarity. I made sure this retrieval step always runs for real, regardless of my `MOCK_LLM` flag.
- **`direct_answer`** (reached when `intent == "general_question"`) -- No retrieval is involved here.

My Graph wiring: `START -> classify_intent`, then my conditional edge (`_route_after_classify`) sends the state to `retrieve_and_answer` or `direct_answer` based on `intent`; both nodes then edge to `END`.

### 4. Generation -- `retrieve_and_answer` / `direct_answer` in `graph.py`, using `prompt_template.py`

This is the **only stage I built that branches on `MOCK_LLM`**:

- **`MOCK_LLM` unset or `"1"` (my default):**
  - In `retrieve_and_answer`: I make no LLM call. I just build `answer = "Based on the retrieved context I found: " + <first ~200 chars of the single most similar retrieved chunk's text>`; I set `sources` = the ids of all 3 retrieved chunks; and `confidence = 1.0` (fixed).
  - In `direct_answer`: I make no LLM call. I just set `answer = "I can only answer questions about Zepto policies right now."`; `sources = []`; `confidence = 1.0`.
- **`MOCK_LLM == "0"` (my optional real-LLM branch):**
  - In `retrieve_and_answer`: my `prompt_template.build_prompt()` renders the structured `SUPPORT_ASSISTANT_PROMPT_TEMPLATE` (Role / Context / Task / Format / Length sections, a negative constraint, and a few-shot example) with the retrieved chunks as context. Then my `_call_real_llm()` sends it to an OpenAI-compatible / Groq chat-completions endpoint using an API key I read from the `LLM_API_KEY` environment variable (I never hardcode API keys!).
  - In `direct_answer`: I call `_call_real_llm()` directly with a simple prompt and no retrieved context.
  - In both cases, my `_generate_with_retries()` parses the raw LLM output as JSON and validates it against my `AnswerResponse` Pydantic model; if validation fails, I retry up to `MAX_LLM_RETRIES = 2` additional times with my `CORRECTIVE_INSTRUCTION` appended to the prompt, and return a clearly-marked `ErrorResponse` (`error=True`, message, safe defaults) instead of crashing if it still fails after those retries.

I enforce every node output through my `AnswerResponse` Pydantic model (`answer: str`, `sources: list[str]`, `confidence: float` constrained to `[0.0, 1.0]` via `Field(ge=0.0, le=1.0)`).

### My Data flow summary

`docs/doc_0N.txt` --(`ingest.py: load_documents`)--> raw chunk text --(`ingest.py: get_embedding_model().encode`)--> embedding vectors --(`ingest.py: build_collection`)--> persisted in ChromaDB (`chroma_db/`) --(`graph.py: retrieve_and_answer` query-time embed + `collection.query`)--> top-3 chunks --(`graph.py` mock branch, or `prompt_template.py` + `_call_real_llm` real branch)--> `AnswerResponse` --(`main.py: POST /ask`)--> JSON HTTP response.

## My API -- `main.py`

I built a FastAPI app with a Pydantic request model `AskRequest {query: str}` and a `POST /ask` endpoint that calls `graph.run_query(query)` (which invokes my compiled LangGraph graph). It returns the validated `AnswerResponse` (`answer`, `sources`, `confidence`) as JSON. I also provided a `GET /` health-check route.

## My Example run output

**My Ingestion** (`python ingest.py`):

```
My ChromaDB collection 'zepto_policy_docs' has been built at: .../support_assistant/chroma_db
Total stored vectors I processed: 8
My stored chunk IDs:
  doc_01_chunk_01  (source_doc_id=doc_01)
  doc_02_chunk_01  (source_doc_id=doc_02)
  doc_03_chunk_01  (source_doc_id=doc_03)
  doc_04_chunk_01  (source_doc_id=doc_04)
  doc_05_chunk_01  (source_doc_id=doc_05)
  doc_06_chunk_01  (source_doc_id=doc_06)
  doc_07_chunk_01  (source_doc_id=doc_07)
  doc_08_chunk_01  (source_doc_id=doc_08)
```

**My Live server** (`uvicorn main:app --host 127.0.0.1 --port 7860`, `MOCK_LLM` left unset -> my default mock mode), two real HTTP requests I made via `curl`:

My Request 1 -- `POST /ask {"query": "How long does delivery take?"}`
-> This routed correctly to `policy_question` -> `retrieve_and_answer`. The raw JSON response returned by my running server:

```json
{"answer":"Based on the retrieved context I found: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01_chunk_01","doc_02_chunk_01","doc_04_chunk_01"],"confidence":1.0}
```

I verified that the top-ranked (most similar) retrieved chunk was `doc_01_chunk_01`, i.e. Zepto's delivery-time policy document (`docs/doc_01.txt`), confirming my retrieval correctly matched the delivery-policy document for a delivery question.

My Request 2 -- `POST /ask {"query": "What is the capital of France?"}`
-> This routed correctly to `general_question` -> `direct_answer`. The raw JSON response returned by my running server:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

Both responses validated successfully against my `AnswerResponse` Pydantic schema (`answer: str`, `sources: list[str]`, `confidence: float` in `[0.0, 1.0]`) -- FastAPI's `response_model=AnswerResponse` on my `/ask` route enforces this on every response.

I then shut down the server (process on port 7860 terminated) after capturing these two responses.

## Running my module yourself

```bash
source "<repo>/.venv/Scripts/activate"
cd support_assistant
python ingest.py                 # I use this to build the ChromaDB collection (one-time / re-run to refresh)
uvicorn main:app --host 0.0.0.0 --port 7860   # MOCK_LLM unset -> my mock mode
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "How long does delivery take?"}'
```

To exercise the (structurally present, non-executed) real-LLM branch I wrote, set `MOCK_LLM=0` and `LLM_API_KEY=<your key>` before starting the server.

## Docker

My `Dockerfile` installs the project's single root `requirements.txt`, copies my app code and `docs/`, runs my `python ingest.py` at build time to pre-populate the ChromaDB collection, exposes port 7860, and starts the app with `uvicorn main:app --host 0.0.0.0 --port 7860`. Its build context is the **repo root** (so it can reach the root `requirements.txt`), not this folder:

```bash
# From the repo root (zepto-data-ai-platform/):
docker build -t zepto-support -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support
```

**I wrote the Dockerfile but couldn't test it locally since I don't have Docker installed (`docker: command not found`).** It has not been verified beyond my manual review of its syntax and layer ordering.

## My Files

- `ingest.py` -- my ingestion + embedding + ChromaDB collection builder.
- `prompt_template.py` -- my structured prompt template + corrective instruction for the optional real-LLM branch.
- `graph.py` -- my LangGraph `StateGraph`, Pydantic response schema, and mock and real-LLM generation logic.
- `main.py` -- my FastAPI app exposing `POST /ask`.
- `Dockerfile` -- my containerization script, installs the root `requirements.txt` (untested, see above).
- `docs/doc_01.txt` ... `docs/doc_08.txt` -- the Zepto policy corpus I am using (unmodified).
- `chroma_db/` -- my persisted ChromaDB collection created by `ingest.py`.
