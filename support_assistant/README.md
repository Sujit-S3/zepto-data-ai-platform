# Module 3 -- Support Assistant (Zepto Data & AI Platform)

A retrieval-augmented support assistant that answers customer questions
about Zepto's policies (delivery, returns, membership, tracking,
cancellation, damaged items, gift cards, support hours) using a local
ChromaDB vector store, sentence-transformers embeddings, a LangGraph
routing graph, and a FastAPI HTTP layer. Runs fully offline by default
(`MOCK_LLM=1`) -- no LLM API key or network call to any LLM provider is
required for the graded baseline.

## Architecture: ingestion -> embedding -> retrieval -> generation

### 1. Ingestion -- `ingest.py`

`load_documents()` reads each of the 8 policy files `docs/doc_01.txt` ...
`docs/doc_08.txt`. Each whole file is treated as a single chunk (acceptable
given their short length, ~300-600 characters each).

### 2. Embedding -- `ingest.py` (`get_embedding_model`, `build_collection`)

Each of the 8 chunks is embedded with sentence-transformers'
`all-MiniLM-L6-v2` model (`SentenceTransformer.encode`). The 8 resulting
384-dim vectors are written into a local persistent ChromaDB collection
(`chromadb.PersistentClient` pointed at `support_assistant/chroma_db`,
collection name `zepto_policy_docs`, cosine distance space). Chunk IDs are
(`doc_01_chunk_01` ... `doc_08_chunk_01`); each chunk's metadata stores
`source_doc_id` (e.g. `doc_01`). Running `python ingest.py` (re)builds the
collection from scratch and prints a summary. The collection ends
up with exactly 8 stored vectors.

The same embedding model and the same persistent collection are reused at
query time by `graph.py` via `get_embedding_model()` / `get_chroma_client()`
imported from `ingest.py`, so ingestion and retrieval never drift apart.

### 3. Retrieval + routing -- `graph.py`

A LangGraph `StateGraph` is built over a `TypedDict` state
(`SupportAssistantState`: `query`, `intent`, `retrieved_chunks`, `answer`,
`sources`, `confidence`) with exactly three nodes:

- **`classify_intent`** -- lowercases the query and checks it for any of the
  keywords `delivery`, `return`, `refund`, `membership`, `tracking`,
  `cancel`, `gift card`, `support hours`. If any match, `intent =
  "policy_question"`; otherwise `intent = "general_question"`. This is a
  pure keyword heuristic with no LLM call, and its logic never depends on
  `MOCK_LLM`.
- **`retrieve_and_answer`** (reached when `intent == "policy_question"`) --
  embeds the incoming query with `all-MiniLM-L6-v2` and queries the
  `zepto_policy_docs` ChromaDB collection for the top 3 most similar chunks
  by cosine similarity. This retrieval step always runs for real,
  regardless of `MOCK_LLM`.
- **`direct_answer`** (reached when `intent == "general_question"`) -- no
  retrieval is involved.

Graph wiring: `START -> classify_intent`, then a conditional edge
(`_route_after_classify`) sends the state to `retrieve_and_answer` or
`direct_answer` based on `intent`; both nodes then edge to `END`.

### 4. Generation -- `retrieve_and_answer` / `direct_answer` in `graph.py`, using `prompt_template.py`

This is the **only stage that branches on `MOCK_LLM`**:

- **`MOCK_LLM` unset or `"1"` (default):**
  - In `retrieve_and_answer`: no LLM call. `answer = "Based on the
    retrieved context: " + <first ~200 chars of the single most similar
    retrieved chunk's text>`; `sources` = the ids of all 3 retrieved
    chunks; `confidence = 1.0` (fixed).
  - In `direct_answer`: no LLM call. `answer = "I can only answer
    questions about Zepto policies right now."`; `sources = []`;
    `confidence = 1.0`.
- **`MOCK_LLM == "0"` (optional real-LLM branch):**
  - In `retrieve_and_answer`: `prompt_template.build_prompt()` renders the
    structured `SUPPORT_ASSISTANT_PROMPT_TEMPLATE` (Role / Context / Task /
    Format / Length sections, a negative constraint, and a few-shot
    example) with the retrieved chunks as context, then `_call_real_llm()`
    sends it to an OpenAI-compatible / Groq chat-completions endpoint using
    an API key read from the `LLM_API_KEY` environment variable (never
    hardcoded).
  - In `direct_answer`: `_call_real_llm()` is called directly with a
    simple prompt and no retrieved context.
  - In both cases, `_generate_with_retries()` parses the raw LLM output as
    JSON and validates it against the `AnswerResponse` Pydantic model; if
    validation fails, it retries up to `MAX_LLM_RETRIES = 2` additional
    times with `CORRECTIVE_INSTRUCTION` appended to the prompt, and returns
    a clearly-marked `ErrorResponse` (`error=True`, message, safe defaults)
    instead of crashing if it still fails after those retries.

Every node output is enforced through the `AnswerResponse` Pydantic model
(`answer: str`, `sources: list[str]`, `confidence: float` constrained to
`[0.0, 1.0]` via `Field(ge=0.0, le=1.0)`).

### Data flow summary

`docs/doc_0N.txt` --(`ingest.py: load_documents`)--> raw chunk text
--(`ingest.py: get_embedding_model().encode`)--> embedding vectors
--(`ingest.py: build_collection`)--> persisted in ChromaDB (`chroma_db/`)
--(`graph.py: retrieve_and_answer` query-time embed + `collection.query`)-->
top-3 chunks --(`graph.py` mock branch, or `prompt_template.py` +
`_call_real_llm` real branch)--> `AnswerResponse` --(`main.py: POST /ask`)-->
JSON HTTP response.

## API -- `main.py`

FastAPI app with a Pydantic request model `AskRequest {query: str}` and a
`POST /ask` endpoint that calls `graph.run_query(query)` (which invokes the
compiled LangGraph graph) and returns the validated `AnswerResponse`
(`answer`, `sources`, `confidence`) as JSON. A `GET /` health-check route is
also provided.

## Example run output

**Ingestion** (`python ingest.py`):

```
ChromaDB collection 'zepto_policy_docs' built at: .../support_assistant/chroma_db
Total stored vectors: 8
Stored chunk IDs:
  doc_01_chunk_01  (source_doc_id=doc_01)
  doc_02_chunk_01  (source_doc_id=doc_02)
  doc_03_chunk_01  (source_doc_id=doc_03)
  doc_04_chunk_01  (source_doc_id=doc_04)
  doc_05_chunk_01  (source_doc_id=doc_05)
  doc_06_chunk_01  (source_doc_id=doc_06)
  doc_07_chunk_01  (source_doc_id=doc_07)
  doc_08_chunk_01  (source_doc_id=doc_08)
```

**Live server** (`uvicorn main:app --host 127.0.0.1 --port 7860`, `MOCK_LLM`
left unset -> default mock mode), two real HTTP requests via `curl`:

Request 1 -- `POST /ask {"query": "How long does delivery take?"}`
-> routed to `policy_question` -> `retrieve_and_answer`. Raw JSON response
returned by the running server:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01_chunk_01","doc_02_chunk_01","doc_04_chunk_01"],"confidence":1.0}
```

The top-ranked (most similar) retrieved chunk was `doc_01_chunk_01`, i.e.
Zepto's delivery-time policy document (`docs/doc_01.txt`), confirming the
retrieval correctly matched the delivery-policy document for a delivery
question.

Request 2 -- `POST /ask {"query": "What is the capital of France?"}`
-> routed to `general_question` -> `direct_answer`. Raw JSON response
returned by the running server:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

Both responses validated successfully against the `AnswerResponse`
Pydantic schema (`answer: str`, `sources: list[str]`,
`confidence: float` in `[0.0, 1.0]`) -- FastAPI's `response_model=
AnswerResponse` on the `/ask` route enforces this on every response.

The server was then shut down (process on port 7860 terminated) after
capturing these two responses.

## Running it yourself

```bash
source "<repo>/.venv/Scripts/activate"
cd support_assistant
python ingest.py                 # build the ChromaDB collection (one-time / re-run to refresh)
uvicorn main:app --host 0.0.0.0 --port 7860   # MOCK_LLM unset -> mock mode
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "How long does delivery take?"}'
```

To exercise the (structurally present, non-executed) real-LLM branch, set
`MOCK_LLM=0` and `LLM_API_KEY=<your key>` before starting the server.

## Docker

`Dockerfile` installs the project's single root `requirements.txt`, copies
the app code and `docs/`, runs `python ingest.py` at build time to
pre-populate the ChromaDB collection, exposes port 7860, and starts the app
with `uvicorn main:app --host 0.0.0.0 --port 7860`. Its build context is the
**repo root** (so it can reach the root `requirements.txt`), not this
folder:

```bash
# From the repo root (zepto-data-ai-platform/):
docker build -t zepto-support -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support
```

**I wrote the Dockerfile but couldn't test it locally since I don't have
Docker installed (`docker: command not found`).** It has not been
verified beyond manual review of its syntax and layer ordering.

## Files

- `ingest.py` -- ingestion + embedding + ChromaDB collection builder.
- `prompt_template.py` -- structured prompt template + corrective
  instruction for the optional real-LLM branch.
- `graph.py` -- LangGraph `StateGraph`, Pydantic response schema, mock and
  real-LLM generation logic.
- `main.py` -- FastAPI app exposing `POST /ask`.
- `Dockerfile` -- containerization, installs the root `requirements.txt` (untested, see above).
- `docs/doc_01.txt` ... `docs/doc_08.txt` -- Zepto policy corpus (unmodified).
- `chroma_db/` -- persisted ChromaDB collection created by `ingest.py`.
