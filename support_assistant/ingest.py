"""
ingest.py -- Module 3 (support_assistant) ingestion stage.

I wrote this script to load all 8 Zepto policy documents from docs/doc_0N.txt. 
I decided to treat each whole document as a single chunk (which I felt was acceptable given their short length). 
Then, I programmed the script to embed each chunk with the sentence-transformers 'all-MiniLM-L6-v2' model, 
and store my resulting 8 embeddings in a local persistent ChromaDB collection.

Run directly to let me (re)build the collection for you:
    python ingest.py
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policy_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

NUM_DOCS = 8


def load_documents():
    """I wrote this helper to read docs/doc_01.txt ... doc_08.txt. Each whole file is one chunk."""
    documents = []
    for i in range(1, NUM_DOCS + 1):
        doc_id = f"doc_{i:02d}"
        path = os.path.join(DOCS_DIR, f"{doc_id}.txt")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        documents.append({"doc_id": doc_id, "text": text})
    return documents


def get_embedding_model():
    # I chose the 'all-MiniLM-L6-v2' model because it balances speed and quality perfectly for this.
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_chroma_client():
    # I'm using PersistentClient so my database survives across runs.
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_collection(reset: bool = True):
    """My function to embed all 8 documents and (re)build the persistent ChromaDB collection."""
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    documents = load_documents()
    model = get_embedding_model()

    ids = []
    texts = []
    metadatas = []
    for doc in documents:
        # I am tagging each chunk carefully so I can trace it back to its source document
        chunk_id = f"{doc['doc_id']}_chunk_01"
        ids.append(chunk_id)
        texts.append(doc["text"])
        metadatas.append({"source_doc_id": doc["doc_id"]})

    # This is where I actually perform the embedding via the sentence-transformers library
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return collection


def main():
    collection = build_collection(reset=True)
    count = collection.count()
    print(f"My ChromaDB collection '{COLLECTION_NAME}' has been built at: {CHROMA_DIR}")
    print(f"Total stored vectors I processed: {count}")
    result = collection.get()
    print("My stored chunk IDs:")
    for cid, meta in zip(result["ids"], result["metadatas"]):
        print(f"  {cid}  (source_doc_id={meta['source_doc_id']})")


if __name__ == "__main__":
    main()
