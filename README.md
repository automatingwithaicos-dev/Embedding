# Help Desk RAG: Retrieval-Augmented Answers from an IT Knowledge Base

A small retrieval-augmented generation (RAG) system that answers IT help desk questions from a folder of runbooks. It splits documents into chunks, turns each chunk into an embedding vector, finds the chunks most similar to a question, and has an LLM write an answer grounded only in those chunks, with citations.

The sample knowledge base covers everyday Tier 1 and Tier 2 work: password resets and lockouts, VPN troubleshooting, MFA, Intune enrollment and compliance, printers, and user onboarding and offboarding.

## How it works

```
data/kb/*.md ──► chunk (120 words, 30 word overlap) ──► embed ──► vector index (index/)
                                                                        │
question ──► embed ──► cosine similarity search ──► top k chunks ──► LLM answer with [1] [2] citations
```

1. **Chunking** (`rag/loader.py`): documents are split on paragraph boundaries into chunks of about 120 words. Each chunk overlaps the previous one by 30 words, so an answer that crosses a boundary is still retrievable.
2. **Embedding** (`rag/embedder.py`): each chunk becomes a normalized vector.
   - `fastembed` (default) runs the `BAAI/bge-small-en-v1.5` sentence-embedding model locally on CPU. No GPU and no API key needed. It matches by meaning, so "an employee is leaving" finds the offboarding runbook.
   - `tfidf` is a keyword-based fallback with no model download, used by the tests and on offline machines.
3. **Retrieval** (`rag/store.py`): the question is embedded the same way, and the index returns the chunks with the highest cosine similarity. The index is saved to disk so documents are only embedded once.
4. **Generation** (`rag/answer.py`): if `ANTHROPIC_API_KEY` is set, Claude answers using only the retrieved passages and cites them. If the best match scores below a relevance threshold, the tool says it doesn't know and suggests escalating, instead of guessing. Without an API key, it shows the best-matching passages.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m rag ingest             # embed data/kb into index/
python -m rag ask "A user's VPN says the server is unreachable" --sources
python -m rag chat               # interactive mode
```

For written answers instead of raw passages:

```bash
export ANTHROPIC_API_KEY=your-key    # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key"
python -m rag ask "How do I reset MFA for someone who lost their phone?"
```

No internet for the model download? Use the keyword backend:

```bash
python -m rag ingest --backend tfidf
```

## Use your own documents

Put `.md` or `.txt` files in any folder and point the tool at it:

```bash
python -m rag ingest --docs path/to/your/runbooks
```

## Tests

```bash
pytest
```

The tests check chunking, that each sample question retrieves the right runbook, that a saved index loads back identically, and that an off-topic question is refused rather than answered.

## Project layout

```
rag/
  loader.py     load documents and split them into overlapping chunks
  embedder.py   fastembed and tfidf embedding backends
  store.py      vector index: build, search, save, load
  answer.py     grounded LLM answer with citations, or extractive fallback
  cli.py        ingest, ask, and chat commands
data/kb/        sample IT help desk runbooks
tests/          pytest suite
```

## Ideas for next steps

- Pull tickets from a PSA such as ConnectWise or ServiceNow and suggest replies from the knowledge base.
- Add an agent step that can take safe actions, such as looking up a user's lockout status, before answering.
- Swap the on-disk index for a vector database like Chroma or pgvector as the knowledge base grows.
