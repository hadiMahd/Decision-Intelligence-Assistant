# Backend

Minimal FastAPI backend for two comparisons:

1. LLM answer quality: with RAG vs without RAG
2. ML inference comparison: raw-text model vs engineered-features model (via external model API)

## Environment

Create `.env` in backend folder:

```env
OPENAI_API_KEY=your_key_here
ANSWERING_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Optional: use remote Qdrant instead of local embedded storage
QDRANT_URL=
QDRANT_COLLECTION=support_tickets
QDRANT_VECTOR_SIZE=1536
QDRANT_LOCAL_PATH=artifacts/qdrant

# External model APIs (one endpoint per model)
EXTERNAL_RAW_MODEL_API_URL=http://localhost:9000/predict/raw
EXTERNAL_ENGINEERED_MODEL_API_URL=http://localhost:9000/predict/engineered
EXTERNAL_MODEL_API_TIMEOUT_SECONDS=15
```

## Install and Run

```bash
cd backend
pip install -e .
python main.py
```

Backend starts at `http://localhost:8000`.

The RAG flow assumes your Qdrant collection already contains embeddings built from your own ticket data.

## Endpoints

- `GET /health`
- `POST /rag/compare`
- `POST /ml/compare-inference`

### Example: RAG Compare

```bash
curl -X POST http://localhost:8000/rag/compare \
	-H "Content-Type: application/json" \
	-d '{"ticket_text":"My order is broken and I need replacement", "top_k":3}'
```

### Example: ML Compare

```bash
curl -X POST http://localhost:8000/ml/compare-inference \
	-H "Content-Type: application/json" \
	-d '{"raw_text":"App is down, login fails for everyone"}'
```
