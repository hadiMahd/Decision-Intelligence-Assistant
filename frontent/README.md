# Frontend (React)

Minimal React (Vite) UI for hardcoded ticket tests.

## What it does

1. Lets you pick a hardcoded support ticket.
2. Runs LLM comparison: with RAG vs without RAG.
3. Runs ML inference comparison via backend external-model endpoint.

## Install and Run

```bash
cd frontent
npm install
npm run dev
```

Open `http://localhost:5173`.

Default backend URL in the page is `http://localhost:8000`.

## Flow

1. Click "Seed hardcoded history" once.
2. Click "Run RAG comparison" to see side-by-side answers.
3. Click "Run ML inference compare" for model output comparison.
