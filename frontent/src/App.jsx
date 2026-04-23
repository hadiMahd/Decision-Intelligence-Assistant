import { useMemo, useState } from "react";

const FETCHED_TICKETS = [
  "How do I cancel my subscription from billing settings?",
  "How can I reset MFA if I lost access to my authenticator app?",
  "How do I export reports as CSV for a selected date range?",
  "How do I invite a teammate to my workspace?",
];

async function postJson(baseUrl, route, payload) {
  const response = await fetch(`${baseUrl}${route}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return response.json();
}

function Badge({ children, tone = "blue" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Section({ title, subtitle, children, tone = "blue" }) {
  return (
    <section className={`card card-${tone}`}>
      <div className="section-header">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

function formatConfidence(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  return Number.isNaN(n) ? String(value) : `${(n * 100).toFixed(1)}%`;
}

function formatSimilarityScore(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  return Number.isNaN(n) ? "-" : `${(n * 100).toFixed(1)}%`;
}

function formatUrgencyLabel(value) {
  if (value === null || value === undefined || value === "") return "-";

  const normalized = String(value).trim().toLowerCase();
  if (normalized === "1" || normalized === "urgent") return "urgent";
  if (normalized === "0" || normalized === "not_urgent" || normalized === "not urgent") return "not urgent";

  return String(value);
}

export default function App() {
  const backendUrl = "http://localhost:8000";
  const [selectedTicket, setSelectedTicket] = useState(FETCHED_TICKETS[0]);
  const [ticketText, setTicketText] = useState("");
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [ragResult, setRagResult] = useState(null);
  const [mlResult, setMlResult] = useState(null);
  const [ingestText, setIngestText] = useState("");
  const [ingestId, setIngestId] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState(null);

  const activeTicket = useMemo(() => ticketText.trim() || selectedTicket, [ticketText, selectedTicket]);

  async function runAction(action) {
    setError("");
    setLoading(true);
    try {
      await action();
    } catch (err) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <Section
        title="Support Assistant Lab"
        subtitle="This app compares support ticket answers with and without RAG, then compares two ML inference paths on any ticket text you enter. Your Qdrant collection should already contain embeddings from your own ticket data."
        tone="blue"
      >
        <div className="hero-tags">
          <Badge tone="blue">Fetched tickets</Badge>
          <Badge tone="green">RAG / No-RAG</Badge>
          <Badge tone="amber">ML compare</Badge>
        </div>
      </Section>

      <Section
        title="Ticket Input"
        subtitle="Pick a fetched ticket or type your own custom ticket text."
        tone="blue"
      >
        <div className="field-stack">
          <div className="field-panel">
            <label className="field field-compact">
              <span>Fetched ticket</span>
              <select value={selectedTicket} onChange={(e) => setSelectedTicket(e.target.value)}>
                {FETCHED_TICKETS.map((ticket) => (
                  <option value={ticket} key={ticket}>
                    {ticket}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="field-panel field-panel-alt">
            <label className="field field-compact">
              <span>Custom ticket text</span>
              <textarea
                rows={5}
                value={ticketText}
                onChange={(e) => setTicketText(e.target.value)}
                placeholder="Type any support ticket here to override the fetched ticket"
              />
            </label>
          </div>
        </div>

        <div className="preview-box">
          <Badge tone="green">Active ticket</Badge>
          <p>{activeTicket || "No ticket entered yet."}</p>
        </div>
      </Section>

      <section className="content">
        <Section
          title="RAG vs No-RAG"
          subtitle="One request gives you both answers plus retrieved tickets from your existing Qdrant data."
          tone="green"
        >
          <div className="actions-row">
            <label className="inline-field">
              <span>Top-K</span>
              <input
                type="number"
                min={1}
                max={5}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value) || 3)}
              />
            </label>

            <button
              className="primary-btn"
              disabled={loading || !activeTicket}
              onClick={() =>
                runAction(async () => {
                  const data = await postJson(backendUrl, "/rag/compare", {
                    ticket_text: activeTicket,
                    top_k: topK,
                  });
                  setRagResult(data);
                })
              }
            >
              Run RAG compare
            </button>
          </div>

          {ragResult && (
            <div className="results-grid">
              <article className="result-card result-soft">
                <Badge tone="blue">Without RAG</Badge>
                <div className="answer-text">{ragResult.no_rag_answer}</div>
              </article>
              <article className="result-card result-accent">
                <Badge tone="green">With RAG</Badge>
                <div className="answer-text">{ragResult.rag_answer}</div>
              </article>
            </div>
          )}

          {ragResult && (
            <div className="subsection">
              <div className="subsection-title">Retrieved tickets</div>
              <div className="ticket-list">
                {(ragResult.retrieved_tickets || []).map((ticket, idx) => (
                  <article className="mini-card" key={ticket.id || idx}>
                    <div className="ticket-header">
                      <div className="ticket-id-source">
                        <strong>ID: {ticket.id || "unknown"}</strong>
                        <span className="ticket-source">Source: {ticket.source || "unknown"}</span>
                        <span className="ticket-score">Similarity: {formatSimilarityScore(ticket.similarity_score)}</span>
                      </div>
                    </div>
                    <p>{ticket.text || "No text"}</p>
                  </article>
                ))}
              </div>
            </div>
          )}

          {ragResult && (
            <div className="metric-row">
              <div className="metric-card">
                <span>Retrieved chunks</span>
                <strong>{ragResult.retrieved_tickets?.length ?? 0}</strong>
              </div>
            </div>
          )}

          {ragResult && (
            <details className="details-box">
              <summary>Raw retrieved payload</summary>
              <pre>{JSON.stringify(ragResult.retrieved_tickets, null, 2)}</pre>
            </details>
          )}
        </Section>

        <Section
          title="RAG Search"
          subtitle="Search your Qdrant collection with a query string."
          tone="green"
        >
          <div className="actions-row">
            <label className="inline-field">
              <span>Top-K</span>
              <input
                type="number"
                min={1}
                max={5}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value) || 3)}
              />
            </label>

            <label className="field field-compact flex-grow">
              <span>Search query</span>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="e.g., vpn disconnecting"
              />
            </label>

            <button
              className="primary-btn"
              disabled={loading || !searchQuery.trim()}
              onClick={() =>
                runAction(async () => {
                  const data = await postJson(backendUrl, "/rag/search", {
                    query: searchQuery,
                    top_k: topK,
                  });
                  setSearchResults(data);
                })
              }
            >
              Search
            </button>
          </div>

          {searchResults && (
            <div className="preview-box">
              <Badge tone="green">Query</Badge>
              <p>{searchResults.query}</p>
            </div>
          )}

          {searchResults && (
            <div className="subsection">
              <div className="subsection-title">Search results ({searchResults.results.length})</div>
              <div className="ticket-list">
                {(searchResults.results || []).map((result, idx) => (
                  <article className="mini-card" key={result.id || idx}>
                    <div className="ticket-header">
                      <div className="ticket-id-source">
                        <strong>ID: {result.id || `Result ${idx + 1}`}</strong>
                        <span className="ticket-source">Source: {result.source || "unknown"}</span>
                        <span className="ticket-score">Similarity: {formatSimilarityScore(result.similarity_score)}</span>
                      </div>
                    </div>
                    <p>{result.text || "No text"}</p>
                  </article>
                ))}
              </div>
            </div>
          )}
        </Section>

        <Section
          title="Manual Text Ingest"
          subtitle="Ingest a custom text into Qdrant for search testing."
          tone="blue"
        >
          <div className="field-stack">
            <label className="field field-compact">
              <span>Text to ingest</span>
              <textarea
                rows={3}
                value={ingestText}
                onChange={(e) => setIngestText(e.target.value)}
                placeholder="Enter ticket text to add to collection"
              />
            </label>

            <label className="field field-compact">
              <span>Ticket ID (optional)</span>
              <input
                type="text"
                value={ingestId}
                onChange={(e) => setIngestId(e.target.value)}
                placeholder="e.g., manual-001"
              />
            </label>
          </div>

          <button
            className="primary-btn"
            disabled={loading || !ingestText.trim()}
            onClick={() =>
              runAction(async () => {
                await postJson(backendUrl, "/rag/ingest-text", {
                  text: ingestText,
                  id: ingestId || undefined,
                  source: "manual_test",
                });
                setIngestText("");
                setIngestId("");
                alert("Text ingested successfully!");
              })
            }
          >
            Ingest text
          </button>
        </Section>

        <Section
          title="ML Inference Compare"
          subtitle="Compare raw-text and engineered-features predictions from the API."
          tone="amber"
        >
          <button
            className="primary-btn"
            disabled={loading || !activeTicket}
            onClick={() =>
              runAction(async () => {
                const data = await postJson(backendUrl, "/ml/compare-inference", {
                  raw_text: activeTicket,
                });
                setMlResult(data);
              })
            }
          >
            Run ML compare
          </button>

          {mlResult && (
            <div className="metric-grid metric-grid-4">
              <div className="metric-card">
                <span>Raw model prediction</span>
                <strong>{formatUrgencyLabel(mlResult.raw_model_prediction)}</strong>
                <small>Confidence: {formatConfidence(mlResult.raw_model_confidence)}</small>
              </div>
              <div className="metric-card">
                <span>Engineered prediction</span>
                <strong>{formatUrgencyLabel(mlResult.engineered_model_prediction)}</strong>
                <small>Confidence: {formatConfidence(mlResult.engineered_model_confidence)}</small>
              </div>
              <div className="metric-card">
                <span>LLM urgency</span>
                <strong>{String(mlResult.llm_prediction)}</strong>
                <small>urgent or not_urgent</small>
              </div>
              <div className="metric-card">
                <span>Raw vs LLM</span>
                <strong>{mlResult.raw_vs_llm_disagreement ? "Disagree" : "Agree"}</strong>
                <small>Models align on urgency</small>
              </div>
            </div>
          )}

          {mlResult && (
            <div className="metric-grid metric-grid-2">
              <div className="metric-card">
                <span>ML models disagreement</span>
                <strong>{String(mlResult.disagreement)}</strong>
                <small>Raw vs Engineered differ</small>
              </div>
              <div className="metric-card">
                <span>Engineered vs LLM</span>
                <strong>{mlResult.engineered_vs_llm_disagreement ? "Disagree" : "Agree"}</strong>
                <small>Models align on urgency</small>
              </div>
            </div>
          )}

          {mlResult && (
            <details className="details-box">
              <summary>Raw model API response</summary>
              <pre>{JSON.stringify(mlResult.external_raw_response, null, 2)}</pre>
            </details>
          )}
        </Section>

        {loading && <div className="status">Running request...</div>}
        {error && <div className="status error">{error}</div>}
      </section>
    </main>
  );
}
