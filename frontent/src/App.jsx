import { useMemo, useState } from "react";

const FETCHED_TICKETS = [
  "I canceled my subscription but still have not received my refund. Can you help?",
  "The app is down and all my team sees is service unavailable during login.",
  "My order arrived broken and I need a replacement immediately.",
  "Checkout fails with error 502 whenever I try to pay by card.",
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

export default function App() {
  const backendUrl = "http://localhost:8000";
  const [selectedTicket, setSelectedTicket] = useState(FETCHED_TICKETS[0]);
  const [ticketText, setTicketText] = useState("");
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [ragResult, setRagResult] = useState(null);
  const [mlResult, setMlResult] = useState(null);

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
                <p>{ragResult.no_rag_answer}</p>
              </article>
              <article className="result-card result-accent">
                <Badge tone="green">With RAG</Badge>
                <p>{ragResult.rag_answer}</p>
              </article>
            </div>
          )}

          {ragResult && (
            <div className="subsection">
              <div className="subsection-title">Retrieved tickets</div>
              <div className="ticket-list">
                {(ragResult.retrieved_tickets || []).map((ticket, idx) => (
                  <article className="mini-card" key={ticket.ticket_id || idx}>
                    <strong>{ticket.ticket_id || `Ticket ${idx + 1}`}</strong>
                    <div>{ticket.title || "No title"}</div>
                    <p>{ticket.description || "No description"}</p>
                    {ticket.resolution && <small>Resolution: {ticket.resolution}</small>}
                  </article>
                ))}
              </div>
            </div>
          )}

          {ragResult && (
            <div className="metric-row">
              <div className="metric-card">
                <span>Context overlap</span>
                <strong>{String(ragResult.scores?.context_overlap_score ?? "-")}</strong>
              </div>
              <div className="metric-card">
                <span>Grounding gain</span>
                <strong>{String(ragResult.scores?.grounding_gain_score ?? "-")}</strong>
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
            <div className="metric-grid metric-grid-3">
              <div className="metric-card">
                <span>Raw model prediction</span>
                <strong>{String(mlResult.raw_model_prediction)}</strong>
                <small>Confidence: {formatConfidence(mlResult.raw_model_confidence)}</small>
              </div>
              <div className="metric-card">
                <span>Engineered prediction</span>
                <strong>{String(mlResult.engineered_model_prediction)}</strong>
                <small>Confidence: {formatConfidence(mlResult.engineered_model_confidence)}</small>
              </div>
              <div className="metric-card">
                <span>Disagreement</span>
                <strong>{String(mlResult.disagreement)}</strong>
                <small>True means labels differ.</small>
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
