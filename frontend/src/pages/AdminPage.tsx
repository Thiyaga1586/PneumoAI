const grafanaUrl = import.meta.env.VITE_GRAFANA_URL ?? '/grafana/';

export default function AdminPage() {
  return (
    <div className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">Admin Console</p>
          <h1>Operations and observability</h1>
          <p className="muted">
            Grafana is the admin surface. The real access control must be enforced at the reverse proxy or SSO layer.
          </p>
        </div>
      </section>

      <div className="card">
        <h2>Grafana</h2>
        <p className="muted">
          Open the production dashboard, review throughput, latency, drift, and runtime health.
        </p>
        <a className="primary-btn inline" href={grafanaUrl} target="_blank" rel="noreferrer">
          Open Grafana
        </a>
      </div>
    </div>
  );
}