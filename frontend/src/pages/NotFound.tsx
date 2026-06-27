import { Link } from 'react-router-dom';
import { maintenanceWindowLabel } from '../lib/Maintenance';

export default function NotFound() {
  return (
    <div className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">PneumoAI</p>
          <h1>Page not found</h1>
          <p className="muted">
            The route you opened does not exist. Scheduled downtime is {maintenanceWindowLabel()}.
          </p>
        </div>
      </section>

      <div className="card">
        <p className="muted">
          If you were looking for predictions, go back to the upload page. If the service is under
          maintenance, this notice is expected.
        </p>

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginTop: '16px' }}>
          <Link className="primary-btn inline" to="/">
            Back to home
          </Link>
          <Link className="secondary-btn" to="/maintenance">
            Maintenance page
          </Link>
        </div>
      </div>
    </div>
  );
}