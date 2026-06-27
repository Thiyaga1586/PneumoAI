import { Link } from 'react-router-dom';
import { maintenanceWindowLabel } from '../lib/Maintenance';

type Props = {
  reason: 'scheduled' | 'outage';
};

export default function MaintenancePage({ reason }: Props) {
  const scheduled = reason === 'scheduled';

  return (
    <div className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">PneumoAI</p>
          <h1>{scheduled ? 'Scheduled maintenance' : 'Service temporarily unavailable'}</h1>
          <p className="muted">
            {scheduled
              ? `PneumoAI is unavailable during ${maintenanceWindowLabel()}.`
              : `The app or backend is not ready right now. If this is within ${maintenanceWindowLabel()}, this is expected.`}
          </p>
        </div>
      </section>

      <div className="card">
        <p className="muted">
          {scheduled
            ? 'Please return after the maintenance window ends.'
            : 'Please try again in a moment.'}
        </p>

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginTop: '16px' }}>
          <button className="primary-btn" onClick={() => window.location.reload()}>
            Retry
          </button>
          <Link className="secondary-btn" to="/">
            Go to home
          </Link>
        </div>
      </div>
    </div>
  );
}