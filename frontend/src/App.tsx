import { Link, Navigate, Route, Routes } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import AdminPage from './pages/AdminPage';
import { useEffect, useState } from 'react';
import { getBackendReady } from './api';
import { isWithinMaintenanceWindow } from './lib/Maintenance';
import MaintenancePage from './pages/MaintenancePage';
import NotFound from './pages/NotFound';

type ServiceState = 'checking' | 'ready' | 'down';

export default function App() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  const scheduledMaintenance = isWithinMaintenanceWindow(now);
  const [serviceState, setServiceState] = useState<ServiceState>('checking');

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      if (scheduledMaintenance) {
        if (!cancelled) setServiceState('down');
        return;
      }

      try {
        const ready = await getBackendReady();
        if (!cancelled) setServiceState(ready ? 'ready' : 'down');
      } catch {
        if (!cancelled) setServiceState('down');
      }
    };

    void check();
    const timer = window.setInterval(check, 30000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [scheduledMaintenance]);

  if (scheduledMaintenance) {
    return <MaintenancePage reason="scheduled" />;
  }

  if (serviceState === 'checking') {
    return (
      <div className="page">
        <div className="card">
          <p className="muted">Checking service availability…</p>
        </div>
      </div>
    );
  }

  if (serviceState === 'down') {
    return <MaintenancePage reason="outage" />;
  }

  return (
    <div>
      <header className="topbar">
        <Link to="/" className="brand">
          PneumoAI
        </Link>
        <nav className="nav">
          <Link to="/">Predict</Link>
          <Link to="/admin">Admin</Link>
        </nav>
      </header>

      <main className="shell">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/jobs/:requestId" element={<ResultPage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="/maintenance" element={<MaintenancePage reason="scheduled" />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}