import { useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, fetchJobStatus } from '../api';
import type { PredictionJob } from '../types';

interface State {
  job: PredictionJob | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useJobPoll(requestId?: string | null, intervalMs = 2500): State {
  const [job, setJob] = useState<PredictionJob | null>(null);
  const [loading, setLoading] = useState<boolean>(Boolean(requestId));
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<number | null>(null);

  const clearTimer = () => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const refresh = useMemo(
    () => async () => {
      if (!requestId) return;
      setLoading(true);
      try {
        const next = await fetchJobStatus(requestId);
        setJob(next);
        setError(null);
      } catch (err) {
        if (err instanceof ApiError) {
          setError(err.detail);
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError('Failed to load prediction status.');
        }
      } finally {
        setLoading(false);
      }
    },
    [requestId]
  );

  useEffect(() => {
    if (!requestId) {
      setJob(null);
      setLoading(false);
      setError(null);
      clearTimer();
      return;
    }

    let cancelled = false;

    const run = async () => {
      try {
        const next = await fetchJobStatus(requestId);
        if (cancelled) return;
        setJob(next);
        setError(null);
        setLoading(false);

        if (next.status === 'completed' || next.status === 'failed') {
          clearTimer();
        }
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError) {
          setError(err.detail);
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError('Failed to load prediction status.');
        }
        setLoading(false);
      }
    };

    run();

    clearTimer();
    timerRef.current = window.setInterval(run, intervalMs);

    return () => {
      cancelled = true;
      clearTimer();
    };
  }, [requestId, intervalMs]);

  return { job, loading, error, refresh };
}