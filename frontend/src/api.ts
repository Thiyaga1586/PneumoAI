import type { PredictionJob } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_PATH ?? '/api').replace(/\/$/, '');
const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB ?? '10');

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get('content-type') ?? '';
  const payload = contentType.includes('application/json')
    ? await response.json().catch(() => null)
    : await response.text().catch(() => '');

  if (!response.ok) {
    const detail =
      typeof payload === 'string'
        ? payload
        : payload?.detail || payload?.message || response.statusText || 'Request failed';
    throw new ApiError(response.status, detail);
  }

  return payload as T;
}

export function getMaxUploadBytes(): number {
  return MAX_UPLOAD_MB * 1024 * 1024;
}

export async function submitPrediction(file: File, trueLabel?: string): Promise<PredictionJob> {
  if (file.size > getMaxUploadBytes()) {
    throw new ApiError(400, `File is too large. Max allowed is ${MAX_UPLOAD_MB} MB.`);
  }

  const form = new FormData();
  form.append('file', file);

  if (trueLabel && trueLabel.trim()) {
    form.append('true_label', trueLabel.trim());
  }

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: form,
  });

  return parseResponse<PredictionJob>(response);
}

export async function fetchJobStatus(requestId: string): Promise<PredictionJob> {
  const response = await fetch(`${API_BASE}/predict/${encodeURIComponent(requestId)}`, {
    method: 'GET',
  });

  return parseResponse<PredictionJob>(response);
}

export async function getBackendReady(): Promise<boolean> {
  const response = await fetch(`${API_BASE}/ready`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });

  return response.ok;
}

export function formatProbability(
    value?: number | null
): string {

    if (value == null)
        return "—";

    const percent = value * 100;

    if (percent >= 99.995)
        return ">99.99%";

    if (percent <= 0.005)
        return "<0.01%";

    return `${percent.toFixed(3)}%`;
}

export function formatLatencyMs(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${value.toFixed(2)} ms`;
}