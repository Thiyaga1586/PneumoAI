export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export interface PredictionJob {
  request_id: string;
  status: JobStatus;
  created_at?: string | null;
  model_version?: string | null;
  prediction?: string | null;
  probability?: number | null;
  threshold?: number | null;
  latency_ms?: number | null;
  backend?: string | null;
  true_label?: string | null;
  error?: string | null;
}