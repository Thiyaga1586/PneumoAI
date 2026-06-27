import { useEffect, useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { ApiError, getMaxUploadBytes, submitPrediction } from '../api';

function bytesToMB(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(0)} MB`;
}

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [trueLabel, setTrueLabel] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
  if (!file) {
      setPreviewUrl(null);
      return;
  }

  const url = URL.createObjectURL(file);
  setPreviewUrl(url);

  return () => URL.revokeObjectURL(url);
  }, [file]);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);

    if (!file) {
      setError('Please choose an image file.');
      return;
    }

    setSubmitting(true);
    try {
      const result = await submitPrediction(file, trueLabel);
      localStorage.setItem('pneumoai:last_request_id', result.request_id);
      navigate(`/jobs/${result.request_id}`, { replace: true });
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Upload failed.');
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">PneumoAI Prediction Portal</p>
          <h1>Upload a chest X-ray and get a prediction job instantly.</h1>
          <p className="muted">
            The backend queues the image, the worker processes it, and this page polls the result by request ID.
          </p>
        </div>
      </section>

      <form className="card upload-card" onSubmit={onSubmit}>
        <label className="field">
          <span>Image file</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              setError(null);
              const next = e.target.files?.[0] ?? null;
              setFile(next);
            }}
          />
        </label>

        <label className="field">
          <span>Optional true label</span>
          <input
            type="text"
            value={trueLabel}
            onChange={(e) => setTrueLabel(e.target.value)}
            placeholder="Example: pneumonia / normal"
          />
        </label>

        <div className="hint">
          Max upload size: {bytesToMB(getMaxUploadBytes())}. Supported: image files.
        </div>

        {file ? (
          <div className="preview">
            <div className="preview-meta">
              <strong>{file.name}</strong>
              <span>{bytesToMB(file.size)}</span>
            </div>
            {previewUrl ? <img src={previewUrl} alt="Preview" /> : null}
          </div>
        ) : null}

        {error ? <div className="error-box">{error}</div> : null}

        <button className="primary-btn" type="submit" disabled={submitting}>
          {submitting ? 'Submitting…' : 'Run prediction'}
        </button>
      </form>
    </div>
  );
}