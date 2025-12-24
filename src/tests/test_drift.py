import unittest
import json
from unittest.mock import patch

from src.pneumonia_system.mlops.drift import psi, drift_check_and_maybe_rollback

class TestDrift(unittest.TestCase):
    def test_psi_zero_when_same(self):
        expected = [0.1] * 10
        actual = [0.1] * 10
        score = psi(expected, actual)
        self.assertAlmostEqual(score, 0.0, places=6)

    @patch("src.pneumonia_system.mlops.drift.rollback")
    @patch("src.pneumonia_system.mlops.drift.recent_requests")
    @patch("src.pneumonia_system.mlops.drift.load_baseline")
    def test_drift_triggers_rollback(self, mock_load_baseline, mock_recent_requests, mock_rollback):
        # baseline: mostly in first bin
        baseline = [0.90] + [0.10/31.0]*31
        mock_load_baseline.return_value = (32, baseline)

        # current: shift mass to last bin => high PSI
        current = [0.10/31.0]*31 + [0.90]

        # recent_requests must return tuples:
        # (ts_utc, model_version, latency_ms, label, probability, hist_json, error, true_label)
        mock_recent_requests.return_value = [
            ("2025-01-01T00:00:00Z", "v1", 10.0, "Normal", 0.1, json.dumps(current), None, None)
        ]

        score = drift_check_and_maybe_rollback(version="v1", window=1, threshold=0.01)
        self.assertTrue(score >= 0.01)
        mock_rollback.assert_called_once()

    @patch("src.pneumonia_system.mlops.drift.rollback")
    @patch("src.pneumonia_system.mlops.drift.recent_requests")
    @patch("src.pneumonia_system.mlops.drift.load_baseline")
    def test_drift_no_rollback_when_low(self, mock_load_baseline, mock_recent_requests, mock_rollback):
        baseline = [0.05]*32
        mock_load_baseline.return_value = (32, baseline)

        current = [0.05]*32
        mock_recent_requests.return_value = [
            ("2025-01-01T00:00:00Z", "v1", 10.0, "Normal", 0.1, json.dumps(current), None, None)
        ]

        score = drift_check_and_maybe_rollback(version="v1", window=1, threshold=0.25)
        self.assertTrue(score < 0.25)
        mock_rollback.assert_not_called()

if __name__ == "__main__":
    unittest.main()
