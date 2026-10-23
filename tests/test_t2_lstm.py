"""
tests/test_t2_lstm.py
---------------------
Unit tests for T²-LSTM Hybrid Anomaly Detector

⚠️ หมายเหตุสำคัญ: ข้อมูลทั้งหมดในไฟล์นี้เป็น Mock Data เพื่อการทดสอบทางเทคนิคเท่านั้น
ไม่ได้ใช้ข้อมูลจริงจาก Tennessee Eastman Process
"""

import numpy as np
import pytest
import torch

from src.models.t2_lstm import T2LSTMDetector, T2LSTMNet


def _make_mock_sequence_data(
    n_samples: int = 100,
    seq_len: int = 15,
    n_features: int = 8,
    is_anomaly: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """สร้าง Mock Data ในรูปแบบ 3D Time-series (⚠️ MOCK DATA)"""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, seq_len)

    data = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    for i in range(n_samples):
        for f in range(n_features):
            phase = rng.uniform(0, np.pi)
            data[i, :, f] = np.sin(t + phase) + rng.normal(0, 0.1, size=seq_len)

    if is_anomaly:
        data += rng.normal(4.5, 1.2, size=data.shape).astype(np.float32)

    return data


class TestT2LSTM:
    def test_t2_lstm_net_shapes(self):
        """ทดสอบ Output Shapes ของ T2LSTMNet (⚠️ MOCK DATA)"""
        batch_size = 16
        seq_len = 20
        input_dim = 10
        latent_dim = 16
        hidden_dim = 32

        net = T2LSTMNet(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
        )

        x_mock = torch.randn(batch_size, seq_len, input_dim)
        recon, z = net(x_mock)

        assert recon.shape == (batch_size, seq_len, input_dim)
        assert z.shape == (batch_size, latent_dim)

    def test_t2_lstm_fit_and_limits(self):
        """ทดสอบว่า T²-LSTM fit สำเร็จและคำนวณ Control Limits ถูกต้อง (⚠️ MOCK DATA)"""
        X_train_mock = _make_mock_sequence_data(n_samples=120, seq_len=12, n_features=6, seed=42)
        X_val_mock = _make_mock_sequence_data(n_samples=40, seq_len=12, n_features=6, seed=43)

        detector = T2LSTMDetector(
            input_dim=6,
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            gamma=0.5,
            significance=0.05,
            random_seed=42,
        )

        detector.fit(
            X_train_normal=X_train_mock,
            X_val_normal=X_val_mock,
            epochs=8,
            batch_size=32,
            lr=5e-3,
        )

        assert detector.is_fitted
        assert detector.t2_limit_ > 0
        assert detector.spe_limit_ > 0
        assert detector.latent_mean_ is not None
        assert detector.latent_cov_inv_ is not None
        assert detector.latent_cov_inv_.shape == (8, 8)
        assert len(detector.history_["train_loss"]) == 8

    def test_t2_lstm_detects_anomaly(self):
        """ทดสอบการตรวจจับความผิดปกติของ T²-LSTM (⚠️ MOCK DATA)"""
        X_train_mock = _make_mock_sequence_data(n_samples=120, seq_len=12, n_features=6, is_anomaly=False, seed=42)
        X_normal_test = _make_mock_sequence_data(n_samples=30, seq_len=12, n_features=6, is_anomaly=False, seed=90)
        X_anomaly_test = _make_mock_sequence_data(n_samples=30, seq_len=12, n_features=6, is_anomaly=True, seed=99)

        detector = T2LSTMDetector(
            input_dim=6,
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            gamma=0.5,
            random_seed=42,
        )
        detector.fit(X_train_normal=X_train_mock, epochs=10, batch_size=32, lr=5e-3)

        comp_normal, t2_norm, spe_norm = detector.score(X_normal_test, return_components=True)
        comp_anom, t2_anom, spe_anom = detector.score(X_anomaly_test, return_components=True)

        assert np.mean(comp_anom) > np.mean(comp_normal)
        assert np.mean(t2_anom) > np.mean(t2_norm)
        assert np.mean(spe_anom) > np.mean(spe_norm)

        preds = detector.predict(X_anomaly_test)
        fdr = np.mean(preds == 1)
        assert fdr >= 0.80, f"FDR ควรสูงกว่า 80% แต่ได้ {fdr:.2%}"

    def test_t2_lstm_save_and_load(self, tmp_path):
        """ทดสอบการบันทึกและโหลด Checkpoint โมเดล T²-LSTM (⚠️ MOCK DATA)"""
        X_train_mock = _make_mock_sequence_data(n_samples=50, seq_len=10, n_features=5, seed=42)

        detector = T2LSTMDetector(input_dim=5, latent_dim=6, hidden_dim=12, num_layers=1, random_seed=42)
        detector.fit(X_train_normal=X_train_mock, epochs=5, batch_size=25)

        X_eval = _make_mock_sequence_data(n_samples=15, seq_len=10, n_features=5, seed=88)
        orig_scores = detector.score(X_eval)

        save_path = tmp_path / "t2_lstm_ckpt.pt"
        detector.save(save_path)

        loaded_detector = T2LSTMDetector.load(save_path)
        loaded_scores = loaded_detector.score(X_eval)

        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)
        assert loaded_detector.t2_limit_ == pytest.approx(detector.t2_limit_)
        assert loaded_detector.spe_limit_ == pytest.approx(detector.spe_limit_)
