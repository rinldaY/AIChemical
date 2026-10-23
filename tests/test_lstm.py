"""
tests/test_lstm.py
------------------
Unit tests for Stacked LSTM Sequence Autoencoder

⚠️ หมายเหตุสำคัญ: ข้อมูลทั้งหมดในไฟล์นี้เป็น Mock Data เพื่อการทดสอบทางเทคนิคเท่านั้น
ไม่ได้ใช้ข้อมูลจริงจาก Tennessee Eastman Process
"""

import numpy as np
import pytest
import torch

from src.models.stacked_lstm import StackedLSTMDetector, StackedLSTMNet


def _make_mock_sequence_data(
    n_samples: int = 100,
    seq_len: int = 20,
    n_features: int = 10,
    is_anomaly: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """สร้าง Mock Data ในรูปแบบ 3D Time-series (⚠️ MOCK DATA)"""
    rng = np.random.RandomState(seed)
    # ข้อมูลสัญญาณปกติ (sine wave + noise)
    t = np.linspace(0, 4 * np.pi, seq_len)
    base_signal = np.sin(t)

    data = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    for i in range(n_samples):
        for f in range(n_features):
            phase = rng.uniform(0, np.pi)
            data[i, :, f] = np.sin(t + phase) + rng.normal(0, 0.1, size=seq_len)

    if is_anomaly:
        # จำลอง Anomaly: ค่าเบี่ยงเบนผิดปกติขนาดใหญ่ (Shift + Noise สูง)
        data += rng.normal(5.0, 1.5, size=data.shape).astype(np.float32)

    return data


class TestStackedLSTM:
    def test_lstm_net_shapes(self):
        """ทดสอบ Output Shapes ของ StackedLSTMNet (⚠️ MOCK DATA)"""
        batch_size = 16
        seq_len = 25
        input_dim = 12
        hidden_dim = 32
        num_layers = 2

        net = StackedLSTMNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        x_mock = torch.randn(batch_size, seq_len, input_dim)
        recon = net(x_mock)

        assert recon.shape == (batch_size, seq_len, input_dim), (
            f"Shape ผลลัพธ์ผิดพลาด: ได้ {recon.shape} คาดหวัง {(batch_size, seq_len, input_dim)}"
        )

        # ทดสอบ encode context_vector
        ctx, latent_seq = net.encode(x_mock)
        assert ctx.shape == (batch_size, hidden_dim)
        assert latent_seq.shape == (batch_size, seq_len, hidden_dim)

    def test_lstm_fit_loss_decreases(self):
        """ทดสอบว่า Loss ของ Stacked LSTM ลดลงเมื่อทำการฝึก (⚠️ MOCK DATA)"""
        X_train_mock = _make_mock_sequence_data(n_samples=120, seq_len=15, n_features=8, seed=42)
        X_val_mock = _make_mock_sequence_data(n_samples=40, seq_len=15, n_features=8, seed=43)

        detector = StackedLSTMDetector(
            input_dim=8,
            hidden_dim=24,
            num_layers=1,
            random_seed=42,
        )

        detector.fit(
            X_train_normal=X_train_mock,
            X_val_normal=X_val_mock,
            epochs=10,
            batch_size=32,
            lr=5e-3,
        )

        losses = detector.history_["train_loss"]
        assert len(losses) == 10
        assert losses[-1] < losses[0], (
            f"Loss ควรลดลง: loss แรก = {losses[0]:.4f}, loss สุดท้าย = {losses[-1]:.4f}"
        )
        assert detector.is_fitted
        assert detector.threshold_ > 0

    def test_lstm_detects_anomaly(self):
        """ทดสอบว่าโมเดลให้ Anomaly Score กับชุดผิดปกติสูงกว่าชุดปกติอย่างมีนัยสำคัญ (⚠️ MOCK DATA)"""
        X_train_mock = _make_mock_sequence_data(n_samples=100, seq_len=15, n_features=6, is_anomaly=False, seed=42)
        X_normal_test = _make_mock_sequence_data(n_samples=30, seq_len=15, n_features=6, is_anomaly=False, seed=99)
        X_anomaly_test = _make_mock_sequence_data(n_samples=30, seq_len=15, n_features=6, is_anomaly=True, seed=101)

        detector = StackedLSTMDetector(
            input_dim=6,
            hidden_dim=20,
            num_layers=1,
            threshold_pct=95.0,
            random_seed=42,
        )
        detector.fit(X_train_normal=X_train_mock, epochs=12, batch_size=32, lr=5e-3)

        normal_scores = detector.score(X_normal_test)
        anomaly_scores = detector.score(X_anomaly_test)

        mean_normal = float(np.mean(normal_scores))
        mean_anomaly = float(np.mean(anomaly_scores))

        assert mean_anomaly > mean_normal, (
            f"Score ผิดปกติควรสูงกว่า: Anomaly={mean_anomaly:.4f}, Normal={mean_normal:.4f}"
        )

        # Anomaly predictions ควรมีสัดส่วนตรวจจับความผิดปกติสูง (> 80%)
        anom_preds = detector.predict(X_anomaly_test)
        fdr = np.mean(anom_preds == 1)
        assert fdr >= 0.80, f"FDR บน mock anomaly ควรสูงกว่า 80% แต่ได้ {fdr:.2%}"

    def test_lstm_save_and_load(self, tmp_path):
        """ทดสอบการบันทึกและโหลด Checkpoint โมเดล (⚠️ MOCK DATA)"""
        X_train_mock = _make_mock_sequence_data(n_samples=60, seq_len=10, n_features=5, seed=42)

        detector = StackedLSTMDetector(input_dim=5, hidden_dim=16, num_layers=1, random_seed=42)
        detector.fit(X_train_normal=X_train_mock, epochs=5, batch_size=32)

        X_eval = _make_mock_sequence_data(n_samples=15, seq_len=10, n_features=5, seed=77)
        original_scores = detector.score(X_eval)

        save_path = tmp_path / "lstm_checkpoint.pt"
        detector.save(save_path)

        loaded_detector = StackedLSTMDetector.load(save_path)
        loaded_scores = loaded_detector.score(X_eval)

        np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)
        assert loaded_detector.threshold_ == pytest.approx(detector.threshold_)
