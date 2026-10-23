"""
tests/test_autoencoder.py
-------------------------
Unit tests สำหรับโมเดล Deep Autoencoder (src/models/autoencoder.py)

⚠️ IMPORTANT: ข้อมูลทั้งหมดในไฟล์นี้เป็น MOCK DATA ที่สร้างขึ้นเพื่อทดสอบความถูกต้อง
   ทางสถาปัตยกรรมและการคำนวณของโครงข่ายประสาทเทียมเท่านั้น ไม่ใช่ข้อมูล TEP จริง

Tests:
    - test_autoencoder_net_shapes:       ตรวจสอบมิติของ Encoder, Latent, และ Decoder
    - test_autoencoder_fit_loss_decreases: ตรวจสอบว่าการฝึกทำให้ reconstruction loss ลดลง
    - test_autoencoder_detects_anomaly:  Reconstruction error ของ anomaly ต้องสูงกว่า normal
    - test_autoencoder_save_and_load:    บันทึก checkpoint และโหลดกลับมาต้องให้ผลลัพธ์ตรงกัน
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import หากมี torch ติดตั้ง
torch = pytest.importorskip("torch")

from src.models.autoencoder import AutoencoderDetector, AutoencoderNet


def _make_mock_normal_data(n_samples: int = 300, n_features: int = 10, seed: int = 42) -> np.ndarray:
    """สร้างข้อมูลปกติจำลอง (MOCK DATA)"""
    rng = np.random.default_rng(seed)
    # ฟีเจอร์มีความสัมพันธ์เชิงเส้นบางส่วน เพื่อให้ Autoencoder บีบอัดได้
    t = rng.normal(0, 1, (n_samples, 2))
    W = rng.normal(0, 1, (2, n_features))
    noise = rng.normal(0, 0.05, (n_samples, n_features))
    return np.dot(t, W) + noise


def _make_mock_anomaly_data(n_samples: int = 50, n_features: int = 10, seed: int = 99) -> np.ndarray:
    """สร้างข้อมูลผิดปกติจำลอง (MOCK DATA: Out-of-distribution noise)"""
    rng = np.random.default_rng(seed)
    return rng.uniform(-5.0, 5.0, (n_samples, n_features))


class TestAutoencoder:
    """ทดสอบโมเดล Autoencoder"""

    def test_autoencoder_net_shapes(self):
        """ตรวจสอบมิติ Input, Latent Bottleneck, และ Output Reconstruction"""
        batch_size = 16
        input_dim = 10
        latent_dim = 3
        net = AutoencoderNet(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[8, 4])

        # MOCK DATA tensor
        x = torch.randn(batch_size, input_dim)
        z = net.encode(x)
        assert z.shape == (batch_size, latent_dim), f"Latent shape ต้องเป็น {(batch_size, latent_dim)} แต่ได้ {z.shape}"

        x_hat = net.decode(z)
        assert x_hat.shape == (batch_size, input_dim), f"Output shape ต้องเป็น {(batch_size, input_dim)} แต่ได้ {x_hat.shape}"

        # Forward pass รวม
        x_rec = net(x)
        assert x_rec.shape == (batch_size, input_dim)

    def test_autoencoder_fit_loss_decreases(self):
        """การฝึก Autoencoder ต้องทำให้ training loss ลดลง"""
        X_train = _make_mock_normal_data(n_samples=200, n_features=8, seed=1)

        detector = AutoencoderDetector(input_dim=8, latent_dim=2, hidden_dims=[6, 4], random_seed=42)
        detector.fit(X_train, epochs=15, batch_size=32, lr=1e-2, device="cpu")

        assert len(detector.history_["train_loss"]) == 15
        initial_loss = detector.history_["train_loss"][0]
        final_loss = detector.history_["train_loss"][-1]

        assert final_loss < initial_loss, f"Final loss ({final_loss}) ต้องต่ำกว่า initial loss ({initial_loss})"

    def test_autoencoder_detects_anomaly(self):
        """Reconstruction error ของข้อมูล anomaly ต้องสูงกว่าข้อมูล normal"""
        X_train = _make_mock_normal_data(n_samples=300, n_features=10, seed=10)
        X_test_normal = _make_mock_normal_data(n_samples=50, n_features=10, seed=11)
        X_test_anomaly = _make_mock_anomaly_data(n_samples=50, n_features=10, seed=12)

        detector = AutoencoderDetector(input_dim=10, latent_dim=3, hidden_dims=[8, 5], threshold_pct=90.0, random_seed=42)
        detector.fit(X_train, epochs=20, batch_size=32, lr=1e-2, device="cpu")

        score_normal = detector.score(X_test_normal)
        score_anomaly = detector.score(X_test_anomaly)

        assert np.mean(score_anomaly) > np.mean(score_normal) * 2.0, \
            "คะแนนความผิดปกติของ anomaly data ต้องสูงกว่า normal data อย่างชัดเจน"

        # การทำนายต้องจับ anomaly ได้มากกว่า normal
        pred_anomaly = detector.predict(X_test_anomaly)
        fdr = np.mean(pred_anomaly == 1)
        assert fdr >= 0.70, f"FDR ({fdr}) ของ Autoencoder ควรสูงบน anomaly ชัดเจน"

    def test_autoencoder_save_and_load(self, tmp_path):
        """บันทึกและโหลด Autoencoder checkpoint ต้องให้คะแนนตรงกัน 100%"""
        X_train = _make_mock_normal_data(n_samples=100, n_features=6, seed=20)
        X_test = _make_mock_normal_data(n_samples=20, n_features=6, seed=21)

        detector = AutoencoderDetector(input_dim=6, latent_dim=2, hidden_dims=[4], random_seed=42)
        detector.fit(X_train, epochs=5, batch_size=16, device="cpu")

        orig_scores = detector.score(X_test)
        save_path = tmp_path / "ae_checkpoint.pt"
        detector.save(save_path)

        loaded_detector = AutoencoderDetector.load(save_path, device="cpu")
        loaded_scores = loaded_detector.score(X_test)

        assert np.allclose(orig_scores, loaded_scores, atol=1e-5), \
            "คะแนนหลังโหลด checkpoint ต้องตรงกับก่อนบันทึก"
        assert loaded_detector.threshold_ == detector.threshold_
