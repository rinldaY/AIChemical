"""
tests/test_baseline.py
----------------------
Unit tests สำหรับโมเดล Baseline (FixedThresholdModel และ PCASPCModel)

⚠️ IMPORTANT: ข้อมูลทั้งหมดในไฟล์นี้เป็น MOCK DATA ที่สร้างขึ้นเพื่อทดสอบความถูกต้อง
   ของการตรวจจับเชิงสถิติเท่านั้น ไม่ใช่ข้อมูล TEP จริง

Tests:
    - FixedThreshold:
      - test_fixed_threshold_fit_and_predict
      - test_fixed_threshold_score
      - test_fixed_threshold_tune_k
    - PCASPC:
      - test_pca_spc_fit_and_limits
      - test_pca_spc_detect_anomaly
      - test_pca_spc_decision_modes
      - test_pca_spc_tune_limits
"""

from pathlib import Path
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fixed_threshold import FixedThresholdModel
from src.models.pca_spc import PCASPCModel


# ---------------------------------------------------------------------------
# MOCK DATA helpers
# ---------------------------------------------------------------------------

def _make_mock_normal_data(n_samples: int = 500, n_features: int = 10, seed: int = 42) -> np.ndarray:
    """สร้างข้อมูลปกติจำลอง (MOCK DATA: Multivariate Gaussian จาก distribution เดียวกัน)"""
    # กำหนด covariance matrix คงที่สำหรับ process เดียวกัน
    rng_cov = np.random.default_rng(100)
    A = rng_cov.normal(0, 1, (n_features, n_features))
    cov = np.dot(A, A.T) + np.eye(n_features) * 0.1

    # สุ่มข้อมูลตัวอย่างตาม seed
    rng_data = np.random.default_rng(seed)
    mean = np.zeros(n_features)
    return rng_data.multivariate_normal(mean, cov, n_samples)


def _make_mock_fault_data(n_samples: int = 100, n_features: int = 10, shift: float = 5.0, seed: int = 99) -> np.ndarray:
    """สร้างข้อมูลผิดปกติจำลอง (MOCK DATA: Mean shift)"""
    normal = _make_mock_normal_data(n_samples, n_features, seed)
    return normal + shift


# ---------------------------------------------------------------------------
# Tests — FixedThresholdModel
# ---------------------------------------------------------------------------

class TestFixedThresholdModel:
    """ทดสอบ FixedThresholdModel Baseline"""

    def test_fixed_threshold_fit_and_predict(self):
        """ข้อมูลปกติส่วนใหญ่ต้องได้ 0 และข้อมูลที่มี shift รุนแรงต้องได้ 1"""
        X_train = _make_mock_normal_data(500, 5, seed=1)
        X_normal_test = _make_mock_normal_data(100, 5, seed=2)
        X_fault_test = _make_mock_fault_data(100, 5, shift=10.0, seed=3)

        model = FixedThresholdModel(k=3.0)
        model.fit(X_train)

        pred_normal = model.predict(X_normal_test)
        pred_fault = model.predict(X_fault_test)

        # ข้อมูลปกติควรมี false alarm น้อยมาก (กฎ 3-sigma ประมาณ < 5%)
        far = np.mean(pred_normal == 1)
        assert far < 0.1, f"FAR ({far}) บนข้อมูลปกติต้องน้อยกว่า 0.1"

        # ข้อมูล fault ที่ shift 10 sigma ควรตรวจพบเกือบ 100%
        fdr = np.mean(pred_fault == 1)
        assert fdr > 0.95, f"FDR ({fdr}) บน fault รุนแรงต้องมากกว่า 0.95"

    def test_fixed_threshold_score(self):
        """Anomaly score ต้องสอดคล้องกับค่าสูงสุดของ absolute z-score"""
        X_train = _make_mock_normal_data(200, 3, seed=1)
        model = FixedThresholdModel(k=3.0)
        model.fit(X_train)

        # ตัวอย่างทดสอบที่เจตนาให้ฟีเจอร์แรกมี z-score สูง
        sample = model.means_ + model.stds_ * 4.5
        sample = sample.reshape(1, -1)

        score = model.score(sample)
        assert np.isclose(score[0], 4.5, atol=0.1)

    def test_fixed_threshold_tune_k(self):
        """ปรับจูน k บน Val data ต้องให้ FAR เข้าใกล้เป้าหมาย"""
        X_train = _make_mock_normal_data(1000, 5, seed=10)
        X_val_normal = _make_mock_normal_data(1000, 5, seed=11)

        model = FixedThresholdModel(k=3.0)
        model.fit(X_train)

        tuned_k = model.tune_k_on_val(X_val_normal, target_far=0.05)
        # ตรวจสอบว่าหลังจากจูนแล้ว FAR บน val set อยู่ใกล้เคียง 5%
        pred_val = model.predict(X_val_normal)
        actual_far = np.mean(pred_val == 1)
        assert np.isclose(actual_far, 0.05, atol=0.02)


# ---------------------------------------------------------------------------
# Tests — PCASPCModel
# ---------------------------------------------------------------------------

class TestPCASPCModel:
    """ทดสอบ PCASPCModel (Hotelling's T² + SPE)"""

    def test_pca_spc_fit_and_limits(self):
        """Fit บนข้อมูลปกติและคำนวณ T²_limit และ SPE_limit > 0"""
        X_train = _make_mock_normal_data(500, 10, seed=42)

        model = PCASPCModel(n_components=0.85, alpha=0.05)
        model.fit(X_train)

        assert model.is_fitted
        assert model.n_components_ > 0
        assert model.t2_limit_ > 0.0
        assert model.spe_limit_ > 0.0

    def test_pca_spc_detect_anomaly(self):
        """PCA-SPC ต้องตรวจจับความผิดปกติที่ shift หลุดจาก correlation เดิมได้"""
        X_train = _make_mock_normal_data(500, 10, seed=1)
        X_normal = _make_mock_normal_data(100, 10, seed=2)
        X_fault = _make_mock_fault_data(100, 10, shift=6.0, seed=3)

        model = PCASPCModel(n_components=0.85, alpha=0.05)
        model.fit(X_train)

        pred_normal = model.predict(X_normal, decision="combined")
        pred_fault = model.predict(X_fault, decision="combined")

        # FAR ต้องต่ำ (< 15% ตาม confidence limit)
        far = np.mean(pred_normal == 1)
        assert far < 0.15, f"FAR ({far}) สูงเกินไป"

        # FDR ต้องสูง (> 90%)
        fdr = np.mean(pred_fault == 1)
        assert fdr > 0.90, f"FDR ({fdr}) ต่ำเกินไปสำหรับ fault รุนแรง"

    def test_pca_spc_decision_modes(self):
        """ทดสอบ decision modes: combined, t2_only, spe_only"""
        X_train = _make_mock_normal_data(300, 8, seed=4)
        X_test = _make_mock_normal_data(50, 8, seed=5)

        model = PCASPCModel(n_components=4, alpha=0.05)
        model.fit(X_train)

        pred_combined = model.predict(X_test, decision="combined")
        pred_t2 = model.predict(X_test, decision="t2_only")
        pred_spe = model.predict(X_test, decision="spe_only")

        # combined ต้องเป็น union ของ t2 และ spe
        assert np.array_equal(pred_combined, (pred_t2 | pred_spe))

    def test_pca_spc_tune_limits(self):
        """ปรับจูน limits บน Val normal data ให้เข้าใกล้เป้าหมาย FAR"""
        X_train = _make_mock_normal_data(500, 6, seed=20)
        X_val = _make_mock_normal_data(500, 6, seed=21)

        model = PCASPCModel(n_components=0.90, alpha=0.05)
        model.fit(X_train)

        t2_lim, spe_lim = model.tune_limits_on_val(X_val, target_far=0.05)
        assert t2_lim > 0.0
        assert spe_lim > 0.0

        # ตรวจสอบ FAR หลังจูน
        pred_val = model.predict(X_val, decision="combined")
        far = np.mean(pred_val == 1)
        assert far < 0.10, f"FAR หลังจูน ({far}) ควรใกล้เคียงเป้าหมาย 0.05"
