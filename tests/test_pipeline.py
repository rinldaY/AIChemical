"""
tests/test_pipeline.py
----------------------
Unit tests สำหรับ Preprocessing Pipeline (TEPPreprocessor และ WindowGenerator)

⚠️ IMPORTANT: ข้อมูลทั้งหมดในไฟล์นี้เป็น MOCK DATA ที่สร้างขึ้นเพื่อทดสอบเท่านั้น
   ไม่มีส่วนใดเป็นข้อมูล TEP จริง และห้ามนำผลทดสอบไปอ้างอิงในการทดลองจริง

Tests:
    - test_scaler_fits_on_train_only:       ตรวจสอบว่า scaler mean/std มาจาก train เท่านั้น
    - test_scaler_transform_before_fit:     ต้อง raise RuntimeError เมื่อ transform ก่อน fit
    - test_scaler_save_and_load:            บันทึกและโหลด scaler แล้วต้องให้ผลลัพธ์เดิม
    - test_window_never_crosses_runs:       หน้าต่างเวลาต้องไม่ปนเปื้อนข้าม simulationRun
    - test_window_output_shape:             ตรวจสอบมิติของ WindowedDataset (3D array)
    - test_window_label_assignment:         ตรวจสอบ label ของ window ตาม sample สุดท้าย
    - test_window_handles_short_runs:       run ที่สั้นกว่า window_size ต้องถูกข้ามอย่างปลอดภัย
    - test_window_stride:                   ตรวจสอบ stride ในการเลื่อนหน้าต่างเวลา
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pipeline import TEPPreprocessor, WindowGenerator, WindowedDataset


# ---------------------------------------------------------------------------
# Helpers for MOCK DATA
# ⚠️ MOCK DATA — ค่าทั้งหมดเป็นข้อมูลจำลองเพื่อการทดสอบทางซอฟต์แวร์เท่านั้น
# ---------------------------------------------------------------------------

def _make_mock_features(n_samples: int = 100, n_features: int = 5, mean: float = 10.0, std: float = 2.0) -> np.ndarray:
    """สร้าง feature matrix จำลอง (MOCK DATA)"""
    rng = np.random.default_rng(42)
    return rng.normal(mean, std, (n_samples, n_features))


# ---------------------------------------------------------------------------
# Tests — TEPPreprocessor
# ---------------------------------------------------------------------------

class TestTEPPreprocessor:
    """ทดสอบการทำงานของ TEPPreprocessor และการป้องกัน Data Leakage"""

    def test_scaler_fits_on_train_only(self):
        """Scaler ต้องคำนวณสถิติจากชุด Train เท่านั้น ไม่ถูกกระทบโดยชุดอื่น"""
        # MOCK DATA
        train_normal = _make_mock_features(n_samples=500, mean=5.0, std=1.0)
        test_data = _make_mock_features(n_samples=200, mean=100.0, std=50.0)

        preprocessor = TEPPreprocessor()
        preprocessor.fit(train_normal)

        # mean ของ scaler ต้องใกล้เคียงกับ mean ของ train_normal (~5.0) ไม่ใช่ ~100.0
        assert np.allclose(preprocessor.scaler.mean_, 5.0, atol=0.2), \
            f"Scaler mean ({preprocessor.scaler.mean_}) ต้องคำนวณจาก train set เท่านั้น"

        # transform ทั้งสองชุดโดยใช้พารามิเตอร์ของ train
        train_scaled = preprocessor.transform(train_normal)
        test_scaled = preprocessor.transform(test_data)

        # train_scaled ต้องมี mean ~0 และ std ~1
        assert np.allclose(np.mean(train_scaled, axis=0), 0.0, atol=0.1)
        assert np.allclose(np.std(train_scaled, axis=0), 1.0, atol=0.1)

        # test_scaled ต้องไม่เป็น 0 (เพราะ distribution ต่างกัน)
        assert not np.allclose(np.mean(test_scaled, axis=0), 0.0, atol=1.0)

    def test_scaler_transform_before_fit(self):
        """เรียก transform ก่อน fit ต้อง raise RuntimeError"""
        preprocessor = TEPPreprocessor()
        mock_data = _make_mock_features(n_samples=10)
        with pytest.raises(RuntimeError, match="fit"):
            preprocessor.transform(mock_data)

    def test_scaler_save_and_load(self, tmp_path):
        """บันทึก scaler ลงไฟล์และโหลดกลับมาต้องให้ค่า transform เท่ากัน"""
        train_data = _make_mock_features(n_samples=100, mean=3.0, std=2.0)
        test_data = _make_mock_features(n_samples=20, mean=4.0, std=2.0)

        preprocessor = TEPPreprocessor(feature_names=["f1", "f2", "f3", "f4", "f5"])
        preprocessor.fit(train_data)
        original_transformed = preprocessor.transform(test_data)

        save_file = tmp_path / "scaler.joblib"
        preprocessor.save(save_file)

        loaded_preprocessor = TEPPreprocessor.load(save_file)
        loaded_transformed = loaded_preprocessor.transform(test_data)

        assert np.allclose(original_transformed, loaded_transformed)
        assert loaded_preprocessor.feature_names == ["f1", "f2", "f3", "f4", "f5"]


# ---------------------------------------------------------------------------
# Tests — WindowGenerator
# ---------------------------------------------------------------------------

class TestWindowGenerator:
    """ทดสอบความปลอดภัยของการสร้าง Sliding Windows (ห้ามข้าม run)"""

    def test_window_never_crosses_runs(self):
        """Window ต้องไม่นำข้อมูลจากคนละ simulationRun มาปนกันเด็ดขาด"""
        # MOCK DATA: 2 runs แต่ละ run มี 10 samples
        run_ids = np.array([1]*10 + [2]*10)
        # ให้ค่าของ run 1 เป็น 100 ทั้งหมด, run 2 เป็น 200 ทั้งหมด
        features = np.zeros((20, 2))
        features[:10] = 100.0
        features[10:] = 200.0
        labels = np.zeros(20, dtype=int)

        generator = WindowGenerator(window_size=5, stride=1)
        dataset = generator.create_windows(features, labels, run_ids)

        assert len(dataset.X) > 0

        # ตรวจสอบว่าทุก window มีค่าเฉพาะ 100 ล้วน หรือ 200 ล้วน ไม่มี window ใดที่มีทั้ง 100 และ 200 ปนกัน
        for i, window in enumerate(dataset.X):
            unique_vals = np.unique(window)
            assert len(unique_vals) == 1, \
                f"Window ที่ {i} มีค่าปนเปื้อนข้าม run: {unique_vals}"
            assert dataset.run_ids[i] in (1, 2)

    def test_window_output_shape(self):
        """WindowedDataset ต้องมีมิติ (N_windows, window_size, N_features)"""
        # MOCK DATA: 1 run มี 50 samples, 4 features
        features = _make_mock_features(n_samples=50, n_features=4)
        labels = np.zeros(50, dtype=int)
        run_ids = np.ones(50, dtype=int)

        generator = WindowGenerator(window_size=10, stride=2)
        dataset = generator.create_windows(features, labels, run_ids)

        # จำนวน window ที่คาดหมาย = (50 - 10) // 2 + 1 = 21 windows
        expected_windows = (50 - 10) // 2 + 1
        assert dataset.X.shape == (expected_windows, 10, 4)
        assert len(dataset.y) == expected_windows
        assert len(dataset.run_ids) == expected_windows

    def test_window_label_assignment(self):
        """Label ของ window ต้องตรงกับ label ของ sample สุดท้ายใน window"""
        # MOCK DATA: 1 run มี 10 samples โดยที่ sample ที่ 7 เป็นต้นไปเป็น fault (1)
        features = np.zeros((10, 2))
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
        run_ids = np.ones(10, dtype=int)

        generator = WindowGenerator(window_size=3, stride=1)
        dataset = generator.create_windows(features, labels, run_ids)

        # window 0: samples 0,1,2 -> end index 2 -> label 0
        # window 4: samples 4,5,6 -> end index 6 -> label 0
        # window 5: samples 5,6,7 -> end index 7 -> label 1
        # window 7: samples 7,8,9 -> end index 9 -> label 1
        assert dataset.y[0] == 0
        assert dataset.y[4] == 0
        assert dataset.y[5] == 1
        assert dataset.y[7] == 1

    def test_window_handles_short_runs(self):
        """Run ที่สั้นกว่า window_size ต้องถูกข้ามโดยไม่ error"""
        # MOCK DATA: run 1 มี 3 samples (สั้นกว่า window_size=5), run 2 มี 8 samples
        features = np.zeros((11, 2))
        labels = np.zeros(11, dtype=int)
        run_ids = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])

        generator = WindowGenerator(window_size=5, stride=1)
        dataset = generator.create_windows(features, labels, run_ids)

        # ต้องมี window เฉพาะจาก run 2 เท่านั้น (8 - 5 + 1 = 4 windows)
        assert len(dataset.X) == 4
        assert np.all(dataset.run_ids == 2)
