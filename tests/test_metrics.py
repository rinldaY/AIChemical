"""
tests/test_metrics.py
---------------------
Unit tests สำหรับโมดูลการประเมินผลการตรวจจับความผิดปกติ (src/evaluation/metrics.py)

⚠️ IMPORTANT: ข้อมูลทั้งหมดในไฟล์นี้เป็น MOCK DATA ที่สร้างขึ้นเพื่อทดสอบความถูกต้อง
   ของการคำนวณทางคณิตศาสตร์สถิติเท่านั้น ไม่ใช่ข้อมูล TEP จริง

Tests:
    - test_calculate_fdr:             ตรวจสอบสูตร FDR = TP / (TP + FN)
    - test_calculate_far:             ตรวจสอบสูตร FAR = FP / (FP + TN)
    - test_division_by_zero_safety:   กรณีไม่มี fault หรือไม่มี normal ต้องไม่ crash
    - test_calculate_detection_delay: ตรวจสอบการนับเวลาหน่วง (samples) จาก fault onset
    - test_calculate_auroc:           ตรวจสอบค่า AUROC เทียบกับ ground truth
    - test_evaluate_predictions_all:  ตรวจสอบ dictionary ผลลัพธ์ครบถ้วน
"""

from pathlib import Path
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    calculate_auroc,
    calculate_detection_delay,
    calculate_far,
    calculate_fdr,
    evaluate_predictions,
)


class TestMetricsCalculation:
    """ทดสอบความแม่นยำของเกณฑ์วัดการตรวจจับ"""

    def test_calculate_fdr(self):
        """FDR = TP / (TP + FN)"""
        # MOCK DATA: 10 faults, ทำนายพบ 8 ตัว
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        # faults = 10, detected = 8 -> FDR = 8/10 = 0.8
        fdr = calculate_fdr(y_true, y_pred)
        assert np.isclose(fdr, 0.8), f"FDR ต้องเป็น 0.8 แต่ได้ {fdr}"

    def test_calculate_far(self):
        """FAR = FP / (FP + TN)"""
        # MOCK DATA: 10 normals, แจ้งเตือนผิด 1 ตัว
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        # normals = 10, false alarms = 1 -> FAR = 1/10 = 0.1
        far = calculate_far(y_true, y_pred)
        assert np.isclose(far, 0.1), f"FAR ต้องเป็น 0.1 แต่ได้ {far}"

    def test_division_by_zero_safety(self):
        """กรณีไม่มี fault หรือไม่มี normal ต้องคืนค่า 0.0 อย่างปลอดภัย"""
        # MOCK DATA: normal ล้วน
        all_normal = np.zeros(20, dtype=int)
        assert calculate_fdr(all_normal, all_normal) == 0.0

        # MOCK DATA: fault ล้วน
        all_fault = np.ones(20, dtype=int)
        assert calculate_far(all_fault, all_fault) == 0.0

    def test_calculate_detection_delay(self):
        """Detection delay คำนวณความหน่วงเฉลี่ยหลัง fault onset"""
        # MOCK DATA: 2 runs แต่ละ run มี 10 samples
        # Run 1: fault onset ที่ sample index 4, ตรวจพบที่ sample index 6 -> delay = 2
        # Run 2: fault onset ที่ sample index 4, ตรวจพบที่ sample index 8 -> delay = 4
        # Delay เฉลี่ย = (2 + 4) / 2 = 3.0 samples
        run_ids = np.array([1]*10 + [2]*10)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 2)

        y_pred = np.zeros(20, dtype=int)
        y_pred[6] = 1   # run 1: onset ที่ index 4, first detected ที่ index 6 -> delay = 2
        y_pred[18] = 1  # run 2: onset ที่ index 14 (sample 4), first detected ที่ 18 (sample 8) -> delay = 4

        delay = calculate_detection_delay(y_true, y_pred, run_ids, fault_onset_sample=4)
        assert delay is not None
        assert np.isclose(delay, 3.0), f"Detection delay ต้องเป็น 3.0 แต่ได้ {delay}"

    def test_calculate_auroc(self):
        """AUROC สำหรับ anomaly scores ที่แยกคลาสได้อย่างสมบูรณ์แบบ"""
        # MOCK DATA: normal ได้ score ต่ำ, fault ได้ score สูง
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9, 0.85, 0.95])
        auc = calculate_auroc(y_true, scores)
        assert auc is not None
        assert np.isclose(auc, 1.0), f"AUROC สมบูรณ์แบบต้องเป็น 1.0 แต่ได้ {auc}"

    def test_evaluate_predictions_all(self):
        """evaluate_predictions ต้องคืนโครงสร้าง dictionary และ confusion matrix ครบถ้วน"""
        # MOCK DATA
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.8, 0.3, 0.9])

        res = evaluate_predictions(y_true, y_pred, anomaly_scores=scores)
        assert "fdr" in res
        assert "far" in res
        assert "auroc" in res
        assert res["confusion_matrix"]["tp"] == 1
        assert res["confusion_matrix"]["fp"] == 1
        assert res["confusion_matrix"]["tn"] == 1
        assert res["confusion_matrix"]["fn"] == 1
