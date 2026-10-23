"""
src/evaluation/metrics.py
-------------------------
ฟังก์ชันและคลาสสำหรับคำนวณเกณฑ์วัดประสิทธิภาพการตรวจจับความผิดปกติ (TEP Anomaly Detection Benchmark)

ข้อกำหนดสำคัญ:
    - ห้าม hardcode ค่า FDR, FAR หรือ Detection Delay
    - ค่าทุกค่าต้องคำนวณจาก ground truth (y_true) และผลทำนาย (y_pred / anomaly_scores) จริง
    - ป้องกัน Division by zero อย่างปลอดภัย

Metrics:
    - Fault Detection Rate (FDR / Recall) = TP / (TP + FN)
    - False Alarm Rate (FAR / Fall-out)   = FP / (FP + TN)
    - Detection Delay                     = เวลาเฉลี่ย (samples) จาก fault onset จนถึง sample แรกที่ตรวจพบ
    - AUROC (Area under ROC curve)        = คำนวณจาก anomaly scores แบบต่อเนื่อง
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def calculate_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    คำนวณ Fault Detection Rate (FDR) หรือ True Positive Rate (Recall)

    FDR = TP / (TP + FN) บนกลุ่มตัวอย่างที่เป็น Fault (y_true == 1)

    Args:
        y_true: Ground truth binary labels (0=Normal, 1=Fault)
        y_pred: Predicted binary labels (0=Normal, 1=Fault)

    Returns:
        float: สัดส่วน FDR อยู่ระหว่าง 0.0 ถึง 1.0 (หากไม่มี fault ใน y_true จะคืน 0.0)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    fault_mask = (y_true_arr == 1)
    total_faults = int(np.sum(fault_mask))

    if total_faults == 0:
        logger.warning("ไม่มีตัวอย่างที่เป็น Fault (y_true==1) ในชุดข้อมูล — FDR คืนค่า 0.0")
        return 0.0

    detected_faults = int(np.sum((y_pred_arr == 1) & fault_mask))
    return float(detected_faults / total_faults)


def calculate_far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    คำนวณ False Alarm Rate (FAR) หรือ False Positive Rate

    FAR = FP / (FP + TN) บนกลุ่มตัวอย่างที่เป็น Normal (y_true == 0)

    Args:
        y_true: Ground truth binary labels (0=Normal, 1=Fault)
        y_pred: Predicted binary labels (0=Normal, 1=Fault)

    Returns:
        float: สัดส่วน FAR อยู่ระหว่าง 0.0 ถึง 1.0 (หากไม่มี normal ใน y_true จะคืน 0.0)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    normal_mask = (y_true_arr == 0)
    total_normals = int(np.sum(normal_mask))

    if total_normals == 0:
        logger.warning("ไม่มีตัวอย่างที่เป็น Normal (y_true==0) ในชุดข้อมูล — FAR คืนค่า 0.0")
        return 0.0

    false_alarms = int(np.sum((y_pred_arr == 1) & normal_mask))
    return float(false_alarms / total_normals)


def calculate_detection_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_ids: np.ndarray,
    fault_onset_sample: int = 160,
) -> Optional[float]:
    """
    คำนวณ Detection Delay เฉลี่ย (หน่วยเป็น samples)

    สำหรับแต่ละ simulationRun ที่มี fault:
    Detection Delay = (Sample index แรกที่ y_pred == 1 หลังจาก fault onset) - fault_onset_sample
    หาก run นั้นไม่สามารถตรวจจับ fault ได้เลย จะไม่นำมารวมในค่าเฉลี่ยและบันทึกเป็น undetected run

    Args:
        y_true:             Ground truth binary labels (0=Normal, 1=Fault)
        y_pred:             Predicted binary labels (0=Normal, 1=Fault)
        run_ids:            Array ระบุ simulationRun ของแต่ละตัวอย่าง
        fault_onset_sample: Index ของ sample ที่ fault เริ่มต้น (default: 160 ตาม TEP testing benchmark)

    Returns:
        float: ค่าเฉลี่ยของ detection delay ในหน่วย samples (หรือ None หากไม่มี run ใดตรวจพบ)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    runs_arr = np.asarray(run_ids, dtype=int)

    delays: List[int] = []
    unique_runs = np.unique(runs_arr)

    for r in unique_runs:
        r_mask = (runs_arr == r)
        r_y_true = y_true_arr[r_mask]
        r_y_pred = y_pred_arr[r_mask]

        # ตรวจสอบว่า run นี้มี fault เกิดขึ้นจริงหรือไม่
        if not np.any(r_y_true == 1):
            continue

        # ค้นหาจุดที่ fault เกิดขึ้นจริง (index ที่ y_true == 1 ตัวแรก)
        fault_indices = np.where(r_y_true == 1)[0]
        actual_onset = int(fault_indices[0])

        # ตรวจสอบว่ามีการตรวจจับ (y_pred == 1) หลัง actual_onset หรือไม่
        detected_indices = np.where((r_y_pred == 1) & (np.arange(len(r_y_pred)) >= actual_onset))[0]

        if len(detected_indices) > 0:
            first_detected = int(detected_indices[0])
            delay = first_detected - actual_onset
            delays.append(delay)

    if not delays:
        return None

    return float(np.mean(delays))


def calculate_auroc(y_true: np.ndarray, anomaly_scores: np.ndarray) -> Optional[float]:
    """
    คำนวณ Area Under ROC Curve (AUROC)

    Args:
        y_true:         Ground truth binary labels (0=Normal, 1=Fault)
        anomaly_scores: คะแนนความผิดปกติแบบต่อเนื่อง (ยิ่งสูง ยิ่งผิดปกติ)

    Returns:
        float: ค่า AUROC ระหว่าง 0.0 ถึง 1.0 (คืน None หากในชุดข้อมูลมีเพียงคลาสเดียว)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    scores_arr = np.asarray(anomaly_scores, dtype=float)

    # AUROC ต้องการตัวอย่างทั้งคลาส 0 และ 1
    if len(np.unique(y_true_arr)) < 2:
        logger.warning("มีเพียงคลาสเดียวใน y_true — ไม่สามารถคำนวณ AUROC ได้")
        return None

    # จัดการกรณีมี NaN หรือ Inf ใน scores
    valid_mask = np.isfinite(scores_arr)
    if not np.all(valid_mask):
        scores_arr = np.nan_to_num(scores_arr, nan=0.0, posinf=1e6, neginf=-1e6)

    return float(roc_auc_score(y_true_arr, scores_arr))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None,
    run_ids: Optional[np.ndarray] = None,
    fault_onset_sample: int = 160,
) -> Dict[str, Any]:
    """
    สรุปผลการประเมินการตรวจจับความผิดปกติแบบครบวงจร

    Args:
        y_true:             Ground truth binary labels
        y_pred:             Predicted binary labels
        anomaly_scores:     คะแนนความผิดปกติแบบต่อเนื่อง (optional)
        run_ids:            Simulation run IDs (optional)
        fault_onset_sample: จุดเริ่มต้นของ fault (optional)

    Returns:
        Dict: สรุปผล FDR, FAR, AUROC, Detection Delay, จำนวน samples
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    fdr = calculate_fdr(y_true_arr, y_pred_arr)
    far = calculate_far(y_true_arr, y_pred_arr)

    auroc = None
    if anomaly_scores is not None:
        auroc = calculate_auroc(y_true_arr, np.asarray(anomaly_scores, dtype=float))

    delay = None
    if run_ids is not None:
        delay = calculate_detection_delay(y_true_arr, y_pred_arr, np.asarray(run_ids, dtype=int), fault_onset_sample)

    tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    tn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))

    results = {
        "fdr": fdr,
        "far": far,
        "auroc": auroc,
        "detection_delay": delay,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total_samples": len(y_true_arr),
        "total_fault_samples": int(np.sum(y_true_arr == 1)),
        "total_normal_samples": int(np.sum(y_true_arr == 0)),
    }
    return results
