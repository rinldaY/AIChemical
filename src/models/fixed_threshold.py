"""
src/models/fixed_threshold.py
-----------------------------
Fixed Threshold Anomaly Detector (Rule-based Baseline)

หลักการ:
    - คำนวณค่าเฉลี่ย (mean) และส่วนเบี่ยงเบนมาตรฐาน (std) ของแต่ละฟีเจอร์จากชุดข้อมูลปกติ (Train Normal Data)
    - กำหนดขอบเขตปกติ: [mean - k*std, mean + k*std] โดยทั่วไป k = 3.0 (กฎ 3-Sigma)
    - การแจ้งเตือน: หากมีฟีเจอร์ใดฟีเจอร์หนึ่งมีค่านอกขอบเขต จะแจ้งเตือนเป็นความผิดปกติ (Anomaly)
    - Anomaly Score: ค่าสูงสุดของ absolute z-score ข้ามทุกฟีเจอร์ max_j (|x_j - mu_j| / sigma_j)
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class FixedThresholdModel:
    """
    โมเดล Fixed Threshold Baseline (Univariate 3-Sigma per feature)

    Args:
        k: จำนวนเท่าของ standard deviation (default: 3.0)
        feature_names: รายชื่อฟีเจอร์ (optional)
    """

    def __init__(self, k: float = 3.0, feature_names: Optional[List[str]] = None) -> None:
        if k <= 0:
            raise ValueError(f"k ต้องมากกว่า 0 (ได้รับ {k})")
        self.k = float(k)
        self.feature_names = feature_names
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def fit(self, X_train_normal: np.ndarray) -> "FixedThresholdModel":
        """
        เรียนรู้ขอบเขตปกติจากเฉพาะข้อมูลปกติ (Train Set)

        Args:
            X_train_normal: Array 2 มิติ (N_samples, N_features)

        Returns:
            self
        """
        X = np.asarray(X_train_normal, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X_train_normal ต้องมี 2 มิติ แต่ได้ shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X_train_normal ไม่มีข้อมูล")

        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)

        # ป้องกัน division by zero กรณีฟีเจอร์มีค่าคงที่ (zero variance)
        zero_var_mask = (self.stds_ == 0)
        if np.any(zero_var_mask):
            self._logger.warning(
                f"พบ {np.sum(zero_var_mask)} ฟีเจอร์ที่มี std=0 — จะตั้ง std เล็กน้อย (1e-6) เพื่อความเสถียร"
            )
            self.stds_[zero_var_mask] = 1e-6

        self.lower_bounds_ = self.means_ - self.k * self.stds_
        self.upper_bounds_ = self.means_ + self.k * self.stds_
        self.is_fitted = True

        self._logger.info(
            f"Fit FixedThreshold สำเร็จ (k={self.k}) บนข้อมูล {X.shape[0]:,} ตัวอย่าง ({X.shape[1]} ฟีเจอร์)"
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        คำนวณคะแนนความผิดปกติ (Anomaly Score)

        Score = max_j (|x_j - mu_j| / sigma_j)

        Args:
            X: Array 2 มิติ (N_samples, N_features)

        Returns:
            np.ndarray: Array 1 มิติ (N_samples,) ของคะแนนความผิดปกติ
        """
        if not self.is_fitted or self.means_ is None or self.stds_ is None:
            raise RuntimeError("ต้องเรียก .fit() ก่อนคำนวณคะแนน")

        X_arr = np.asarray(X, dtype=float)
        # คำนวณ absolute z-score ของทุกฟีเจอร์
        z_scores = np.abs((X_arr - self.means_) / self.stds_)
        # Anomaly score คือ z-score สูงสุดในแต่ละตัวอย่าง
        return np.max(z_scores, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ทำนายว่าแต่ละตัวอย่างเป็นความผิดปกติหรือไม่ (0=Normal, 1=Fault)

        เกณฑ์: หากมีฟีเจอร์ใดฟีเจอร์หนึ่งหลุดขอบเขต (score > k) จะทำนายเป็น 1

        Args:
            X: Array 2 มิติ (N_samples, N_features)

        Returns:
            np.ndarray: Array 1 มิติของผลทำนาย (0 หรือ 1)
        """
        scores = self.score(X)
        return (scores > self.k).astype(int)

    def tune_k_on_val(self, X_val: np.ndarray, target_far: float = 0.05) -> float:
        """
        ปรับจูนค่า k บนชุด Validation ของข้อมูลปกติ เพื่อให้ได้ False Alarm Rate (FAR) ตามที่ต้องการ

        Args:
            X_val: ข้อมูลปกติในชุด Validation (y_val == 0)
            target_far: เป้าหมาย FAR ที่ยอมรับได้ (default: 0.05 หรือ 5%)

        Returns:
            float: ค่า k ที่ปรับจูนแล้ว
        """
        scores = self.score(X_val)
        # คำนวณ percentile ที่สอดคล้องกับ (1 - target_far)
        new_k = float(np.percentile(scores, (1.0 - target_far) * 100))
        self.k = new_k
        self.lower_bounds_ = self.means_ - self.k * self.stds_
        self.upper_bounds_ = self.means_ + self.k * self.stds_
        self._logger.info(f"ปรับจูนค่า k สำเร็จ: k = {self.k:.4f} (เป้าหมาย FAR = {target_far*100:.1f}%)")
        return self.k
