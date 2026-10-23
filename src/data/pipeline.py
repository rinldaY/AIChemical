"""
src/data/pipeline.py
--------------------
Preprocessing Pipeline สำหรับชุดข้อมูล Tennessee Eastman Process (TEP)

หลักการสำคัญ:
    1. ป้องกัน Data Leakage:
       - Fit Scaler จากเฉพาะชุดข้อมูลปกติในการฝึก (Train Normal data) เท่านั้น
       - ห้ามใช้สถิติจาก Validation หรือ Test set ในการปรับสเกล
    2. แบ่งข้อมูลตาม Simulation Run:
       - แบ่งข้อมูลระดับ run ID เพื่อไม่ให้ลำดับเวลาของ run เดียวกันปะปนข้าม split
    3. การสร้าง Sliding Window สำหรับ Sequential Models (LSTM / T²-LSTM):
       - Window ต้องอยู่ภายใน simulationRun เดียวกัน ห้ามคาบเกี่ยวข้าม run
       - ห้ามมีการสุ่มแบ่งหรือซ้อนทับข้ามชุด Train, Validation และ Test

Classes:
    TEPPreprocessor  — จัดการการปรับสเกล (StandardScaler) และการแปลงข้อมูล
    WindowGenerator  — สร้าง Sliding Windows ตาม run boundaries อย่างปลอดภัย
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class WindowedDataset:
    """
    ชุดข้อมูลที่ถูกจัดเตรียมในรูปแบบ Sliding Window

    Attributes:
        X:             Array 3 มิติ (N_windows, window_size, N_features)
        y:             Label 1 มิติ (0=normal, 1=anomaly) โดยกำหนดตาม sample ท้ายสุดของ window
        run_ids:       List หรือ Array ของ simulationRun ประจำแต่ละ window
        start_indices: ตำแหน่งเริ่มต้น (sample index) ของแต่ละ window ใน run
        end_indices:   ตำแหน่งสิ้นสุด (sample index) ของแต่ละ window ใน run
    """
    X: np.ndarray
    y: np.ndarray
    run_ids: np.ndarray
    start_indices: np.ndarray
    end_indices: np.ndarray

    def summary(self) -> str:
        """สรุปข้อมูลของ WindowedDataset"""
        n_windows, window_size, n_features = self.X.shape if self.X.ndim == 3 else (0, 0, 0)
        n_normal = int(np.sum(self.y == 0))
        n_fault = int(np.sum(self.y == 1))
        return (
            f"WindowedDataset: {n_windows:,} windows | "
            f"Shape: ({n_windows}, {window_size}, {n_features}) | "
            f"Normal: {n_normal:,} | Fault: {n_fault:,}"
        )


# ---------------------------------------------------------------------------
# TEPPreprocessor
# ---------------------------------------------------------------------------

class TEPPreprocessor:
    """
    Preprocessor สำหรับการปรับสเกลข้อมูล TEP

    การันตีว่า fit StandardScaler จากเฉพาะ Normal Training Data เท่านั้น
    """

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        self.scaler = StandardScaler()
        self.feature_names = feature_names
        self.is_fitted: bool = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def fit(self, X_train_normal: np.ndarray) -> "TEPPreprocessor":
        """
        คำนวณ mean และ std จากเฉพาะชุดข้อมูลปกติ (Train Set)

        Args:
            X_train_normal: Array 2 มิติ (N_samples, N_features) ของชุดฝึกปกติ

        Returns:
            self
        """
        if X_train_normal.ndim != 2:
            raise ValueError(f"X_train_normal ต้องมี 2 มิติ แต่ได้ shape {X_train_normal.shape}")
        if X_train_normal.shape[0] == 0:
            raise ValueError("X_train_normal ไม่มีข้อมูล ไม่สามารถ fit scaler ได้")

        self.scaler.fit(X_train_normal)
        self.is_fitted = True
        self._logger.info(
            f"Fit StandardScaler สำเร็จบนข้อมูล {X_train_normal.shape[0]:,} ตัวอย่าง "
            f"({X_train_normal.shape[1]} ฟีเจอร์)"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        ปรับสเกลข้อมูลโดยใช้พารามิเตอร์ที่ fit ไว้แล้ว

        Args:
            X: Array (N_samples, N_features)

        Returns:
            X_scaled: Array ที่ปรับสเกลแล้ว
        """
        if not self.is_fitted:
            raise RuntimeError("ต้องเรียก .fit() ก่อน .transform()")
        return self.scaler.transform(X)

    def fit_transform(self, X_train_normal: np.ndarray) -> np.ndarray:
        """Fit บนชุดข้อมูลปกติและ transform ทันที"""
        return self.fit(X_train_normal).transform(X_train_normal)

    def save(self, filepath: Union[str, Path]) -> None:
        """บันทึก scaler ลงไฟล์"""
        if not self.is_fitted:
            raise RuntimeError("ไม่สามารถบันทึก scaler ที่ยังไม่ได้ fit")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, path)
        self._logger.info(f"บันทึก scaler ไปที่: {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TEPPreprocessor":
        """โหลด scaler จากไฟล์"""
        data = joblib.load(filepath)
        instance = cls(feature_names=data.get("feature_names"))
        instance.scaler = data["scaler"]
        instance.is_fitted = True
        return instance


# ---------------------------------------------------------------------------
# WindowGenerator
# ---------------------------------------------------------------------------

class WindowGenerator:
    """
    สร้าง Sliding Windows สำหรับโมเดล Sequential (เช่น Stacked LSTM, T²-LSTM)

    ข้อกำหนดความปลอดภัยทางวิทยาศาสตร์ข้อมูล:
        - ห้ามสร้าง window ข้าม simulationRun (ต้อง reset ทุกครั้งที่เปลี่ยน run)
        - window ต้องมีขนาดคงที่ `window_size`
        - ป้ายกำกับ (Label) ของ window กำหนดตามจุดสิ้นสุดของ window (Current time step)
    """

    def __init__(self, window_size: int = 50, stride: int = 1) -> None:
        if window_size < 1:
            raise ValueError(f"window_size ต้องมากกว่าหรือเท่ากับ 1 (ได้รับ {window_size})")
        if stride < 1:
            raise ValueError(f"stride ต้องมากกว่าหรือเท่ากับ 1 (ได้รับ {stride})")

        self.window_size = window_size
        self.stride = stride
        self._logger = logging.getLogger(self.__class__.__name__)

    def create_windows(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        run_ids: np.ndarray,
    ) -> WindowedDataset:
        """
        สร้าง sliding windows โดยแบ่งขอบเขตตาม simulationRun

        Args:
            features: Array 2 มิติ (N_samples, N_features)
            labels:   Array 1 มิติ (N_samples,) ของป้ายกำกับ (0=ปกติ, 1=ผิดปกติ)
            run_ids:  Array 1 มิติ (N_samples,) ระบุ simulationRun ของแต่ละ sample

        Returns:
            WindowedDataset: ชุดข้อมูล 3 มิติพร้อม metadata
        """
        if len(features) != len(labels) or len(features) != len(run_ids):
            raise ValueError(
                f"ขนาดของข้อมูลไม่สอดคล้องกัน: features={len(features)}, "
                f"labels={len(labels)}, run_ids={len(run_ids)}"
            )

        if len(features) == 0:
            return WindowedDataset(
                X=np.empty((0, self.window_size, features.shape[1] if features.ndim == 2 else 0)),
                y=np.empty((0,), dtype=int),
                run_ids=np.empty((0,), dtype=int),
                start_indices=np.empty((0,), dtype=int),
                end_indices=np.empty((0,), dtype=int),
            )

        window_list: List[np.ndarray] = []
        label_list: List[int] = []
        run_list: List[int] = []
        start_idx_list: List[int] = []
        end_idx_list: List[int] = []

        # หาจุดเปลี่ยนของ run เพื่อแบ่งกลุ่มอย่างมีประสิทธิภาพ
        unique_runs = pd.unique(run_ids)

        for r_id in unique_runs:
            mask = (run_ids == r_id)
            r_features = features[mask]
            r_labels = labels[mask]
            n_r_samples = len(r_features)

            if n_r_samples < self.window_size:
                # จำนวนตัวอย่างใน run น้อยกว่าขนาด window จึงไม่สามารถสร้าง window ได้
                continue

            for start in range(0, n_r_samples - self.window_size + 1, self.stride):
                end = start + self.window_size
                window_list.append(r_features[start:end])
                # กำหนด label ตามตัวอย่างสุดท้ายของหน้าต่างเวลา
                label_list.append(int(r_labels[end - 1]))
                run_list.append(int(r_id))
                start_idx_list.append(start)
                end_idx_list.append(end - 1)

        if not window_list:
            n_feat = features.shape[1] if features.ndim == 2 else 0
            return WindowedDataset(
                X=np.empty((0, self.window_size, n_feat)),
                y=np.empty((0,), dtype=int),
                run_ids=np.empty((0,), dtype=int),
                start_indices=np.empty((0,), dtype=int),
                end_indices=np.empty((0,), dtype=int),
            )

        X_out = np.asarray(window_list)
        y_out = np.asarray(label_list, dtype=int)
        runs_out = np.asarray(run_list, dtype=int)
        start_out = np.asarray(start_idx_list, dtype=int)
        end_out = np.asarray(end_idx_list, dtype=int)

        dataset = WindowedDataset(
            X=X_out,
            y=y_out,
            run_ids=runs_out,
            start_indices=start_out,
            end_indices=end_out,
        )
        self._logger.info(dataset.summary())
        return dataset
