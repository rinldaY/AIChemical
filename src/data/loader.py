"""
src/data/loader.py
------------------
TEPDataLoader — โหลดและแบ่งข้อมูล Tennessee Eastman Process

หลักการสำคัญ:
    - แบ่งข้อมูลตาม simulationRun (ไม่แบ่งกลาง run)
    - ห้าม window หน้าต่างซ้อนทับข้าม Train/Val/Test
    - Train Set ใช้ข้อมูลปกติ (faultNumber=0) เท่านั้น
    - Scaler fit จาก Train Set เท่านั้น

Classes:
    SplitResult  — dataclass เก็บผลการแบ่งข้อมูล
    TEPDataLoader — โหลด, ตรวจสอบ, และแบ่งข้อมูล TEP
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.validator import SchemaType, TEPValidator, ValidationResult
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# SplitResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """
    ผลการแบ่งข้อมูล TEP

    Attributes:
        X_train:      feature matrix สำหรับ training (normal data เท่านั้น)
        X_val:        feature matrix สำหรับ validation
        X_test:       feature matrix สำหรับ test
        y_val:        labels สำหรับ validation (0=normal, 1=fault)
        y_test:       labels สำหรับ test
        run_ids_train: simulationRun IDs ใน train set
        run_ids_val:   simulationRun IDs ใน val set
        run_ids_test:  simulationRun IDs ใน test set
        fault_ids_val: fault IDs ที่มีใน val set
        fault_ids_test: fault IDs ที่มีใน test set
        feature_names: ชื่อ feature columns
        meta:          ข้อมูลเพิ่มเติม (split ratios จริง, จำนวน runs, ฯลฯ)
    """
    X_train: np.ndarray = field(default_factory=lambda: np.array([]))
    X_val: np.ndarray = field(default_factory=lambda: np.array([]))
    X_test: np.ndarray = field(default_factory=lambda: np.array([]))
    y_val: np.ndarray = field(default_factory=lambda: np.array([]))
    y_test: np.ndarray = field(default_factory=lambda: np.array([]))
    run_ids_train: List[int] = field(default_factory=list)
    run_ids_val: List[int] = field(default_factory=list)
    run_ids_test: List[int] = field(default_factory=list)
    fault_ids_val: List[int] = field(default_factory=list)
    fault_ids_test: List[int] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """สรุปผลการแบ่งข้อมูล"""
        lines = [
            "--- SplitResult Summary ---",
            f"Train  : {self.X_train.shape} | runs={len(self.run_ids_train)} | normal only",
            f"Val    : {self.X_val.shape} | runs={len(self.run_ids_val)} "
            f"| faults={self.fault_ids_val}",
            f"Test   : {self.X_test.shape} | runs={len(self.run_ids_test)} "
            f"| faults={self.fault_ids_test}",
            f"Features: {len(self.feature_names)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TEPDataLoader
# ---------------------------------------------------------------------------

class TEPDataLoader:
    """
    โหลดและแบ่งข้อมูล Tennessee Eastman Process (TEP)

    รองรับ:
        - Schema A: Harvard Dataverse CSV (Rieth et al. 2017)
        - Schema B: MATLAB .dat (Downs & Vogel 1993) — บางส่วน

    ไม่รองรับ:
        - .RData, .mat, .xlsx — ดู docs/data_checklist.md

    Args:
        train_normal_ratio: สัดส่วน normal runs สำหรับ Train (default: 0.70)
        val_normal_ratio:   สัดส่วน normal runs สำหรับ Val (default: 0.15)
        val_fault_ratio:    สัดส่วน fault runs ต่อ fault ID สำหรับ Val (default: 0.50)
        random_seed:        seed สำหรับการสุ่มลำดับ runs (default: 42)

    Example:
        >>> loader = TEPDataLoader()
        >>> result = loader.load_and_split(
        ...     normal_file="data/raw/TEP_FaultFree_Training.csv",
        ...     fault_file="data/raw/TEP_Faulty_Training.csv",
        ... )
        >>> print(result.summary())

    Note:
        ยังไม่ fit Scaler ในขั้นตอนนี้ — จะเพิ่มใน preprocessing pipeline ภายหลัง
    """

    def __init__(
        self,
        train_normal_ratio: float = 0.70,
        val_normal_ratio: float = 0.15,
        val_fault_ratio: float = 0.50,
        random_seed: int = 42,
    ) -> None:
        if not (0 < train_normal_ratio < 1):
            raise ValueError("train_normal_ratio ต้องอยู่ระหว่าง 0 และ 1")
        if not (0 < val_normal_ratio < 1):
            raise ValueError("val_normal_ratio ต้องอยู่ระหว่าง 0 และ 1")
        if abs(train_normal_ratio + val_normal_ratio - 0.85) > 0.01:
            # test ratio = 1 - train - val
            pass  # อนุญาต — คำนวณ test_ratio อัตโนมัติ

        self.train_normal_ratio = train_normal_ratio
        self.val_normal_ratio = val_normal_ratio
        self.val_fault_ratio = val_fault_ratio
        self.random_seed = random_seed
        self._validator = TEPValidator()
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_split(
        self,
        normal_file: str,
        fault_file: Optional[str] = None,
        fault_ids: Optional[List[int]] = None,
    ) -> SplitResult:
        """
        โหลดข้อมูลจากไฟล์และแบ่งเป็น Train/Val/Test

        Args:
            normal_file: path ของไฟล์ข้อมูลปกติ (faultNumber=0)
            fault_file:  path ของไฟล์ข้อมูลผิดปกติ (ถ้ามีไฟล์แยก)
                         None = ไฟล์เดียวที่มีทั้ง normal และ fault
            fault_ids:   รายการ fault ID ที่จะโหลด (None = ทั้งหมดที่พบ)

        Returns:
            SplitResult: ผลการแบ่งข้อมูล

        Raises:
            FileNotFoundError: ถ้าไม่พบไฟล์
            ValueError: ถ้าข้อมูลไม่ผ่านการตรวจสอบ schema
        """
        # โหลดและตรวจสอบไฟล์ปกติ
        self._logger.info(f"กำลังโหลดไฟล์ปกติ: {normal_file}")
        normal_df = self._load_validated_file(normal_file, expect_normal_only=True)

        # โหลดไฟล์ fault (ถ้ามี)
        fault_df: Optional[pd.DataFrame] = None
        if fault_file is not None:
            self._logger.info(f"กำลังโหลดไฟล์ fault: {fault_file}")
            fault_df = self._load_validated_file(fault_file, expect_normal_only=False)

        # ถ้าไม่มีไฟล์แยก ให้แยก normal/fault จาก normal_df
        if fault_df is None:
            fault_df = normal_df[normal_df["faultNumber"] != 0].copy()
            normal_df = normal_df[normal_df["faultNumber"] == 0].copy()

        # กรอง fault_ids ถ้าระบุ
        if fault_ids is not None:
            fault_df = fault_df[fault_df["faultNumber"].isin(fault_ids)]

        # ตรวจสอบว่ามีข้อมูลเพียงพอ
        self._check_sufficient_data(normal_df, fault_df)

        # แบ่งข้อมูล
        return self._split_by_run(normal_df, fault_df)

    def load_file_only(self, filepath: str) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        โหลดไฟล์และตรวจสอบ schema โดยไม่แบ่งข้อมูล
        ใช้สำหรับตรวจสอบไฟล์เบื้องต้น

        Args:
            filepath: path ของไฟล์

        Returns:
            (DataFrame, ValidationResult)
        """
        result = self._validator.validate_file(filepath)
        if not result.passed:
            self._logger.error(
                f"ไฟล์ไม่ผ่านการตรวจสอบ:\n{result.summary()}"
            )
            return pd.DataFrame(), result

        path = Path(filepath)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_csv(filepath, sep=r"\s+", header=None)

        self._logger.info(
            f"โหลดสำเร็จ: {path.name} | shape={df.shape} | "
            f"schema={result.schema_type.value}"
        )
        return df, result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_validated_file(
        self, filepath: str, expect_normal_only: bool = False
    ) -> pd.DataFrame:
        """
        โหลดและตรวจสอบไฟล์ — raise ถ้าไม่ผ่าน

        สำหรับ MATLAB .dat:
            ไฟล์ TEP ต้นฉบับเป็น transposed (52 vars × N samples)
            Loader จะ transpose อัตโนมัติผ่าน validator
            ผลที่ได้: DataFrame (N samples × 52 vars) พร้อมใช้งาน
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์: {filepath}")

        validation = self._validator.validate_file(filepath)
        if not validation.passed:
            raise ValueError(
                f"ไฟล์ไม่ผ่านการตรวจสอบ:\n{validation.summary()}"
            )

        # โหลด — ใช้ logic เดียวกับ validator (รวม transpose สำหรับ .dat)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(filepath)
        elif suffix == ".dat":
            df_raw = pd.read_csv(filepath, sep=r"\s+", header=None)
            n_rows_raw, n_cols_raw = df_raw.shape
            if n_rows_raw == 52 and n_cols_raw > 52:
                # Transposed format: (52 vars × N samples) → transpose
                df = df_raw.T.reset_index(drop=True)
                df.columns = range(df.shape[1])
                self._logger.info(
                    f"MATLAB .dat: transpose ({n_rows_raw}×{n_cols_raw}) "
                    f"→ ({df.shape[0]}×{df.shape[1]})"
                )
            else:
                df = df_raw
        else:
            raise ValueError(f"รูปแบบไฟล์ไม่รองรับ: {suffix}")

        # ตรวจสอบ normal-only constraint (สำหรับ CSV ที่มี faultNumber)
        if expect_normal_only and "faultNumber" in df.columns:
            n_fault = (df["faultNumber"] != 0).sum()
            if n_fault > 0:
                self._logger.warning(
                    f"ไฟล์ปกติพบ fault samples {n_fault} rows "
                    f"(faultNumber != 0) — จะกรองออก"
                )
                df = df[df["faultNumber"] == 0].copy()

        return df

    def _check_sufficient_data(
        self, normal_df: pd.DataFrame, fault_df: pd.DataFrame
    ) -> None:
        """ตรวจสอบว่ามี runs เพียงพอสำหรับการแบ่ง"""
        if "simulationRun" not in normal_df.columns:
            raise ValueError(
                "ไม่พบคอลัมน์ 'simulationRun' — ไม่สามารถแบ่งข้อมูลตาม run ได้\n"
                "ตรวจสอบ schema ใน docs/data_checklist.md"
            )
        n_normal_runs = normal_df["simulationRun"].nunique()
        min_runs_needed = 3  # อย่างน้อย 1 run ต่อ split
        if n_normal_runs < min_runs_needed:
            raise ValueError(
                f"พบ normal runs เพียง {n_normal_runs} runs "
                f"(ต้องการอย่างน้อย {min_runs_needed})"
            )
        self._logger.info(
            f"ข้อมูลปกติ: {n_normal_runs} runs | "
            f"ข้อมูลผิดปกติ: {fault_df['simulationRun'].nunique() if not fault_df.empty else 0} runs"
        )

    def _split_by_run(
        self, normal_df: pd.DataFrame, fault_df: pd.DataFrame
    ) -> SplitResult:
        """
        แบ่งข้อมูลตาม simulationRun

        หลักการ:
        - Normal runs แบ่งเป็น Train / Val / Test
        - Fault runs (ต่อ fault ID) แบ่งเป็น Val / Test
        - ไม่แบ่งกลาง run — ทุก sample ใน run เดียวกันอยู่ใน split เดียวกัน
        """
        rng = np.random.default_rng(self.random_seed)

        # รวบรวม feature columns
        feature_cols = self._get_feature_cols(normal_df)
        if not feature_cols:
            raise ValueError("ไม่พบ feature columns (XMEAS/XMV)")

        # แบ่ง normal runs
        normal_run_ids = sorted(normal_df["simulationRun"].unique())
        rng.shuffle(normal_run_ids)  # type: ignore[arg-type]
        n_total = len(normal_run_ids)
        n_train = max(1, int(n_total * self.train_normal_ratio))
        n_val = max(1, int(n_total * self.val_normal_ratio))
        # n_test = เหลือทั้งหมด

        train_runs = normal_run_ids[:n_train]
        val_runs = normal_run_ids[n_train:n_train + n_val]
        test_runs = normal_run_ids[n_train + n_val:]

        self._logger.info(
            f"Normal runs — Train: {len(train_runs)}, "
            f"Val: {len(val_runs)}, Test: {len(test_runs)}"
        )

        # แบ่ง fault runs ต่อ fault ID
        fault_val_dfs: List[pd.DataFrame] = []
        fault_test_dfs: List[pd.DataFrame] = []
        fault_ids_found: List[int] = []

        if not fault_df.empty and "simulationRun" in fault_df.columns:
            for fid in sorted(fault_df["faultNumber"].unique()):
                f_runs = sorted(
                    fault_df[fault_df["faultNumber"] == fid]["simulationRun"].unique()
                )
                rng.shuffle(f_runs)  # type: ignore[arg-type]
                n_f_val = max(1, int(len(f_runs) * self.val_fault_ratio))
                val_f_runs = f_runs[:n_f_val]
                test_f_runs = f_runs[n_f_val:]

                mask_val = (
                    (fault_df["faultNumber"] == fid) &
                    (fault_df["simulationRun"].isin(val_f_runs))
                )
                mask_test = (
                    (fault_df["faultNumber"] == fid) &
                    (fault_df["simulationRun"].isin(test_f_runs))
                )
                fault_val_dfs.append(fault_df[mask_val])
                fault_test_dfs.append(fault_df[mask_test])
                fault_ids_found.append(int(fid))

        # รวม normal val/test กับ fault val/test
        normal_val_df = normal_df[normal_df["simulationRun"].isin(val_runs)]
        normal_test_df = normal_df[normal_df["simulationRun"].isin(test_runs)]

        val_normal_part = normal_val_df[feature_cols].values
        test_normal_part = normal_test_df[feature_cols].values

        if fault_val_dfs:
            fault_val_combined = pd.concat(fault_val_dfs, ignore_index=True)
            fault_test_combined = pd.concat(fault_test_dfs, ignore_index=True)
            X_val = np.vstack([
                val_normal_part,
                fault_val_combined[feature_cols].values,
            ])
            y_val = np.concatenate([
                np.zeros(len(val_normal_part), dtype=int),
                np.ones(len(fault_val_combined), dtype=int),
            ])
            X_test = np.vstack([
                test_normal_part,
                fault_test_combined[feature_cols].values,
            ])
            y_test = np.concatenate([
                np.zeros(len(test_normal_part), dtype=int),
                np.ones(len(fault_test_combined), dtype=int),
            ])
        else:
            X_val = val_normal_part
            y_val = np.zeros(len(val_normal_part), dtype=int)
            X_test = test_normal_part
            y_test = np.zeros(len(test_normal_part), dtype=int)

        X_train = normal_df[
            normal_df["simulationRun"].isin(train_runs)
        ][feature_cols].values

        result = SplitResult(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_val=y_val,
            y_test=y_test,
            run_ids_train=list(map(int, train_runs)),
            run_ids_val=list(map(int, val_runs)),
            run_ids_test=list(map(int, test_runs)),
            fault_ids_val=fault_ids_found,
            fault_ids_test=fault_ids_found,
            feature_names=feature_cols,
            meta={
                "n_normal_runs_total": len(normal_run_ids),
                "train_ratio_actual": len(train_runs) / len(normal_run_ids),
                "val_ratio_actual": len(val_runs) / len(normal_run_ids),
                "test_ratio_actual": len(test_runs) / len(normal_run_ids),
                "random_seed": self.random_seed,
                "note": (
                    "Scaler ยังไม่ได้ fit — ใช้ preprocessing pipeline ใน pipeline.py"
                ),
            },
        )

        self._logger.info(result.summary())
        return result

    @staticmethod
    def _get_feature_cols(df: pd.DataFrame) -> List[str]:
        """หาคอลัมน์ feature (XMEAS + XMV) จาก DataFrame"""
        patterns = ["xmeas_", "xmv_", "XMEAS_", "XMV_", "XMEAS(", "XMV("]
        return [
            c for c in df.columns
            if any(str(c).lower().startswith(p.lower()) for p in patterns)
        ]
