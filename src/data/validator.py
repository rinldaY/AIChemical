"""
src/data/validator.py
---------------------
ตรวจสอบไฟล์ข้อมูล TEP ก่อนโหลด

Classes:
    SchemaType    — Enum ระบุประเภท schema ที่รองรับ
    ValidationResult — dataclass เก็บผลการตรวจสอบ
    TEPValidator  — ตรวจสอบ schema, dtype, completeness, และสถิติเบื้องต้น
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class SchemaType(Enum):
    """ประเภท schema ของไฟล์ TEP ที่รองรับ"""
    HARVARD_CSV = "harvard_csv"          # Rieth et al. (2017) Harvard Dataverse
    MATLAB_DAT = "matlab_dat"            # Downs & Vogel (1993) MATLAB .dat
    UNKNOWN = "unknown"                  # ไม่ระบุ — ต้องตรวจสอบด้วยตนเอง


# คอลัมน์ metadata ที่ต้องมีใน Schema A
_SCHEMA_A_META_COLS: List[str] = ["simulationRun", "sample", "faultNumber"]

# Pattern ชื่อคอลัมน์ feature ที่ยอมรับ (ตาม schema A)
_XMEAS_PATTERNS: List[str] = ["xmeas_", "XMEAS_", "XMEAS(", "x_meas_"]
_XMV_PATTERNS: List[str] = ["xmv_", "XMV_", "XMV(", "x_mv_"]

_N_XMEAS_EXPECTED: int = 41
_N_XMV_EXPECTED: int = 11
_VALID_FAULT_IDS: List[int] = list(range(0, 21))  # 0–20


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    ผลการตรวจสอบไฟล์ข้อมูล TEP

    Attributes:
        passed:      True หากผ่านการตรวจสอบทั้งหมด
        schema_type: ประเภท schema ที่ตรวจพบ
        errors:      รายการข้อผิดพลาดที่พบ (ทำให้ fail)
        warnings:    รายการคำเตือน (ไม่ทำให้ fail แต่ควรตรวจสอบ)
        info:        ข้อมูลสรุปของ dataset
    """
    passed: bool = False
    schema_type: SchemaType = SchemaType.UNKNOWN
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """สรุปผลการตรวจสอบเป็น string"""
        status = "[PASSED]" if self.passed else "[FAILED]"
        lines = [
            f"--- ValidationResult ---",
            f"Status      : {status}",
            f"Schema      : {self.schema_type.value}",
            f"Errors ({len(self.errors)})  : " + (
                "; ".join(self.errors) if self.errors else "none"
            ),
            f"Warnings ({len(self.warnings)}): " + (
                "; ".join(self.warnings) if self.warnings else "none"
            ),
        ]
        if self.info:
            lines.append(f"Info        : {self.info}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TEPValidator
# ---------------------------------------------------------------------------

class TEPValidator:
    """
    ตรวจสอบไฟล์ข้อมูล TEP ก่อนโหลดเข้าระบบ

    ตรวจสอบ:
    1. ไฟล์มีอยู่จริงและอ่านได้
    2. Schema type (Harvard CSV หรือ MATLAB .dat)
    3. คอลัมน์ครบถ้วนและชื่อถูกต้อง
    4. dtype ถูกต้อง
    5. ไม่มี missing values ในคอลัมน์หลัก
    6. ค่า faultNumber อยู่ในช่วงที่กำหนด
    7. สถิติเบื้องต้น (zero-variance columns, ค่าผิดปกติ)

    Example:
        >>> validator = TEPValidator()
        >>> result = validator.validate_file("data/raw/TEP_FaultFree_Training.csv")
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_file(
        self,
        filepath: str,
        expected_schema: Optional[SchemaType] = None,
    ) -> ValidationResult:
        """
        ตรวจสอบไฟล์ข้อมูล TEP ไฟล์เดียว

        Args:
            filepath:        path ของไฟล์ที่จะตรวจสอบ
            expected_schema: schema ที่คาดหวัง (None = ตรวจสอบอัตโนมัติ)

        Returns:
            ValidationResult ที่มีรายละเอียดผลการตรวจสอบ

        Raises:
            ไม่ raise exception — ข้อผิดพลาดทั้งหมดบันทึกใน result.errors
        """
        result = ValidationResult()
        path = Path(filepath)

        # ขั้นตอนที่ 1: ตรวจสอบไฟล์
        if not self._check_file_exists(path, result):
            return result

        # ขั้นตอนที่ 2: ตรวจสอบ extension และพยายามอ่าน
        df, ok = self._load_dataframe(path, result)
        if not ok or df is None:
            return result

        # ขั้นตอนที่ 3: ตรวจสอบ schema
        detected_schema = self._detect_schema(df, path)
        result.schema_type = detected_schema

        if expected_schema is not None and detected_schema != expected_schema:
            result.warnings.append(
                f"Schema ที่ตรวจพบ ({detected_schema.value}) "
                f"ไม่ตรงกับที่คาดหวัง ({expected_schema.value})"
            )

        # ขั้นตอนที่ 4: ตรวจสอบตาม schema
        if detected_schema == SchemaType.HARVARD_CSV:
            self._validate_schema_a(df, result)
        elif detected_schema == SchemaType.MATLAB_DAT:
            self._validate_schema_b(df, result)
        else:
            result.errors.append(
                "ไม่รู้จัก schema — กรุณาตรวจสอบ docs/data_checklist.md "
                "และระบุ schema ด้วยตนเอง"
            )

        # ขั้นตอนที่ 5: สถิติเบื้องต้น (เฉพาะถ้าไม่มี error ร้ายแรง)
        if not result.errors:
            self._check_basic_statistics(df, result)

        # สรุปผล
        result.passed = len(result.errors) == 0
        self._logger.info(result.summary())
        return result

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        ตรวจสอบ DataFrame ที่โหลดมาแล้ว

        Args:
            df: DataFrame ที่ต้องการตรวจสอบ

        Returns:
            ValidationResult
        """
        result = ValidationResult()
        detected_schema = self._detect_schema_from_df(df)
        result.schema_type = detected_schema

        if detected_schema == SchemaType.HARVARD_CSV:
            self._validate_schema_a(df, result)
        elif detected_schema == SchemaType.MATLAB_DAT:
            self._validate_schema_b(df, result)
        else:
            result.errors.append(
                "ไม่รู้จัก schema ของ DataFrame — "
                "ตรวจสอบ columns และดู docs/data_checklist.md"
            )

        if not result.errors:
            self._check_basic_statistics(df, result)

        result.passed = len(result.errors) == 0
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_file_exists(self, path: Path, result: ValidationResult) -> bool:
        """ตรวจสอบว่าไฟล์มีอยู่จริง"""
        if not path.exists():
            result.errors.append(f"ไม่พบไฟล์: {path}")
            return False
        if not path.is_file():
            result.errors.append(f"Path ไม่ใช่ไฟล์: {path}")
            return False
        size_kb = path.stat().st_size / 1024
        result.info["file_size_kb"] = round(size_kb, 2)
        if size_kb < 1:
            result.warnings.append(
                f"ไฟล์มีขนาดเล็กมาก ({size_kb:.2f} KB) — อาจเป็นไฟล์ว่างหรือไม่สมบูรณ์"
            )
        return True

    def _load_dataframe(
        self, path: Path, result: ValidationResult
    ) -> Tuple[Optional[pd.DataFrame], bool]:
        """
        พยายามอ่านไฟล์เป็น DataFrame

        สำหรับ MATLAB .dat:
            ไฟล์ TEP ต้นฉบับเก็บข้อมูลในรูป (52 variables × N samples)
            คือ rows = variables, cols = samples — ต้อง transpose ก่อนใช้งาน
            ผลหลัง transpose: (N samples × 52 variables) ซึ่งเป็นรูปแบบที่ถูกต้อง

        Returns:
            (df, success) — df เป็น None ถ้าอ่านไม่ได้
        """
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix == ".dat":
                # MATLAB .dat: whitespace-delimited, ไม่มี header
                # ไฟล์ TEP .dat เก็บแบบ transposed: (52 vars × N samples)
                # ต้อง transpose เพื่อให้ได้ (N samples × 52 vars)
                df_raw = pd.read_csv(path, sep=r"\s+", header=None)
                n_rows_raw, n_cols_raw = df_raw.shape

                # ตรวจสอบว่าต้อง transpose หรือไม่
                # ถ้า rows < cols และ rows == 52 → แสดงว่าเป็น transposed format
                if n_rows_raw == 52 and n_cols_raw > 52:
                    df = df_raw.T.reset_index(drop=True)
                    df.columns = range(df.shape[1])
                    result.info["dat_transposed"] = True
                    result.info["dat_original_shape"] = (n_rows_raw, n_cols_raw)
                    self._logger.info(
                        f"MATLAB .dat: ตรวจพบรูปแบบ transposed "
                        f"({n_rows_raw}×{n_cols_raw}) → transpose เป็น "
                        f"({df.shape[0]}×{df.shape[1]})"
                    )
                elif n_cols_raw == 52:
                    # ไม่ต้อง transpose — อยู่ในรูปแบบปกติแล้ว
                    df = df_raw
                    result.info["dat_transposed"] = False
                else:
                    # รูปแบบไม่ตรงทั้งสองแบบ — ส่งต่อไปให้ validate จัดการ
                    df = df_raw
                    result.info["dat_transposed"] = False
                    result.warnings.append(
                        f"MATLAB .dat มี shape ({n_rows_raw}×{n_cols_raw}) "
                        f"ไม่ตรงกับรูปแบบที่รู้จัก "
                        f"(คาดว่า 52×N หรือ N×52)"
                    )
            elif suffix in (".rdata", ".rda", ".mat", ".xlsx"):
                result.errors.append(
                    f"รูปแบบไฟล์ '{suffix}' ยังไม่รองรับ — "
                    f"ดูรายละเอียดใน docs/data_checklist.md (ส่วน Schema C)"
                )
                return None, False
            else:
                result.errors.append(
                    f"ไม่รู้จักนามสกุลไฟล์ '{suffix}' — "
                    f"รองรับเฉพาะ .csv และ .dat"
                )
                return None, False
        except Exception as exc:
            result.errors.append(f"อ่านไฟล์ไม่สำเร็จ: {exc}")
            return None, False

        result.info["shape"] = df.shape
        result.info["columns"] = list(df.columns)
        return df, True

    def _detect_schema(self, df: pd.DataFrame, path: Path) -> SchemaType:
        """ตรวจสอบ schema จาก DataFrame และ path"""
        if path.suffix.lower() == ".dat":
            return SchemaType.MATLAB_DAT
        return self._detect_schema_from_df(df)

    def _detect_schema_from_df(self, df: pd.DataFrame) -> SchemaType:
        """ตรวจสอบ schema จาก column names"""
        cols = [str(c).lower() for c in df.columns]
        has_meta = any("simulationrun" in c or "faultnumber" in c for c in cols)
        has_xmeas = any(
            any(c.startswith(p.lower()) for p in _XMEAS_PATTERNS) for c in cols
        )
        if has_meta or has_xmeas:
            return SchemaType.HARVARD_CSV
        # .dat ที่ถูก rename หรือไม่มี header (คอลัมน์เป็นตัวเลขทั้งหมด)
        if all(str(c).isdigit() or isinstance(c, (int, np.integer)) for c in df.columns):
            return SchemaType.MATLAB_DAT
        return SchemaType.UNKNOWN

    def _validate_schema_a(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """ตรวจสอบ Schema A (Harvard CSV)"""
        cols_lower = {c.lower(): c for c in df.columns}

        # ตรวจสอบ metadata columns
        for meta in _SCHEMA_A_META_COLS:
            if meta.lower() not in cols_lower:
                result.errors.append(
                    f"ไม่พบคอลัมน์ metadata ที่จำเป็น: '{meta}'"
                )

        # ตรวจสอบ XMEAS columns
        xmeas_cols = self._find_pattern_cols(df.columns, _XMEAS_PATTERNS)
        if len(xmeas_cols) < _N_XMEAS_EXPECTED:
            result.errors.append(
                f"พบ XMEAS columns {len(xmeas_cols)} คอลัมน์ "
                f"(ต้องการ {_N_XMEAS_EXPECTED})"
            )
        result.info["n_xmeas_found"] = len(xmeas_cols)

        # ตรวจสอบ XMV columns
        xmv_cols = self._find_pattern_cols(df.columns, _XMV_PATTERNS)
        if len(xmv_cols) < _N_XMV_EXPECTED:
            result.errors.append(
                f"พบ XMV columns {len(xmv_cols)} คอลัมน์ "
                f"(ต้องการ {_N_XMV_EXPECTED})"
            )
        result.info["n_xmv_found"] = len(xmv_cols)

        # ตรวจสอบ missing values ใน feature columns
        feature_cols = xmeas_cols + xmv_cols
        if feature_cols:
            n_missing = df[feature_cols].isnull().sum().sum()
            if n_missing > 0:
                result.errors.append(
                    f"พบ missing values {n_missing} ค่าใน feature columns"
                )
            result.info["n_missing_features"] = int(n_missing)

        # ตรวจสอบ faultNumber values
        if "faultNumber" in df.columns or "faultnumber" in cols_lower:
            actual_col = cols_lower.get("faultnumber", "faultNumber")
            if actual_col in df.columns:
                invalid_faults = df[actual_col][
                    ~df[actual_col].isin(_VALID_FAULT_IDS)
                ]
                if len(invalid_faults) > 0:
                    result.errors.append(
                        f"พบ faultNumber ที่ไม่ถูกต้อง: "
                        f"{sorted(invalid_faults.unique().tolist())}"
                    )
                result.info["fault_ids_found"] = sorted(
                    df[actual_col].unique().tolist()
                )

        # ตรวจสอบ dtype ของ feature columns
        if feature_cols:
            non_numeric = [
                c for c in feature_cols
                if not pd.api.types.is_numeric_dtype(df[c])
            ]
            if non_numeric:
                result.errors.append(
                    f"Feature columns ต่อไปนี้ไม่ใช่ numeric: {non_numeric[:5]}"
                )

    def _validate_schema_b(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """
        ตรวจสอบ Schema B (MATLAB .dat)

        หลัง transpose แล้ว df ควรมี shape (N_samples × 52)
        โดย 52 = 41 XMEAS + 11 XMV
        """
        n_rows, n_cols = df.shape
        if n_cols != 52:
            result.errors.append(
                f"MATLAB .dat ต้องมี 52 คอลัมน์ (41 XMEAS + 11 XMV) "
                f"แต่พบ {n_cols} คอลัมน์ (shape หลัง transpose: {n_rows}×{n_cols})"
            )
        else:
            result.info["n_samples"] = n_rows
            result.info["n_features"] = n_cols
            result.info["n_xmeas"] = 41
            result.info["n_xmv"] = 11

        result.warnings.append(
            "Schema B (MATLAB .dat): ไม่มีคอลัมน์ simulationRun และ faultNumber — "
            "ต้องระบุ metadata ด้วยตนเอง และต้องระบุ fault ID จากชื่อไฟล์ "
            "(เช่น d00.dat = fault 0, d01.dat = fault 1)"
        )
        n_missing = df.isnull().sum().sum()
        if n_missing > 0:
            result.errors.append(f"พบ missing values {n_missing} ค่า")
        result.info["n_missing_features"] = int(n_missing)

    def _check_basic_statistics(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """ตรวจสอบสถิติเบื้องต้นของ feature columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return

        # หา zero-variance columns
        zero_var = numeric_df.columns[numeric_df.std() == 0].tolist()
        if zero_var:
            result.warnings.append(
                f"พบคอลัมน์ที่ variance = 0 (ค่าคงที่): {zero_var[:5]}"
            )
        result.info["zero_variance_cols"] = zero_var

        # สรุปสถิติ
        result.info["n_rows"] = len(df)
        result.info["n_numeric_cols"] = len(numeric_df.columns)

    @staticmethod
    def _find_pattern_cols(columns, patterns: List[str]) -> List[str]:
        """หาคอลัมน์ที่ชื่อขึ้นต้นด้วย pattern ที่กำหนด (case-insensitive)"""
        result = []
        for col in columns:
            col_str = str(col)
            if any(col_str.lower().startswith(p.lower()) for p in patterns):
                result.append(col_str)
        return result
