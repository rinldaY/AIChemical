"""
tests/test_validator.py
-----------------------
Unit tests สำหรับ TEPValidator

⚠️ IMPORTANT: ข้อมูลทั้งหมดในไฟล์นี้เป็น MOCK DATA ที่สร้างขึ้นเพื่อทดสอบเท่านั้น
   ไม่มีส่วนใดเป็นข้อมูล TEP จริง และห้ามนำผลทดสอบไปอ้างอิงในการทดลองจริง

Tests:
    - test_schema_a_valid:           DataFrame ที่ถูกต้องตาม schema A ต้องผ่าน
    - test_missing_metadata_col:     ขาด metadata column ต้อง fail
    - test_missing_xmeas_cols:       XMEAS columns ไม่ครบต้อง fail
    - test_missing_xmv_cols:         XMV columns ไม่ครบต้อง fail
    - test_invalid_fault_number:     faultNumber นอกช่วง 0–20 ต้อง fail
    - test_missing_values_in_features: missing values ต้อง fail
    - test_zero_variance_warning:    zero-variance column ต้อง warn
    - test_nonexistent_file:         ไฟล์ไม่มีอยู่ต้อง fail
    - test_unsupported_extension:    .mat/.RData ต้อง fail พร้อม error message ที่อธิบาย
    - test_unknown_schema:           คอลัมน์ไม่ตรง pattern ใดต้อง UNKNOWN schema
    - test_schema_b_dat_52cols:      DataFrame 52 คอลัมน์ต้องตรวจพบ schema B
    - test_schema_b_wrong_col_count: .dat ที่มีคอลัมน์ไม่ใช่ 52 ต้อง fail
"""

import io
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# MOCK DATA NOTE: ข้อมูลทั้งหมดด้านล่างเป็น MOCK DATA ไม่ใช่ข้อมูล TEP จริง
# ---------------------------------------------------------------------------

# ที่อยู่ของ src เพื่อให้ import ได้จาก root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.validator import SchemaType, TEPValidator, ValidationResult


# ---------------------------------------------------------------------------
# Fixtures — MOCK DATA
# ---------------------------------------------------------------------------

def _make_mock_schema_a(
    n_rows: int = 100,
    n_xmeas: int = 41,
    n_xmv: int = 11,
    n_runs: int = 5,
    fault_ids: list = None,
    include_missing: bool = False,
    zero_var_col: bool = False,
) -> pd.DataFrame:
    """
    สร้าง DataFrame จำลอง Schema A (Harvard CSV format)

    ⚠️ MOCK DATA — ไม่ใช่ข้อมูล TEP จริง
    ค่าตัวเลขเป็นค่าสุ่มจาก numpy random ไม่มีความหมายทางฟิสิกส์
    """
    rng = np.random.default_rng(0)  # seed คงที่สำหรับ test reproducibility

    if fault_ids is None:
        fault_ids = [0]

    rows_per_run = n_rows // n_runs

    meta = {
        "simulationRun": np.repeat(range(1, n_runs + 1), rows_per_run),
        "sample": np.tile(range(rows_per_run), n_runs),
        "faultNumber": np.zeros(rows_per_run * n_runs, dtype=int),
    }

    # override fault IDs for testing
    if len(fault_ids) > 1:
        per_fault = len(meta["faultNumber"]) // len(fault_ids)
        for i, fid in enumerate(fault_ids):
            start = i * per_fault
            end = start + per_fault if i < len(fault_ids) - 1 else len(meta["faultNumber"])
            meta["faultNumber"][start:end] = fid

    xmeas_cols = {f"xmeas_{i}": rng.normal(0, 1, rows_per_run * n_runs)
                  for i in range(1, n_xmeas + 1)}
    xmv_cols = {f"xmv_{i}": rng.normal(0, 1, rows_per_run * n_runs)
                for i in range(1, n_xmv + 1)}

    df = pd.DataFrame({**meta, **xmeas_cols, **xmv_cols})

    if include_missing:
        df.loc[0, "xmeas_1"] = np.nan

    if zero_var_col:
        df["xmeas_41"] = 999.0  # ค่าคงที่

    return df


def _make_mock_schema_b(n_rows: int = 100, n_cols: int = 52) -> pd.DataFrame:
    """
    สร้าง DataFrame จำลอง Schema B (MATLAB .dat format — ไม่มี header)

    ⚠️ MOCK DATA — ไม่ใช่ข้อมูล TEP จริง
    """
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, (n_rows, n_cols))
    return pd.DataFrame(data, columns=range(n_cols))


# ---------------------------------------------------------------------------
# Tests — Schema A (Harvard CSV)
# ---------------------------------------------------------------------------

class TestSchemaAValidation:
    """Tests สำหรับ Schema A (Harvard Dataverse CSV format)"""

    def test_schema_a_valid(self):
        """DataFrame ที่ถูกต้องครบตาม schema A ต้องผ่านการตรวจสอบ"""
        df = _make_mock_schema_a()  # MOCK DATA
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert result.passed, f"ต้องผ่าน แต่พบ errors: {result.errors}"
        assert result.schema_type == SchemaType.HARVARD_CSV

    def test_missing_metadata_col_simulationRun(self):
        """ขาดคอลัมน์ simulationRun ต้อง fail"""
        df = _make_mock_schema_a()  # MOCK DATA
        df = df.drop(columns=["simulationRun"])
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert not result.passed
        assert any("simulationRun" in e for e in result.errors), \
            f"ต้องมี error เกี่ยวกับ simulationRun แต่ได้: {result.errors}"

    def test_missing_metadata_col_faultNumber(self):
        """ขาดคอลัมน์ faultNumber ต้อง fail"""
        df = _make_mock_schema_a()  # MOCK DATA
        df = df.drop(columns=["faultNumber"])
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert not result.passed
        assert any("faultNumber" in e for e in result.errors)

    def test_missing_xmeas_cols(self):
        """XMEAS columns ไม่ครบ 41 ต้อง fail"""
        df = _make_mock_schema_a(n_xmeas=10)  # MOCK DATA: เจตนาให้น้อย
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert not result.passed
        assert any("XMEAS" in e for e in result.errors), \
            f"ต้องมี error เกี่ยวกับ XMEAS แต่ได้: {result.errors}"

    def test_missing_xmv_cols(self):
        """XMV columns ไม่ครบ 11 ต้อง fail"""
        df = _make_mock_schema_a(n_xmv=5)  # MOCK DATA: เจตนาให้น้อย
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert not result.passed
        assert any("XMV" in e for e in result.errors)

    def test_invalid_fault_number_out_of_range(self):
        """faultNumber = 21 (นอกช่วง 0–20) ต้อง fail"""
        df = _make_mock_schema_a()  # MOCK DATA
        df.loc[0, "faultNumber"] = 99  # ค่าไม่ถูกต้อง
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert not result.passed
        assert any("faultNumber" in e.lower() or "fault" in e.lower()
                   for e in result.errors), \
            f"ต้องมี error เกี่ยวกับ faultNumber แต่ได้: {result.errors}"

    def test_missing_values_in_features(self):
        """NaN ใน feature columns ต้อง fail"""
        df = _make_mock_schema_a(include_missing=True)  # MOCK DATA
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert not result.passed
        assert any("missing" in e.lower() for e in result.errors)

    def test_zero_variance_column_gives_warning(self):
        """คอลัมน์ที่ค่าคงที่ตลอด ต้องมี warning (ไม่ใช่ error)"""
        df = _make_mock_schema_a(zero_var_col=True)  # MOCK DATA
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        # อาจผ่านหรือไม่ผ่านก็ได้ แต่ต้องมี warning
        assert len(result.warnings) > 0, \
            "ต้องมี warning สำหรับ zero-variance column"
        assert any("variance" in w.lower() or "zero" in w.lower()
                   for w in result.warnings), \
            f"ต้องมี warning เกี่ยวกับ variance แต่ได้: {result.warnings}"

    def test_valid_fault_numbers_in_range(self):
        """faultNumber 0–20 ทั้งหมดต้องผ่าน"""
        df = _make_mock_schema_a(n_runs=21, fault_ids=list(range(21)))  # MOCK DATA
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        # ต้องไม่มี error เกี่ยวกับ faultNumber range
        fault_errors = [e for e in result.errors
                        if "faultNumber" in e and "ไม่ถูกต้อง" in e]
        assert len(fault_errors) == 0, \
            f"faultNumber 0–20 ต้องถูกต้อง แต่พบ: {fault_errors}"


# ---------------------------------------------------------------------------
# Tests — File-level validation
# ---------------------------------------------------------------------------

class TestFileValidation:
    """Tests สำหรับการตรวจสอบไฟล์"""

    def test_nonexistent_file_fails(self):
        """ไฟล์ที่ไม่มีอยู่ต้อง fail พร้อม error ชัดเจน"""
        validator = TEPValidator()
        result = validator.validate_file("data/raw/this_file_does_not_exist.csv")

        assert not result.passed
        assert len(result.errors) > 0
        assert any("ไม่พบไฟล์" in e or "not found" in e.lower()
                   for e in result.errors), \
            f"ต้องมี error ว่าไม่พบไฟล์ แต่ได้: {result.errors}"

    def test_unsupported_rdata_extension(self, tmp_path):
        """ไฟล์ .RData ต้อง fail พร้อม error ที่บอกว่า unsupported"""
        fake_rdata = tmp_path / "test.RData"
        fake_rdata.write_bytes(b"fake rdata content")

        validator = TEPValidator()
        result = validator.validate_file(str(fake_rdata))

        assert not result.passed
        assert any(
            "ยังไม่รองรับ" in e or "unsupported" in e.lower() or ".rdata" in e.lower()
            for e in result.errors
        ), f"ต้องมี error ว่า .RData ยังไม่รองรับ แต่ได้: {result.errors}"

    def test_unsupported_mat_extension(self, tmp_path):
        """ไฟล์ .mat ต้อง fail พร้อม error ที่อ้างถึง data_checklist"""
        fake_mat = tmp_path / "test.mat"
        fake_mat.write_bytes(b"fake mat content")

        validator = TEPValidator()
        result = validator.validate_file(str(fake_mat))

        assert not result.passed
        assert any(".mat" in e.lower() or "ยังไม่รองรับ" in e
                   for e in result.errors)

    def test_unsupported_unknown_extension(self, tmp_path):
        """ไฟล์นามสกุลที่ไม่รู้จักต้อง fail"""
        fake = tmp_path / "test.xyz123"
        fake.write_bytes(b"unknown content")

        validator = TEPValidator()
        result = validator.validate_file(str(fake))

        assert not result.passed
        assert len(result.errors) > 0

    def test_valid_csv_file(self, tmp_path):
        """ไฟล์ CSV ที่ถูกต้องตาม schema A ต้องผ่าน"""
        df = _make_mock_schema_a()  # MOCK DATA
        csv_path = tmp_path / "mock_tep_normal.csv"  # ชื่อระบุชัดว่า mock
        df.to_csv(str(csv_path), index=False)

        validator = TEPValidator()
        result = validator.validate_file(str(csv_path))

        assert result.passed, f"ต้องผ่าน แต่พบ errors: {result.errors}"
        assert result.schema_type == SchemaType.HARVARD_CSV

    def test_empty_csv_file(self, tmp_path):
        """ไฟล์ CSV ว่างเปล่าต้องจัดการได้โดยไม่ crash"""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("", encoding="utf-8")

        validator = TEPValidator()
        result = validator.validate_file(str(empty_csv))

        # อาจผ่านหรือไม่ผ่าน แต่ต้องไม่ raise exception
        assert isinstance(result, ValidationResult)


# ---------------------------------------------------------------------------
# Tests — Schema B (.dat format)
# ---------------------------------------------------------------------------

class TestSchemaBValidation:
    """Tests สำหรับ Schema B (MATLAB .dat format)"""

    def test_schema_b_52_cols_detected(self):
        """DataFrame 52 คอลัมน์ที่มี integer column names ต้องตรวจพบ schema B"""
        df = _make_mock_schema_b(n_cols=52)  # MOCK DATA
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert result.schema_type == SchemaType.MATLAB_DAT, \
            f"ต้องตรวจพบ schema B แต่ได้: {result.schema_type}"

    def test_schema_b_wrong_col_count_fails(self):
        """DataFrame .dat ที่มีคอลัมน์ไม่ใช่ 52 ต้อง fail"""
        df = _make_mock_schema_b(n_cols=30)  # MOCK DATA: เจตนาให้ผิด
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert not result.passed
        assert any("52" in e for e in result.errors), \
            f"ต้องมี error เกี่ยวกับจำนวนคอลัมน์ 52 แต่ได้: {result.errors}"

    def test_schema_b_has_warning_no_metadata(self):
        """Schema B ต้องมี warning ว่าไม่มี simulationRun และ faultNumber"""
        df = _make_mock_schema_b(n_cols=52)  # MOCK DATA
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert any("simulationRun" in w or "faultNumber" in w
                   for w in result.warnings), \
            f"ต้องมี warning เกี่ยวกับ metadata แต่ได้: {result.warnings}"

    def test_schema_b_transposed_file_auto_transpose(self, tmp_path):
        """ไฟล์ .dat รูปแบบ transposed (52 rows × 500 cols) ต้องถูก transpose เป็น (500, 52) และผ่าน"""
        # MOCK DATA: จำลองไฟล์ .dat ที่มี 52 rows และ 100 cols (variables x samples)
        rng = np.random.default_rng(42)
        mock_raw_data = rng.normal(0, 1, (52, 100))
        dat_path = tmp_path / "mock_d00.dat"
        np.savetxt(str(dat_path), mock_raw_data)

        validator = TEPValidator()
        result = validator.validate_file(str(dat_path))

        assert result.passed, f"ต้องผ่าน แต่ได้ errors: {result.errors}"
        assert result.schema_type == SchemaType.MATLAB_DAT
        assert result.info.get("dat_transposed") is True
        assert result.info.get("shape") == (100, 52)


# ---------------------------------------------------------------------------
# Tests — Unknown Schema
# ---------------------------------------------------------------------------

class TestUnknownSchema:
    """Tests สำหรับ schema ที่ไม่รู้จัก"""

    def test_unknown_schema_columns_fails(self):
        """DataFrame ที่มีชื่อคอลัมน์ไม่ตรง pattern ใดต้องได้ UNKNOWN schema และ fail"""
        df = pd.DataFrame({
            "col_a": [1.0, 2.0],
            "col_b": [3.0, 4.0],
        })
        validator = TEPValidator()
        result = validator.validate_dataframe(df)

        assert result.schema_type == SchemaType.UNKNOWN
        assert not result.passed
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Tests — ValidationResult dataclass
# ---------------------------------------------------------------------------

class TestValidationResult:
    """Tests สำหรับ ValidationResult dataclass"""

    def test_summary_shows_passed(self):
        """summary() ต้องแสดง PASSED เมื่อ passed=True"""
        result = ValidationResult(passed=True, schema_type=SchemaType.HARVARD_CSV)
        summary = result.summary()
        assert "PASSED" in summary

    def test_summary_shows_failed(self):
        """summary() ต้องแสดง FAILED เมื่อมี errors"""
        result = ValidationResult(
            passed=False,
            errors=["error 1"],
            schema_type=SchemaType.UNKNOWN,
        )
        summary = result.summary()
        assert "FAILED" in summary

    def test_summary_lists_errors(self):
        """summary() ต้องแสดงรายการ errors"""
        result = ValidationResult(
            passed=False,
            errors=["ไม่พบคอลัมน์ simulationRun"],
            schema_type=SchemaType.UNKNOWN,
        )
        assert "simulationRun" in result.summary()
