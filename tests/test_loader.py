"""
tests/test_loader.py
--------------------
Unit tests สำหรับ TEPDataLoader

⚠️ IMPORTANT: ข้อมูลทั้งหมดในไฟล์นี้เป็น MOCK DATA ที่สร้างขึ้นเพื่อทดสอบเท่านั้น
   ไม่มีส่วนใดเป็นข้อมูล TEP จริง และห้ามนำผลทดสอบไปอ้างอิงในการทดลองจริง
   การทดสอบนี้ตรวจสอบเฉพาะ logic การทำงานของ loader เท่านั้น

Tests:
    - test_split_by_run_no_leakage:       run IDs ต้องไม่ซ้อนทับข้ามชุด
    - test_train_is_normal_only:          Train set ต้องมีเฉพาะ normal data
    - test_split_ratios_approximate:      สัดส่วน runs ต้องใกล้เคียงกับที่กำหนด
    - test_feature_count:                 จำนวน feature columns ต้องถูกต้อง
    - test_y_val_binary:                  y_val ต้องเป็น binary (0/1)
    - test_file_not_found_raises:         ไม่พบไฟล์ต้อง raise FileNotFoundError
    - test_invalid_schema_raises:         schema ไม่ถูกต้องต้อง raise ValueError
    - test_missing_simulationRun_raises:  ไม่มีคอลัมน์ simulationRun ต้อง raise
    - test_reproducibility:              seed เดิมให้ผลเดิม
    - test_insufficient_runs_raises:      runs น้อยเกินไปต้อง raise ValueError
    - test_load_file_only_invalid_schema: load_file_only กับไฟล์ schema ไม่ถูกต้อง
"""

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import SplitResult, TEPDataLoader
from src.data.validator import SchemaType


# ---------------------------------------------------------------------------
# MOCK DATA helpers
# ⚠️ ข้อมูลด้านล่างทั้งหมดเป็น MOCK DATA ไม่ใช่ข้อมูล TEP จริง
# ---------------------------------------------------------------------------

def _make_mock_normal_df(
    n_runs: int = 20,
    samples_per_run: int = 50,
    n_xmeas: int = 41,
    n_xmv: int = 11,
    seed: int = 42,
) -> pd.DataFrame:
    """
    สร้าง DataFrame จำลอง Normal operation (faultNumber=0)

    ⚠️ MOCK DATA — ค่าเป็น Gaussian noise ไม่มีความหมายทางฟิสิกส์
    """
    rng = np.random.default_rng(seed)
    total = n_runs * samples_per_run
    records = {
        "simulationRun": np.repeat(range(1, n_runs + 1), samples_per_run),
        "sample": np.tile(range(samples_per_run), n_runs),
        "faultNumber": np.zeros(total, dtype=int),
    }
    for i in range(1, n_xmeas + 1):
        records[f"xmeas_{i}"] = rng.normal(0, 1, total)
    for i in range(1, n_xmv + 1):
        records[f"xmv_{i}"] = rng.normal(0, 1, total)
    return pd.DataFrame(records)


def _make_mock_fault_df(
    fault_ids: List[int],
    n_runs_per_fault: int = 10,
    samples_per_run: int = 50,
    n_xmeas: int = 41,
    n_xmv: int = 11,
    seed: int = 99,
) -> pd.DataFrame:
    """
    สร้าง DataFrame จำลอง Fault data

    ⚠️ MOCK DATA — ค่าเป็น Gaussian noise บวก offset เล็กน้อย
       ไม่ใช่พฤติกรรม fault จริงของ TEP
    """
    rng = np.random.default_rng(seed)
    dfs = []
    run_start = 1000  # เริ่ม run ID ต่างจาก normal เพื่อไม่ชน

    for fid in fault_ids:
        total = n_runs_per_fault * samples_per_run
        rec = {
            "simulationRun": np.repeat(
                range(run_start, run_start + n_runs_per_fault), samples_per_run
            ),
            "sample": np.tile(range(samples_per_run), n_runs_per_fault),
            "faultNumber": np.full(total, fid, dtype=int),
        }
        for i in range(1, n_xmeas + 1):
            rec[f"xmeas_{i}"] = rng.normal(1, 1, total)  # offset เล็กน้อย (mock)
        for i in range(1, n_xmv + 1):
            rec[f"xmv_{i}"] = rng.normal(0, 1, total)
        dfs.append(pd.DataFrame(rec))
        run_start += n_runs_per_fault

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests — Data Split Logic
# ---------------------------------------------------------------------------

class TestSplitByRun:
    """Tests ตรวจสอบ logic การแบ่งข้อมูลตาม simulationRun"""

    def test_split_run_ids_no_overlap(self):
        """run IDs ต้องไม่ซ้อนทับระหว่าง Train, Val, Test — ป้องกัน data leakage"""
        normal_df = _make_mock_normal_df(n_runs=30)  # MOCK DATA
        loader = TEPDataLoader(random_seed=42)
        result = loader._split_by_run(
            normal_df,
            fault_df=pd.DataFrame(),
        )
        train_set = set(result.run_ids_train)
        val_set = set(result.run_ids_val)
        test_set = set(result.run_ids_test)

        assert train_set.isdisjoint(val_set), \
            f"Train และ Val มี run ซ้อนทับ: {train_set & val_set}"
        assert train_set.isdisjoint(test_set), \
            f"Train และ Test มี run ซ้อนทับ: {train_set & test_set}"
        assert val_set.isdisjoint(test_set), \
            f"Val และ Test มี run ซ้อนทับ: {val_set & test_set}"

    def test_train_contains_normal_data_only(self):
        """X_train ต้องมาจาก normal data (faultNumber=0) เท่านั้น"""
        normal_df = _make_mock_normal_df(n_runs=20)  # MOCK DATA
        fault_df = _make_mock_fault_df([1, 2])       # MOCK DATA
        loader = TEPDataLoader(random_seed=42)
        result = loader._split_by_run(normal_df, fault_df)

        # Train rows ต้องมาจาก normal_df ที่มี faultNumber=0
        # ตรวจสอบโดยนับ rows: X_train.shape[0] ต้องเท่ากับ normal rows ใน train runs
        n_train_runs = len(result.run_ids_train)
        samples_per_run = 50  # ตาม mock
        expected_rows = n_train_runs * samples_per_run
        assert result.X_train.shape[0] == expected_rows, \
            f"X_train มี {result.X_train.shape[0]} rows แต่คาด {expected_rows}"

    def test_split_ratios_approximate(self):
        """สัดส่วน runs ต้องใกล้เคียงกับ train_normal_ratio=0.70"""
        normal_df = _make_mock_normal_df(n_runs=100)  # MOCK DATA
        loader = TEPDataLoader(train_normal_ratio=0.70, val_normal_ratio=0.15, random_seed=42)
        result = loader._split_by_run(normal_df, pd.DataFrame())

        total_runs = len(result.run_ids_train) + len(result.run_ids_val) + len(result.run_ids_test)
        assert total_runs == 100, f"รวม runs ต้องได้ 100 แต่ได้ {total_runs}"

        train_ratio = len(result.run_ids_train) / total_runs
        assert 0.60 <= train_ratio <= 0.80, \
            f"train ratio {train_ratio:.2f} ต้องอยู่ระหว่าง 0.60–0.80"

    def test_feature_count_correct(self):
        """จำนวน features ต้องเท่ากับ 52 (41 XMEAS + 11 XMV)"""
        normal_df = _make_mock_normal_df(n_runs=20)  # MOCK DATA
        loader = TEPDataLoader(random_seed=42)
        result = loader._split_by_run(normal_df, pd.DataFrame())

        assert result.X_train.shape[1] == 52, \
            f"ต้องมี 52 features แต่ได้ {result.X_train.shape[1]}"

    def test_y_val_is_binary(self):
        """y_val ต้องเป็น binary array (0=normal, 1=fault)"""
        normal_df = _make_mock_normal_df(n_runs=20)  # MOCK DATA
        fault_df = _make_mock_fault_df([1, 2, 3])    # MOCK DATA
        loader = TEPDataLoader(random_seed=42)
        result = loader._split_by_run(normal_df, fault_df)

        unique_labels = set(result.y_val.tolist())
        assert unique_labels.issubset({0, 1}), \
            f"y_val ต้องมีเฉพาะ 0 และ 1 แต่พบ: {unique_labels}"

    def test_y_val_has_both_classes_when_fault_exists(self):
        """y_val ต้องมีทั้ง class 0 และ 1 เมื่อมี fault data"""
        normal_df = _make_mock_normal_df(n_runs=20)  # MOCK DATA
        fault_df = _make_mock_fault_df([1])          # MOCK DATA
        loader = TEPDataLoader(random_seed=42)
        result = loader._split_by_run(normal_df, fault_df)

        assert 0 in result.y_val, "y_val ต้องมี class 0 (normal)"
        assert 1 in result.y_val, "y_val ต้องมี class 1 (fault)"

    def test_reproducibility_same_seed(self):
        """seed เดิมต้องให้การแบ่ง run เดิม"""
        normal_df = _make_mock_normal_df(n_runs=30)  # MOCK DATA
        loader1 = TEPDataLoader(random_seed=42)
        loader2 = TEPDataLoader(random_seed=42)
        result1 = loader1._split_by_run(normal_df, pd.DataFrame())
        result2 = loader2._split_by_run(normal_df, pd.DataFrame())

        assert result1.run_ids_train == result2.run_ids_train, \
            "seed เดิมต้องให้ run_ids_train เดิม"
        assert result1.run_ids_val == result2.run_ids_val

    def test_different_seeds_give_different_splits(self):
        """seed ต่างกันต้องให้การแบ่งที่ต่างกัน (ส่วนใหญ่)"""
        normal_df = _make_mock_normal_df(n_runs=50)  # MOCK DATA
        loader1 = TEPDataLoader(random_seed=0)
        loader2 = TEPDataLoader(random_seed=999)
        result1 = loader1._split_by_run(normal_df, pd.DataFrame())
        result2 = loader2._split_by_run(normal_df, pd.DataFrame())

        # ความน่าจะเป็นที่จะเหมือนกันสำหรับ 50 runs แทบเป็นศูนย์
        assert result1.run_ids_train != result2.run_ids_train, \
            "seed ต่างกันควรให้ splits ต่างกัน"


# ---------------------------------------------------------------------------
# Tests — Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests ตรวจสอบการจัดการข้อผิดพลาด"""

    def test_file_not_found_raises(self):
        """ไฟล์ไม่มีอยู่ต้อง raise FileNotFoundError"""
        loader = TEPDataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_and_split("data/raw/this_does_not_exist.csv")

    def test_invalid_schema_file_raises(self, tmp_path):
        """ไฟล์ที่ schema ไม่ถูกต้องต้อง raise ValueError"""
        # สร้าง CSV ที่ schema ไม่ถูกต้อง
        bad_csv = tmp_path / "bad_schema.csv"
        pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(
            str(bad_csv), index=False
        )
        loader = TEPDataLoader()
        with pytest.raises(ValueError, match=".*"):
            loader.load_and_split(str(bad_csv))

    def test_missing_simulationRun_column_raises(self):
        """ไม่มีคอลัมน์ simulationRun ต้อง raise ValueError"""
        normal_df = _make_mock_normal_df(n_runs=10)  # MOCK DATA
        normal_df = normal_df.drop(columns=["simulationRun"])
        loader = TEPDataLoader()
        with pytest.raises(ValueError, match="simulationRun"):
            loader._check_sufficient_data(normal_df, pd.DataFrame())

    def test_insufficient_runs_raises(self):
        """normal runs น้อยกว่า 3 ต้อง raise ValueError"""
        normal_df = _make_mock_normal_df(n_runs=2)  # MOCK DATA: เจตนาให้น้อย
        loader = TEPDataLoader()
        with pytest.raises(ValueError, match=".*runs.*"):
            loader._check_sufficient_data(normal_df, pd.DataFrame())

    def test_unsupported_file_format(self, tmp_path):
        """ไฟล์ .RData ต้อง raise ValueError ที่อธิบาย format ที่ไม่รองรับ"""
        fake = tmp_path / "test.RData"
        fake.write_bytes(b"fake")
        loader = TEPDataLoader()
        with pytest.raises((FileNotFoundError, ValueError)):
            loader.load_and_split(str(fake))

    def test_invalid_train_ratio_raises(self):
        """train_normal_ratio ที่ผิดปกติต้อง raise ValueError ตอน init"""
        with pytest.raises(ValueError):
            TEPDataLoader(train_normal_ratio=0.0)

        with pytest.raises(ValueError):
            TEPDataLoader(train_normal_ratio=1.5)

    def test_invalid_val_ratio_raises(self):
        """val_normal_ratio ที่ผิดปกติต้อง raise ValueError ตอน init"""
        with pytest.raises(ValueError):
            TEPDataLoader(val_normal_ratio=-0.1)


# ---------------------------------------------------------------------------
# Tests — load_file_only
# ---------------------------------------------------------------------------

class TestLoadFileOnly:
    """Tests สำหรับ load_file_only method"""

    def test_load_valid_csv_returns_df(self, tmp_path):
        """ไฟล์ CSV ที่ถูกต้องต้องคืน DataFrame และ result.passed=True"""
        df = _make_mock_normal_df(n_runs=5)  # MOCK DATA
        csv_path = tmp_path / "mock_normal.csv"
        df.to_csv(str(csv_path), index=False)

        loader = TEPDataLoader()
        result_df, validation = loader.load_file_only(str(csv_path))

        assert not result_df.empty, "ต้องคืน DataFrame ที่ไม่ว่าง"
        assert validation.passed

    def test_load_invalid_schema_returns_empty_df(self, tmp_path):
        """ไฟล์ schema ไม่ถูกต้องต้องคืน DataFrame ว่างและ passed=False"""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(str(bad_csv), index=False)

        loader = TEPDataLoader()
        result_df, validation = loader.load_file_only(str(bad_csv))

        assert result_df.empty
        assert not validation.passed


# ---------------------------------------------------------------------------
# Tests — SplitResult
# ---------------------------------------------------------------------------

class TestSplitResult:
    """Tests สำหรับ SplitResult dataclass"""

    def test_summary_contains_shape(self):
        """summary() ต้องแสดง shape ของ X_train, X_val, X_test"""
        result = SplitResult(
            X_train=np.zeros((100, 52)),
            X_val=np.zeros((30, 52)),
            X_test=np.zeros((30, 52)),
            y_val=np.zeros(30),
            y_test=np.zeros(30),
        )
        summary = result.summary()
        assert "100" in summary
        assert "Train" in summary
