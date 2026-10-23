"""
scripts/run_benchmark.py
------------------------
สคริปต์รัน Benchmark เปรียบเทียบทั้ง 5 โมเดลบนชุดข้อมูลจริงของ Tennessee Eastman Process (TEP)

โมเดลที่ประเมิน:
    1. Fixed Threshold (3-sigma rule)
    2. PCA / Hotelling's T² + SPE
    3. Deep Autoencoder
    4. Stacked LSTM (Sequence Model)
    5. T²-LSTM (Proposed Hybrid Model)

ข้อกำหนดความถูกต้องทางวิทยาศาสตร์:
    - ใช้ข้อมูลจริงจาก data/raw
    - Split ข้อมูลโดย simulationRun เพื่อป้องกัน Data Leakage
    - ฝึกบน Normal Data เท่านั้น
    - Scaler fit จาก Train Normal เท่านั้น
    - คำนวณ FDR, FAR, Detection Delay, AUROC จากผลการทดลองจริง ไม่มีการ Hardcode ตัวเลข
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.data.loader import TEPDataLoader
from src.data.pipeline import TEPPreprocessor, WindowGenerator
from src.evaluation.metrics import (
    calculate_auroc,
    calculate_detection_delay,
    calculate_far,
    calculate_fdr,
)
from src.models import (
    AutoencoderDetector,
    FixedThresholdModel,
    PCASPCModel,
    StackedLSTMDetector,
    T2LSTMDetector,
)
from src.utils.logging_utils import seed_everything, setup_logger

logger = setup_logger("benchmark", log_dir="logs")


def run_benchmark():
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"เริ่มการรัน Benchmark บนอุปกรณ์: {device}")

    raw_dir = Path("data/raw")
    ff_train_path = raw_dir / "TEP_FaultFree_Training.csv"
    faulty_train_path = raw_dir / "TEP_Faulty_Training.csv"

    if not ff_train_path.exists() or not faulty_train_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล TEP ใน {raw_dir}")

    # 1. โหลดข้อมูลจริง (แบ่ง simulationRun)
    logger.info("กำลังโหลดชุดข้อมูล TEP จริง...")

    # โหลดข้อมูลปกติ (FaultFree): Train 40 runs, Val 15 runs, Test 15 runs
    df_normal = pd.read_csv(ff_train_path)
    train_runs = list(range(1, 41))
    val_normal_runs = list(range(41, 56))
    test_normal_runs = list(range(56, 71))

    df_train_norm = df_normal[df_normal["simulationRun"].isin(train_runs)]
    df_val_norm = df_normal[df_normal["simulationRun"].isin(val_normal_runs)]
    df_test_norm = df_normal[df_normal["simulationRun"].isin(test_normal_runs)]

    feature_cols = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]

    # โหลดข้อมูลความผิดปกติ (Faulty): เลือกรันตัวแทน Fault 1, 2, 4 (รวม 15 runs สำหรับ Test)
    # Fault 1: Step in A/C feed ratio
    # Fault 2: Step in B composition
    # Fault 4: Step in reactor cooling water inlet temp
    faults_to_evaluate = [1, 2, 4]
    
    # อ่านเฉพาะ Fault ที่สนใจ
    chunks = []
    chunksize = 250000
    for chunk in pd.read_csv(faulty_train_path, chunksize=chunksize):
        sub = chunk[chunk["faultNumber"].isin(faults_to_evaluate) & (chunk["simulationRun"].isin(range(1, 6)))]
        if len(sub) > 0:
            chunks.append(sub)
    df_faulty_test = pd.concat(chunks, ignore_index=True)

    logger.info(f"Train Normal: {len(df_train_norm)} samples ({len(train_runs)} runs)")
    logger.info(f"Test Normal:  {len(df_test_norm)} samples ({len(test_normal_runs)} runs)")
    logger.info(f"Test Faulty:  {len(df_faulty_test)} samples (Faults {faults_to_evaluate})")

    # 2. ปรับสเกลข้อมูล (Fit StandardScaler บนชุด Train Normal เท่านั้น)
    preprocessor = TEPPreprocessor(feature_names=feature_cols)
    X_train_raw = df_train_norm[feature_cols].values
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val_norm = preprocessor.transform(df_val_norm[feature_cols].values)

    # รวมชุด Test (Normal + Faulty)
    df_test_combined = pd.concat([df_test_norm, df_faulty_test], ignore_index=True)
    X_test = preprocessor.transform(df_test_combined[feature_cols].values)
    
    # กำหนด Ground Truth: ใน TEP Faulty จะเริ่มเกิดความผิดปกติที่ sample > 160
    y_test = np.zeros(len(df_test_combined), dtype=int)
    is_fault = (df_test_combined["faultNumber"] > 0) & (df_test_combined["sample"] > 160)
    y_test[is_fault] = 1

    test_runs = df_test_combined["simulationRun"].values
    test_fault_nums = df_test_combined["faultNumber"].values
    test_samples = df_test_combined["sample"].values

    logger.info(f"ชุดทดสอบรวม: {len(X_test)} samples (ปกติ: {np.sum(y_test == 0)}, ผิดปกติ: {np.sum(y_test == 1)})")

    # 3. เตรียม Sequence Data สำหรับ LSTM และ T²-LSTM
    window_size = 20
    stride = 2
    wg = WindowGenerator(window_size=window_size, stride=stride)

    win_train = wg.create_windows(
        features=X_train,
        labels=np.zeros(len(X_train), dtype=int),
        run_ids=df_train_norm["simulationRun"].values,
    )
    win_val_norm = wg.create_windows(
        features=X_val_norm,
        labels=np.zeros(len(X_val_norm), dtype=int),
        run_ids=df_val_norm["simulationRun"].values,
    )
    win_test = wg.create_windows(
        features=X_test,
        labels=y_test,
        run_ids=test_runs,
    )

    logger.info(f"Train Windows: {win_train.X.shape}")
    logger.info(f"Test Windows:  {win_test.X.shape}")

    results = {}

    # =========================================================================
    # Model 1: Fixed Threshold (3-sigma)
    # =========================================================================
    logger.info("--- [1/5] ประเมิน Fixed Threshold Model ---")
    m1 = FixedThresholdModel(k=3.0)
    t0 = time.time()
    m1.fit(X_train)
    m1.tune_k_on_val(X_val_norm, target_far=0.05)
    t1_time = time.time() - t0

    m1_scores = m1.score(X_test)
    m1_preds = m1.predict(X_test)
    m1_fdr = calculate_fdr(y_test, m1_preds)
    m1_far = calculate_far(y_test, m1_preds)
    m1_auroc = calculate_auroc(y_test, m1_scores)
    m1_delay = calculate_detection_delay(y_test, m1_preds, test_runs)
    if m1_delay is None:
        m1_delay = 0.0

    results["Fixed Threshold (3σ)"] = {
        "FDR (%)": round(m1_fdr * 100, 2),
        "FAR (%)": round(m1_far * 100, 2),
        "AUROC": round(m1_auroc, 4),
        "Detection Delay (samples)": round(m1_delay, 1),
        "Fit Time (s)": round(t1_time, 2),
    }

    # =========================================================================
    # Model 2: PCA / Hotelling's T² + SPE
    # =========================================================================
    logger.info("--- [2/5] ประเมิน PCA / Hotelling's T² + SPE Model ---")
    m2 = PCASPCModel(n_components=0.85, alpha=0.05)
    t0 = time.time()
    m2.fit(X_train)
    m2.tune_limits_on_val(X_val_norm, target_far=0.05)
    t2_time = time.time() - t0

    m2_scores = m2.score(X_test)
    m2_preds = m2.predict(X_test, decision="combined")
    m2_fdr = calculate_fdr(y_test, m2_preds)
    m2_far = calculate_far(y_test, m2_preds)
    m2_auroc = calculate_auroc(y_test, m2_scores)
    m2_delay = calculate_detection_delay(y_test, m2_preds, test_runs)
    if m2_delay is None:
        m2_delay = 0.0

    results["PCA (T² + SPE)"] = {
        "FDR (%)": round(m2_fdr * 100, 2),
        "FAR (%)": round(m2_far * 100, 2),
        "AUROC": round(m2_auroc, 4),
        "Detection Delay (samples)": round(m2_delay, 1),
        "Fit Time (s)": round(t2_time, 2),
    }

    # =========================================================================
    # Model 3: Deep Autoencoder
    # =========================================================================
    logger.info("--- [3/5] ประเมิน Deep Autoencoder ---")
    m3 = AutoencoderDetector(input_dim=52, latent_dim=12, hidden_dims=[32, 20], threshold_pct=95.0)
    t0 = time.time()
    m3.fit(X_train, X_val_norm, epochs=15, batch_size=256, device=device)
    t3_time = time.time() - t0

    m3_scores = m3.score(X_test)
    m3_preds = m3.predict(X_test)
    m3_fdr = calculate_fdr(y_test, m3_preds)
    m3_far = calculate_far(y_test, m3_preds)
    m3_auroc = calculate_auroc(y_test, m3_scores)
    m3_delay = calculate_detection_delay(y_test, m3_preds, test_runs)
    if m3_delay is None:
        m3_delay = 0.0

    results["Autoencoder"] = {
        "FDR (%)": round(m3_fdr * 100, 2),
        "FAR (%)": round(m3_far * 100, 2),
        "AUROC": round(m3_auroc, 4),
        "Detection Delay (samples)": round(m3_delay, 1),
        "Fit Time (s)": round(t3_time, 2),
    }

    # =========================================================================
    # Model 4: Stacked LSTM
    # =========================================================================
    logger.info("--- [4/5] ประเมิน Stacked LSTM ---")
    m4 = StackedLSTMDetector(input_dim=52, hidden_dim=48, num_layers=2, threshold_pct=95.0)
    t0 = time.time()
    m4.fit(win_train.X, win_val_norm.X, epochs=10, batch_size=128, device=device)
    m4.tune_threshold_on_val(win_val_norm.X, np.zeros(len(win_val_norm.X)), target_far=0.05, device=device)
    t4_time = time.time() - t0

    m4_scores = m4.score(win_test.X, device=device)
    m4_preds = m4.predict(win_test.X, device=device)
    m4_fdr = calculate_fdr(win_test.y, m4_preds)
    m4_far = calculate_far(win_test.y, m4_preds)
    m4_auroc = calculate_auroc(win_test.y, m4_scores)
    m4_delay = calculate_detection_delay(win_test.y, m4_preds, win_test.run_ids)
    if m4_delay is None:
        m4_delay = 0.0

    results["Stacked LSTM"] = {
        "FDR (%)": round(m4_fdr * 100, 2),
        "FAR (%)": round(m4_far * 100, 2),
        "AUROC": round(m4_auroc, 4),
        "Detection Delay (samples)": round(m4_delay, 1),
        "Fit Time (s)": round(t4_time, 2),
    }

    # =========================================================================
    # Model 5: T²-LSTM (Proposed Hybrid)
    # =========================================================================
    logger.info("--- [5/5] ประเมิน T²-LSTM (Proposed Method) ---")
    m5 = T2LSTMDetector(input_dim=52, latent_dim=24, hidden_dim=48, num_layers=2, gamma=0.5, significance=0.05)
    t0 = time.time()
    m5.fit(win_train.X, win_val_norm.X, epochs=10, batch_size=128, device=device)
    m5.tune_threshold_on_val(win_val_norm.X, np.zeros(len(win_val_norm.X)), target_far=0.05, device=device)
    t5_time = time.time() - t0

    m5_scores = m5.score(win_test.X, device=device)
    m5_preds = m5.predict(win_test.X, device=device)
    m5_fdr = calculate_fdr(win_test.y, m5_preds)
    m5_far = calculate_far(win_test.y, m5_preds)
    m5_auroc = calculate_auroc(win_test.y, m5_scores)
    m5_delay = calculate_detection_delay(win_test.y, m5_preds, win_test.run_ids)
    if m5_delay is None:
        m5_delay = 0.0

    results["T²-LSTM (Proposed)"] = {
        "FDR (%)": round(m5_fdr * 100, 2),
        "FAR (%)": round(m5_far * 100, 2),
        "AUROC": round(m5_auroc, 4),
        "Detection Delay (samples)": round(m5_delay, 1),
        "Fit Time (s)": round(t5_time, 2),
    }

    # สรุปและบันทึกผล
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # บันทึก Markdown table
    df_res = pd.DataFrame(results).T
    md_table = df_res.to_markdown()
    with open(out_dir / "benchmark_table.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark Results: Tennessee Eastman Process (TEP)\n\n")
        f.write(md_table + "\n")

    logger.info("บันทึกผลการประเมินลง results/benchmark_results.json และ results/benchmark_table.md เรียบร้อยแล้ว")
    try:
        print("\n" + "=" * 80)
        print("Benchmark Results Summary (Tennessee Eastman Process)")
        print("=" * 80)
        print(df_res.to_string())
        print("=" * 80)
    except UnicodeEncodeError:
        print("\nBenchmark results successfully computed and saved to results/benchmark_results.json")
    return results


if __name__ == "__main__":
    run_benchmark()
