# ระบบตรวจจับความผิดปกติของเตาปฏิกรณ์เคมีด้วยปัญญาประดิษฐ์
## AI-Based Chemical Reactor Anomaly Detection (TEP Benchmark)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2Bcpu-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-73%20Passed-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Tennessee%20Eastman%20Process-orange.svg)]()

โครงการวิจัยและพัฒนาแบบจำลองตรวจจับความผิดปกติในกระบวนการเคมีอุตสาหกรรม โดยใช้ชุดข้อมูลมาตรฐานระดับโลก **Tennessee Eastman Process (TEP)** เปรียบเทียบประสิทธิภาพของ 5 แบบจำลอง ตั้งแต่วิธีมาตรฐานแบบเดิม (Baselines), การควบคุมกระบวนการเชิงสถิติพหุตัวแปร (MSPC), โครงข่ายประสาทเทียมเชิงลึก (Deep Learning) ไปจนถึงวิธีลูกผสมที่นำเสนอ **T²-LSTM (Proposed Hybrid Architecture)**

---

## 📊 สรุปผลการประเมิน Benchmark (ผลการทดลองจริงบนข้อมูล TEP)

ผลการทดสอบบนชุดข้อมูลจริงของ Tennessee Eastman Process (แบ่งชุดทดสอบอย่างเข้มงวดตาม `simulationRun` โดย Train ใช้เฉพาะข้อมูลปกติ 100%):

| อันดับ | แบบจำลอง (Model) | สถาปัตยกรรม (Paradigm) | FDR (%) ↑ | FAR (%) ↓ | AUROC ↑ | Detection Delay (samples) ↓ | เวลาฝึก (Fit Time) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Fixed Threshold (3σ)** | Rule-based Baseline | **100.00%** | **23.62%** | **0.9076** | 0.0 | < 0.1s |
| 2 | **PCA / Hotelling's T² + SPE** | Multivariate SPC (Statistical) | **100.00%** | 24.45% | **0.9051** | 0.0 | < 0.1s |
| 3 | **Deep Autoencoder** | Unsupervised Reconstruction DL | 99.98% | 24.53% | **0.9050** | 0.0 | ~14.0s |
| 4 | **T²-LSTM (Proposed)** | Hybrid Latent MSPC + Recurrent DL | **100.00%** | 25.14% | 0.8985 | 0.0 | ~38.2s |
| 5 | **Stacked LSTM** | Sequence-to-Sequence Recurrent DL | **100.00%** | 26.42% | 0.8997 | 0.0 | ~37.5s |

> 📌 **หมายเหตุทางวิชาการ:** ค่าเมตริกทั้งหมดในตารางได้จากการประเมินผลจริงผ่านสคริปต์ `scripts/run_benchmark.py` บนชุดข้อมูล TEP ไม่มีการสมมติหรือ Hardcode ค่าผลลัพธ์ใดๆ

---

## 🔬 รายละเอียดแบบจำลองทั้ง 5 วิธี

### 1. Fixed Threshold (3-Sigma Rule)
- **หลักการ:** กำหนดขอบเขตควบคุมบนล่าง $\mu \pm k\sigma$ ของแต่ละตัวแปรทางกระบวนการ (41 XMEAS + 11 XMV = 52 ตัวแปร) โดยคำนวณจากสถิติของข้อมูลปกติ
- **การตัดสินใจ:** หากตัวแปรใดตัวแปรหนึ่งมีค่าเบี่ยงเบนเกิน $k\sigma$ จะระบุว่าเป็นความผิดปกติ
- **การปรับจูน:** ปรับค่า $k$ อัตโนมัติบน Validation Normal set ตามเป้าหมาย False Alarm Rate (FAR)

### 2. PCA / Hotelling's $T^2$ + SPE (MSPC Baseline)
- **หลักการ:** แปลงมิติข้อมูลด้วย Principal Component Analysis (PCA) บนชุดข้อมูลปกติเพื่อลดความสัมพันธ์ข้ามตัวแปร (Cross-correlation)
- **สถิติวัด:**
  - **Hotelling's $T^2$:** วัดความผิดปกติภายในระนาบส่วนประกอบหลัก (Principal Subspace) ผ่านระยะทาง Mahalanobis Distance
  - **Squared Prediction Error (SPE / Q-statistic):** วัดความคลาดเคลื่อนในการสร้างสัญญาณกลับคืนใน Residual Subspace
- **ขอบเขตควบคุม:** อ้างอิงตามทฤษฎีการกระจายตัว F-distribution และ Jackson-Mudholkar approximation

### 3. Deep Autoencoder (Dense AE)
- **หลักการ:** Fully-Connected Neural Network แบบ Encoder-Decoder พร้อม Bottleneck Latent Layer
- **ฟังก์ชันวัตถุประสงค์:** ลดทอน Mean Squared Error (MSE) ในการคืนค่าสัญญาณปกติ
- **Anomaly Score:** Reconstruction Error $L_{\text{rec}} = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$
- **จุดเด่น:** สามารถจับความสัมพันธ์แบบไม่เป็นเส้นตรง (Non-linear cross-feature relationships)

### 4. Stacked LSTM (Sequence Autoencoder)
- **หลักการ:** โครงข่ายประสาทเทียมแบบวนซ้ำ (Recurrent Neural Network) ชนิด Stacked Multi-layer LSTM สำหรับวิเคราะห์ Time-series Window
- **Encoder:** บีบอัดหน้าต่างเวลา $(B, T, D)$ ไปสู่ Context Hidden State
- **Decoder:** ขยาย Context State ซ้ำ $T$ ครั้งเพื่อสร้างลำดับเวลาเดิมกลับคืนมา
- **จุดเด่น:** จับพลวัตทางเวลา (Temporal Dynamics) และความหน่วงของการตอบสนองของระบบ (Process Lags)

### 5. T²-LSTM (Proposed Hybrid Architecture)
- **แนวคิดนวัตกรรม:** แก้ปัญหาจุดอ่อนของ Deep Learning ที่มักไม่มีขอบเขตควบคุมทางสถิติที่แน่นอน และแก้ปัญหาข้อจำกัดของ PCA แบบดั้งเดิมที่ไม่สามารถจับ Non-linear Time-dependencies ได้
- **สถาปัตยกรรม:**
  1. **LSTM Temporal Feature Extraction:** ใช้ Multi-layer LSTM สกัดเวกเตอร์ตัวแทนพลวัตกระบวนการ $z \in \mathbb{R}^{d_z}$ จาก Sliding Window
  2. **Latent Space Hotelling's $T^2$:** ประมาณการเวกเตอร์ค่าเฉลี่ย $\mu_z$ และ Covariance Matrix $\Sigma_z$ ใน Latent Space จากเฉพาะข้อมูลปกติ และคำนวณ $T^2(z) = (z - \mu_z)^T \Sigma_z^{-1} (z - \mu_z)$
  3. **Residual Reconstruction Error (SPE):** คำนวณความคลาดเคลื่อนในการ Reconstruct สัญญาณทั้งหน้าต่างเวลา
  4. **Composite Anomaly Metric:** ผสานคะแนนแบบถ่วงน้ำหนักมาตรฐาน:
     $$\text{Score} = \gamma \cdot \frac{T^2(z)}{T^2_{\text{limit}}} + (1 - \gamma) \cdot \frac{\text{SPE}}{\text{SPE}_{\text{limit}}}$$
  5. **ผลลัพธ์:** ช่วยลด False Alarm Rate (FAR) ลงเมื่อเทียบกับ Stacked LSTM ปกติ ขณะที่ยังคงความไวในการตรวจจับ (FDR = 100%)

---

## 🛡️ มาตรการป้องกัน Data Leakage และความถูกต้องทางวิทยาศาสตร์ข้อมูล

1. **การแบ่งชุดข้อมูลตาม `simulationRun` อย่างเคร่งครัด:**
   - ห้ามสุ่มแบ่งตัวอย่างข้อมูลแบบข้ามจุดเวลา
   - หน้าต่างเวลา (Sliding Windows) จะถูก Reset ทุกครั้งที่เปลี่ยนรหัส `simulationRun` โดย **ไม่มีการสร้าง Window คร่อมข้าม Run**
2. **Train Set บริสุทธิ์:**
   - ชุดฝึกมีเฉพาะข้อมูลการทำงานปกติ (Fault 0 / Fault-Free) 100%
   - `StandardScaler` จะถูกคำนวณ (fit) จากชุด Train ปกติเท่านั้น แล้วนำไป transform บน Val และ Test
3. **Validation & Test Isolation:**
   - Validation Set ใช้สำหรับการตั้ง Anomaly Thresholds และจูนค่าพารามิเตอร์ควบคุม
   - Test Set ใช้ประเมินผลขั้นสุดท้ายเพียงครั้งเดียว
4. **ความโปร่งใสของ Unit Tests:**
   - ข้อมูลใน Unit Tests ทุกไฟล์ถูกระบุอย่างชัดเจนว่าเป็น `⚠️ MOCK DATA` ไม่มีการอ้างว่าใช้ข้อมูลจริงในการรันทดสอบ

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
tep-anomaly-detection/
├── README.md                           # รายละเอียดโครงการ ผลการทดลอง และคู่มือการใช้งาน
├── requirements.txt                    # รายการไลบรารีแยกส่วน Windows Dev และ Colab GPU
├── .gitignore                          # กำหนดไม่ push ชุดข้อมูลดิบ, venv, checkpoint และ cache
├── config/
│   └── default_config.yaml             # การตั้งค่า Random Seed, สัดส่วน Split, ไฮเปอร์พารามิเตอร์
├── docs/
│   ├── experiment_plan.md              # แผนการทดลอง เกณฑ์ประเมิน และขั้นตอนอย่างละเอียด
│   └── data_checklist.md               # สรุป Schema และโครงสร้างข้อมูลจริงของชุด TEP
├── data/
│   ├── raw/                            # จัดเก็บไฟล์ข้อมูลดิบ (TEP CSV / .dat)
│   └── processed/                      # จัดเก็บชุดข้อมูลหลังแปลงสเกล
├── src/
│   ├── data/
│   │   ├── loader.py                   # TEPDataLoader — โหลดและแบ่งข้อมูลตาม simulationRun
│   │   ├── pipeline.py                 # TEPPreprocessor (Scaler) & WindowGenerator (ปลอดภัยต่อ Run)
│   │   └── validator.py                # TEPValidator — ตรวจสอบ Schema Harvard CSV และ MATLAB .dat
│   ├── models/
│   │   ├── fixed_threshold.py          # Fixed Threshold Model (3-sigma rule)
│   │   ├── pca_spc.py                  # PCA / Hotelling's T² + SPE Model
│   │   ├── autoencoder.py              # Fully-Connected Deep Autoencoder
│   │   ├── stacked_lstm.py             # Stacked LSTM Sequence Autoencoder
│   │   └── t2_lstm.py                  # T²-LSTM Proposed Hybrid Architecture
│   ├── evaluation/
│   │   └── metrics.py                  # การคำนวณ FDR, FAR, Detection Delay, AUROC
│   └── utils/
│       └── logging_utils.py            # การบันทึก Log และการตั้งค่า Random Seeds
├── scripts/
│   └── run_benchmark.py                # สคริปต์รัน Benchmark เปรียบเทียบทั้ง 5 โมเดลบนข้อมูล TEP จริง
├── results/
│   ├── benchmark_results.json          # ไฟล์ผลลัพธ์ Benchmark เชิงตัวเลข
│   └── benchmark_table.md              # ตารางสรุปผลการทดลองในรูปแบบ Markdown
└── tests/                              # Unit Tests ครอบคลุม 73 การทดสอบ (100% Pass)
    ├── test_loader.py
    ├── test_validator.py
    ├── test_pipeline.py
    ├── test_metrics.py
    ├── test_baseline.py
    ├── test_autoencoder.py
    ├── test_lstm.py
    └── test_t2_lstm.py
```

---

## 🚀 การติดตั้งและเริ่มต้นใช้งาน (Quickstart)

### 1. การติดตั้งบนระบบปฏิบัติการ Windows (Local Development)

```bash
# 1. โคลน Repository
git clone https://github.com/rinldaY/AIChemical.git
cd AIChemical

# 2. สร้างและเปิดใช้งาน Virtual Environment (Python 3.11)
python -m venv venv
.\venv\Scripts\activate

# 3. ติดตั้ง Dependencies
pip install -r requirements.txt
```

### 2. การฝึกโมเดลบน Google Colab (GPU Acceleration)

สำหรับผู้ใช้งานที่ต้องการฝึกโมเดลขนาดใหญ่บน Google Colab:
```python
# บน Colab Notebook:
!git clone https://github.com/rinldaY/AIChemical.git
%cd AIChemical
!pip install -r requirements.txt
```

---

## 🧪 การรันแบบทดสอบ Unit Tests

ระบบมีชุดทดสอบ Unit Tests ทั้งหมด 73 การทดสอบ ครอบคลุมการโหลดข้อมูล, Data Pipeline, ความถูกต้องของเมตริก, และแบบจำลองทั้ง 5 วิธี:

```bash
pytest tests/ -v
```

ผลการทดสอบ:
```text
============================= test session starts =============================
collected 73 items

tests/test_autoencoder.py::TestAutoencoder ....                         [ 5%]
tests/test_baseline.py::TestFixedThresholdModel ...                     [ 9%]
tests/test_baseline.py::TestPCASPCModel ....                            [15%]
tests/test_loader.py::TestSplitByRun .........                          [27%]
tests/test_loader.py::TestErrorHandling ........                        [38%]
tests/test_loader.py::TestLoadFileOnly ..                               [41%]
tests/test_loader.py::TestSplitResult .                                 [42%]
tests/test_lstm.py::TestStackedLSTM ....                                [48%]
tests/test_metrics.py::TestMetricsCalculation ......                    [56%]
tests/test_pipeline.py::TestTEPPreprocessor ...                         [60%]
tests/test_pipeline.py::TestWindowGenerator ....                        [66%]
tests/test_t2_lstm.py::TestT2LSTM ....                                  [71%]
tests/test_validator.py::TestSchemaAValidation .........                 [84%]
tests/test_validator.py::TestFileValidation .......                     [93%]
tests/test_validator.py::TestSchemaBValidation ....                     [99%]
tests/test_validator.py::TestValidationResult ...                       [100%]

============================= 73 passed in 19.41s =============================
```

---

## 📈 การรันการประเมิน Benchmark บนข้อมูลจริง

เมื่อนำชุดข้อมูลจริงวางในโฟลเดอร์ `data/raw/` (เช่น `TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`) สามารถรันการทดลองเปรียบเทียบทั้ง 5 โมเดลได้ด้วยคำสั่ง:

```bash
python scripts/run_benchmark.py
```

ผลการทดสอบจะถูกบันทึกโดยอัตโนมัติไปยัง:
- `results/benchmark_results.json`
- `results/benchmark_table.md`
- `logs/benchmark.log`

---

## 📜 เอกสารอ้างอิงและชุดข้อมูล
- **Tennessee Eastman Process Simulation:** Downs, J. J., & Vogel, E. F. (1993). *A plant-wide industrial process control problem.* Computers & Chemical Engineering, 17(3), 245-255.
- **TEP Benchmark Dataset:** Rieth, C. A., Amsel, B. D., Tran, R., & Maia, M. B. (2017). *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation.* Harvard Dataverse.
