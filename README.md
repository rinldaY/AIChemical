<<<<<<< HEAD
# AIChemical
=======
# ระบบตรวจจับความผิดปกติของเตาปฏิกรณ์เคมีด้วยปัญญาประดิษฐ์
## AI-Based Chemical Reactor Anomaly Detection (TEP Benchmark)

> **สถานะโครงงาน:** ขั้นตอนที่ 1 — โครงสร้างโปรเจกต์และ Data Loader (ยังไม่มีข้อมูล TEP จริง)

---

## ภาพรวม

โครงงานนี้เปรียบเทียบวิธีตรวจจับความผิดปกติ 5 วิธีบนชุดข้อมูล **Tennessee Eastman Process (TEP)**:

| วิธี | ประเภท | หมายเหตุ |
|------|--------|---------|
| Fixed Threshold | Rule-based Baseline | ใช้ค่า threshold คงที่ต่อตัวแปร |
| PCA + Hotelling's T² + SPE | Statistical | Multivariate SPC แบบดั้งเดิม |
| Autoencoder | Deep Learning | Unsupervised reconstruction error |
| Stacked LSTM | Deep Learning | Sequential anomaly detection |
| T²-LSTM (Proposed) | Hybrid | รวม T² statistic กับ LSTM |

---

## โครงสร้างโฟลเดอร์

```
tep-anomaly-detection/
├── README.md
├── requirements.txt            # Windows dev + Colab training (แยก section)
├── .gitignore
│
├── docs/
│   ├── experiment_plan.md      # แผนทดลอง, เกณฑ์วัด, ตาราง comparison
│   └── data_checklist.md       # TEP schema ที่รองรับ + checklist ตรวจสอบข้อมูล
│
├── config/
│   └── default_config.yaml     # random seeds, paths, split ratios, logging
│
├── data/
│   ├── raw/                    # [ว่าง] วางไฟล์ TEP ดิบที่นี่
│   └── processed/              # [ว่าง] หลัง preprocessing
│
├── src/
│   ├── data/
│   │   ├── loader.py           # TEPDataLoader — อ่าน, ตรวจสอบ, แบ่งข้อมูล
│   │   └── validator.py        # TEPValidator — ตรวจ schema, dtype, completeness
│   └── utils/
│       └── logging_utils.py    # setup_logger(), seed_everything()
│
└── tests/
    ├── test_loader.py          # unit tests loader (mock data เท่านั้น)
    └── test_validator.py       # unit tests validator (mock data เท่านั้น)
```

---

## การติดตั้ง

### Windows (พัฒนาและทดสอบ)

```bash
# สร้าง virtual environment
python -m venv .venv
.venv\Scripts\activate

# ติดตั้ง dependencies สำหรับพัฒนา
pip install -r requirements.txt
```

### Google Colab (ฝึกโมเดล)

```python
# ใน Colab notebook
!pip install -r requirements.txt
# หรือติดตั้งเฉพาะ training dependencies (ดู requirements.txt section [colab])
```

---

## หลักการแบ่งข้อมูล (ไม่เกิด Data Leakage)

```
ข้อมูลทั้งหมด
├── Train Set    → ข้อมูลปกติ (Fault 0) เท่านั้น | fit scaler, fit PCA, train models
├── Val Set      → ข้อมูลปกติ + ข้อมูลผิดปกติ    | ตั้ง threshold, tune hyperparams
└── Test Set     → ข้อมูลปกติ + ข้อมูลผิดปกติ    | ประเมินขั้นสุดท้าย (แตะครั้งเดียว)
```

**กฎสำคัญ:**
- แบ่งตาม `simulationRun` ห้ามแบ่งกลางไฟล์
- ไม่มีการสุ่มแบ่งหน้าต่างเวลาที่ซ้อนทับกันข้ามชุด
- Scaler fit จาก Train เท่านั้น — transform บน Val และ Test

---

## Roadmap

- [x] **ขั้นตอนที่ 1** — โครงสร้างโปรเจกต์, เอกสาร, Data Loader เบื้องต้น
- [ ] **ขั้นตอนที่ 2** — ตรวจสอบและ validate ไฟล์ข้อมูล TEP จริง
- [ ] **ขั้นตอนที่ 3** — Preprocessing pipeline + EDA
- [ ] **ขั้นตอนที่ 4** — Fixed Threshold + PCA/T²/SPE baseline
- [ ] **ขั้นตอนที่ 5** — Autoencoder
- [ ] **ขั้นตอนที่ 6** — Stacked LSTM
- [ ] **ขั้นตอนที่ 7** — T²-LSTM (Proposed)
- [ ] **ขั้นตอนที่ 8** — Comparison, รายงาน, สรุปผล

---

## เกณฑ์วัด

| เกณฑ์ | ชื่อย่อ | คำอธิบาย |
|-------|--------|---------|
| Fault Detection Rate | FDR | สัดส่วน fault samples ที่ตรวจพบ (= Recall) |
| False Alarm Rate | FAR | สัดส่วน normal samples ที่แจ้งผิด |
| Detection Delay | DD | ค่าเฉลี่ย sample ก่อนตรวจพบ fault |
| AUROC | AUC | Area Under ROC Curve |

> ⚠️ ค่า FDR/FAR ทั้งหมดต้องมาจากการรัน experiment จริงบนข้อมูล TEP เท่านั้น ห้าม hardcode

---

## การอ้างอิง

- Downs, J.J., & Vogel, E.F. (1993). A plant-wide industrial process control problem. *Computers & Chemical Engineering*, 17(3), 245–255.
- Rieth, C.A. et al. (2017). Issues and advances in anomaly detection evaluation for joint human-machine networks. *Proceedings of HFES 2017*.
- Rieth, C.A. et al. (2017). *Additional Tennessee Eastman Process Simulation Data*. Harvard Dataverse. https://doi.org/10.7910/DVN/6C3JR1
>>>>>>> 0839f57 (feat: initial project structure, preprocessing pipeline, baseline models, autoencoder, and unit tests)
