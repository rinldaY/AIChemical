# แผนการทดลอง — ระบบตรวจจับความผิดปกติของเตาปฏิกรณ์เคมีด้วยปัญญาประดิษฐ์

> **เวอร์ชัน:** 0.1 (ขั้นตอนที่ 1 — ยังไม่มีข้อมูล TEP จริง)
> **อัปเดตล่าสุด:** 2026-10-23

---

## 1. วัตถุประสงค์

เปรียบเทียบประสิทธิภาพการตรวจจับความผิดปกติ 5 วิธีบนชุดข้อมูล Tennessee Eastman Process (TEP) โดยวัดจาก FDR, FAR, Detection Delay และ AUROC บน **Test Set เท่านั้น** (ห้ามดูผล Test Set ก่อนขั้นตอนสุดท้าย)

---

## 2. วิธีที่เปรียบเทียบ

### 2.1 Fixed Threshold (Baseline)
- ตั้ง threshold ต่อตัวแปรด้วย ±3σ จาก Train Set (normal data)
- แจ้งเตือนเมื่อค่าใดค่าหนึ่งเกิน threshold
- **ข้อจำกัด:** ไม่คำนึงถึง correlation ระหว่างตัวแปร

### 2.2 PCA + Hotelling's T² + SPE
- Fit PCA จาก Train Set (normal data)
- คำนวณ T² และ SPE จาก principal components ที่เลือก
- ตั้ง threshold ที่ความเชื่อมั่น 95% และ 99% จาก Val Set
- **เกณฑ์เลือก PC:** อธิบาย variance ≥ 85% หรือ Scree plot

### 2.3 Autoencoder
- Architecture: encoder-decoder fully connected
- ฝึกบน normal data (Train Set) เท่านั้น
- Anomaly score = reconstruction error (MSE per sample)
- Threshold จาก percentile บน Val Set

### 2.4 Stacked LSTM
- Input: sliding window ความยาว `window_size` samples
- ฝึก sequence reconstruction บน normal data
- Anomaly score = reconstruction error บน window
- **ข้อควรระวัง:** window ต้องไม่ซ้อนทับข้ามชุด Train/Val/Test

### 2.5 T²-LSTM (Proposed Hybrid)
- ใช้ LSTM encoder สกัด hidden state
- คำนวณ Hotelling's T² จาก hidden states ใน latent space
- รวม T² score กับ reconstruction error เป็น anomaly score รวม
- **สมมติฐาน:** latent space มีโครงสร้าง multivariate Gaussian

---

## 3. ชุดข้อมูล Tennessee Eastman Process (TEP)

### 3.1 แหล่งที่มา (ยังไม่ยืนยัน — ต้องตรวจสอบ)
| แหล่ง | รูปแบบ | Faults | หมายเหตุ |
|-------|--------|--------|---------|
| Downs & Vogel (1993) | `.dat` MATLAB | IDV(1)–IDV(20) | ต้นฉบับ |
| Rieth et al. (2017) Harvard Dataverse | `.csv` | IDV(0)–IDV(20) | มี simulationRun |
| Kaggle | `.csv` | IDV(0)–IDV(20) | อาจต่างจาก Harvard |

### 3.2 Fault IDs ที่ต้องการ
- **IDV(0):** Normal operation (ใช้เป็น Train)
- **IDV(1)–IDV(20):** Fault conditions (ใช้ใน Val และ Test)
- **IDV(3), IDV(9), IDV(15):** อาจยากตรวจจับ (documented in literature)

---

## 4. การแบ่งข้อมูล (ป้องกัน Data Leakage)

### 4.1 หลักการ
```
แบ่งตาม simulationRun (ไม่แบ่งกลาง run)
ไม่ใช้การสุ่ม random split บนหน้าต่างเวลา
```

### 4.2 สัดส่วนที่วางแผน (ปรับได้หลังดูข้อมูลจริง)

| ชุด | Normal Runs | Fault Runs | วัตถุประสงค์ |
|-----|-------------|------------|------------|
| Train | 70% ของ run ปกติ | ไม่มี | fit scaler, fit PCA, train models |
| Validation | 15% ของ run ปกติ | 50% ของแต่ละ fault | ตั้ง threshold, tune hyperparams |
| Test | 15% ของ run ปกติ | 50% ของแต่ละ fault | ประเมินขั้นสุดท้าย |

> [!IMPORTANT]
> สัดส่วนข้างต้นเป็นแผนเบื้องต้น จะปรับตามจำนวน run จริงหลังดูข้อมูล
> Scaler (StandardScaler) fit จาก Train Set เท่านั้น

### 4.3 การสร้าง Sliding Window (สำหรับ LSTM)
- Window ต้องอยู่ภายใน run เดียวกัน (ไม่ข้าม run)
- ไม่มี window ที่ซ้อนทับข้ามชุด Train/Val/Test
- บันทึก `(run_id, start_idx)` สำหรับ reproducibility

---

## 5. เกณฑ์วัดประสิทธิภาพ

| เกณฑ์ | สูตร | หน่วย | เป้าหมายขั้นต่ำ (TBD) |
|-------|------|-------|----------------------|
| FDR (Fault Detection Rate) | TP / (TP + FN) | % | — (จะกำหนดหลังเห็นผล baseline) |
| FAR (False Alarm Rate) | FP / (FP + TN) | % | — |
| Detection Delay | mean(first detected sample - fault start) | samples | — |
| AUROC | Area under ROC | — | — |

> [!WARNING]
> ห้าม hardcode หรือกำหนดเป้าหมาย FDR/FAR ก่อนเห็นผลจากข้อมูลจริง
> เป้าหมายจะกำหนดหลังจากได้ baseline Fixed Threshold และ PCA แล้ว

---

## 6. Random Seeds และ Reproducibility

- Global seed: `42` (กำหนดใน `config/default_config.yaml`)
- บันทึก seed และ config ทุก experiment run ใน `outputs/<timestamp>/config_used.yaml`
- บันทึก library versions ทุก run

---

## 7. Checkpoints และ Acceptance Criteria

| ขั้นตอน | เกณฑ์ผ่าน |
|---------|----------|
| 1. โครงสร้างโปรเจกต์ | `pytest tests/` ผ่านทุก test, ไฟล์ครบ 12 รายการ |
| 2. ตรวจสอบข้อมูล | validator รายงาน schema ถูกต้อง, ไม่มี missing column |
| 3. Preprocessing | scaler fit จาก Train เท่านั้น, ไม่มี leakage |
| 4. Baseline | Fixed Threshold และ PCA ให้ผลที่สมเหตุสมผล |
| 5–7. Deep Learning | ผลบน Val Set ใช้ tune threshold ได้ |
| 8. Final | Test Set ถูกแตะครั้งเดียวเท่านั้น |

---

## 8. สิ่งที่ยังไม่รู้และต้องตรวจสอบ

- [ ] รูปแบบ `.csv` ของ TEP ที่จะใช้ (Harvard Dataverse vs. Kaggle vs. อื่น)
- [ ] จำนวน simulationRun ต่อ fault และต่อ normal
- [ ] ความยาวของแต่ละ run (จำนวน sample)
- [ ] ชื่อคอลัมน์จริงในไฟล์ที่จะใช้
- [ ] มี header หรือไม่
- [ ] sampling rate และหน่วยของแต่ละตัวแปร
