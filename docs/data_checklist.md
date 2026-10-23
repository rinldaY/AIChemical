# Data Checklist — Tennessee Eastman Process (TEP)

> **สถานะ:** ยังไม่มีข้อมูล TEP จริง — เอกสารนี้บันทึก schema ที่รองรับ, ความไม่แน่นอน และ checklist ที่ต้องตรวจสอบ

---

## 1. แหล่งข้อมูล TEP ที่รู้จัก

| แหล่ง | URL | รูปแบบ | ความน่าเชื่อถือ |
|-------|-----|--------|--------------|
| Harvard Dataverse (Rieth et al. 2017) | https://doi.org/10.7910/DVN/6C3JR1 | `.csv` | สูง — peer-reviewed |
| Kaggle | https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset | `.csv` | ปานกลาง — อาจแตกต่าง |
| MATLAB original (Downs & Vogel 1993) | ไม่มี URL สาธารณะ | `.dat` | ต้นฉบับ — ต้องแปลงก่อน |
| UCI ML Repository | ตรวจสอบแล้ว — ไม่พบ TEP โดยตรง | — | — |

> [!NOTE]
> แนะนำใช้ **Harvard Dataverse** เนื่องจากมีคอลัมน์ `simulationRun` และ `faultNumber` ชัดเจน

---

## 2. Schema ที่ Data Loader รองรับ

### Schema A — Harvard Dataverse / Rieth et al. (2017) [รองรับหลัก]

**ลักษณะ:**
- ไฟล์แยกตาม fault: `d00.csv` (normal), `d01.csv` ... `d20.csv` (faults)
- หรือไฟล์รวม: `TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`, `TEP_FaultFree_Testing.csv`, `TEP_Faulty_Testing.csv`

**คอลัมน์ที่คาดว่ามี:**
| คอลัมน์ | dtype | คำอธิบาย |
|---------|-------|---------|
| `simulationRun` | int | หมายเลข run (1–500 สำหรับ training) |
| `sample` | int | ลำดับ sample ภายใน run |
| `faultNumber` | int | 0 = normal, 1–20 = fault ID |
| `xmeas_1` – `xmeas_41` | float64 | Process measurements (41 ตัวแปร) |
| `xmv_1` – `xmv_11` | float64 | Manipulated variables (11 ตัวแปร) |

**จำนวนคอลัมน์รวม:** 3 + 41 + 11 = **55 คอลัมน์**

> [!WARNING]
> ชื่อคอลัมน์อาจต่างกัน เช่น `XMEAS(1)` vs `xmeas_1` vs `x1`
> Loader จะทดลอง pattern matching และแจ้งเตือนหากพบชื่อไม่ตรง

---

### Schema B — MATLAB `.dat` format [รองรับแล้ว — Transposed Format]

**ลักษณะ:**
- ไฟล์ text แบบ whitespace-delimited
- ไม่มี header
- เก็บในรูปแบบ transposed: **52 rows (variables) × N columns (samples)** (เช่น 52 × 500 หรือ 52 × 960)
- `TEPValidator` และ `TEPDataLoader` จะทำการ transpose ให้อัตโนมัติเป็น **N samples × 52 variables** (41 XMEAS + 11 XMV)
- **ไม่มี** `simulationRun`, `faultNumber`, `sample` ในตัวไฟล์ — จำเป็นต้องระบุ metadata หรือกำหนด fault ID อิงตามชื่อไฟล์ (เช่น `d00.dat` คือ Normal, `d01.dat`–`d21.dat` คือ Fault 1–21, `*_te.dat` คือ Testing run)

**สถานะการรองรับ:** ✅ รองรับการ transpose อัตโนมัติเป็น (N, 52) แต่ต้องระบุ metadata/fault ID เสริมก่อนเข้าขั้นตอน split run

---

### Schema C — รูปแบบอื่น ๆ ที่อาจพบ [ยังไม่รองรับ]

| รูปแบบ | สถานะ | หมายเหตุ |
|--------|-------|---------|
| `.RData` / `.rda` (R format) | ❌ ยังไม่รองรับ | ต้องแปลงด้วย R หรือ `pyreadr` |
| `.mat` (MATLAB format) | ❌ ยังไม่รองรับ | ต้องใช้ `scipy.io.loadmat` |
| `.xlsx` | ❌ ยังไม่รองรับ | ต้องใช้ `openpyxl` |
| CSV ที่ไม่มี header | ⚠️ รองรับบางส่วน | ต้องกำหนด column names ด้วยตนเอง |

---

## 3. ตัวแปรใน TEP (52 ตัวแปรหลัก)

### Process Measurements (XMEAS) — 41 ตัวแปร

| หมายเลข | คำอธิบาย (อ้างอิง Downs & Vogel 1993) | หน่วย |
|---------|--------------------------------------|------|
| XMEAS(1) | A feed (stream 1) | kscmh |
| XMEAS(2) | D feed (stream 2) | kg/hr |
| XMEAS(3) | E feed (stream 3) | kg/hr |
| XMEAS(4) | A and C feed (stream 4) | kscmh |
| XMEAS(5) | Recycle flow (stream 8) | kscmh |
| XMEAS(6) | Reactor feed rate (stream 6) | kscmh |
| XMEAS(7) | Reactor pressure | kPa |
| XMEAS(8) | Reactor level | % |
| XMEAS(9) | Reactor temperature | °C |
| XMEAS(10) | Purge rate (stream 9) | kscmh |
| XMEAS(11) | Product separator temperature | °C |
| XMEAS(12) | Product separator level | % |
| XMEAS(13) | Product separator pressure | kPa |
| XMEAS(14) | Product separator underflow (stream 10) | m³/hr |
| XMEAS(15) | Stripper level | % |
| XMEAS(16) | Stripper pressure | kPa |
| XMEAS(17) | Stripper underflow (stream 11) | m³/hr |
| XMEAS(18) | Stripper temperature | °C |
| XMEAS(19) | Stripper steam flow | kg/hr |
| XMEAS(20) | Compressor work | kW |
| XMEAS(21) | Reactor cooling water outlet temperature | °C |
| XMEAS(22) | Separator cooling water outlet temperature | °C |
| XMEAS(23)–XMEAS(41) | Composition measurements (delayed) | mol% |

### Manipulated Variables (XMV) — 11 ตัวแปร

| หมายเลข | คำอธิบาย | หน่วย |
|---------|---------|------|
| XMV(1) | D feed flow valve | % |
| XMV(2) | E feed flow valve | % |
| XMV(3) | A feed flow valve | % |
| XMV(4) | A and C feed flow valve | % |
| XMV(5) | Compressor recycle valve | % |
| XMV(6) | Purge valve | % |
| XMV(7) | Separator pot liquid flow valve | % |
| XMV(8) | Stripper liquid product flow valve | % |
| XMV(9) | Stripper steam valve | % |
| XMV(10) | Reactor cooling water flow | m³/hr |
| XMV(11) | Condenser cooling water flow | m³/hr |

---

## 4. Fault Descriptions

| IDV | คำอธิบาย | ประเภท | ระดับความยาก |
|-----|---------|--------|------------|
| IDV(0) | Normal operation | — | — |
| IDV(1) | A/C feed ratio step change | Step | ง่าย |
| IDV(2) | B composition step change | Step | ง่าย |
| IDV(3) | D feed temperature step change | Step | **ยาก** |
| IDV(4) | Reactor cooling water inlet temperature step | Step | ง่าย |
| IDV(5) | Condenser cooling water inlet temperature step | Step | ง่าย |
| IDV(6) | A feed loss | Step | ง่าย |
| IDV(7) | C header pressure loss | Step | ง่าย |
| IDV(8) | A, B, C feed composition | Random variation | ปานกลาง |
| IDV(9) | D feed temperature random variation | Random | **ยาก** |
| IDV(10) | C feed temperature random variation | Random | ปานกลาง |
| IDV(11) | Reactor cooling water inlet temperature random | Random | ปานกลาง |
| IDV(12) | Condenser cooling water inlet temperature random | Random | ปานกลาง |
| IDV(13) | Reaction kinetics slow drift | Slow drift | ปานกลาง |
| IDV(14) | Reactor cooling water valve sticking | Sticking | ปานกลาง |
| IDV(15) | Condenser cooling water valve sticking | Sticking | **ยาก** |
| IDV(16)–IDV(20) | Unknown faults | Unknown | ปานกลาง–ยาก |

---

## 5. Checklist — สิ่งที่ต้องตรวจสอบเมื่อได้ไฟล์ข้อมูล

### 5.1 ตรวจสอบไฟล์
- [ ] ระบุ schema ที่ตรงกับรูปแบบใด (A, B, หรืออื่น)
- [ ] นับจำนวนไฟล์และชื่อไฟล์
- [ ] ตรวจสอบขนาดไฟล์ (ไม่ควรน้อยกว่า 1 MB)

### 5.2 ตรวจสอบโครงสร้าง
- [ ] มี header หรือไม่
- [ ] จำนวนคอลัมน์ตรงกับ schema ที่คาด
- [ ] ชื่อคอลัมน์ตรงกับ pattern ที่รองรับ
- [ ] dtype ของแต่ละคอลัมน์ถูกต้อง (float64 สำหรับ measurements)

### 5.3 ตรวจสอบความสมบูรณ์
- [ ] ไม่มี NaN / missing values ในคอลัมน์หลัก
- [ ] ค่า `faultNumber` อยู่ในช่วง 0–20
- [ ] `simulationRun` ต่อเนื่องและไม่มีช่องว่าง
- [ ] จำนวน samples ต่อ run สม่ำเสมอ

### 5.4 ตรวจสอบสถิติเบื้องต้น
- [ ] ค่า min/max ของแต่ละตัวแปรสมเหตุสมผล (ไม่มี outlier สุดขีด)
- [ ] ไม่มีตัวแปรที่มีค่าคงที่ตลอด (zero variance)
- [ ] Normal data (IDV=0) มีค่าเฉลี่ยคงที่หรือใกล้เคียงกันข้าม run

### 5.5 ตรวจสอบป้ายกำกับ
- [ ] ทุก sample มี `faultNumber` ที่ระบุ
- [ ] fault เริ่มต้นที่ sample ที่ถูกต้องภายใน run
- [ ] บันทึก fault onset time (sample ที่ fault เริ่ม) ต่อ run

---

## 6. สถิติจริงของชุดข้อมูลที่ได้รับการตรวจสอบแล้ว (Harvard Dataverse)

จากการตรวจสอบไฟล์จริงใน `data/raw/` เมื่อ 2026-10-23:

| ไฟล์ | ขนาด | แถวทั้งหมด | Fault IDs | จำนวน Runs ต่อ Fault | Samples ต่อ Run | ข้อสังเกต |
|---|---|---|---|---|---|---|
| `TEP_FaultFree_Training.csv` | 89.25 MB | 250,000 | 0 | 500 runs | 500 | Normal ล้วน เหมาะสำหรับ Train |
| `TEP_FaultFree_Testing.csv` | 171.41 MB | 480,000 | 0 | 500 runs | 960 | Normal ล้วนสำหรับประเมิน FAR |
| `TEP_Faulty_Training.csv` | 1,786 MB | 5,000,000 | 1–20 | 500 runs | 500 | 20 faults × 500 runs = 10,000 runs |
| `TEP_Faulty_Testing.csv` | 3,430 MB | 9,600,000 | 1–20 | 500 runs | 960 | 20 faults × 500 runs = 10,000 runs (Fault เริ่มที่ sample 161) |

- **จำนวนตัวแปรฟีเจอร์:** 52 ตัวแปร (41 XMEAS + 11 XMV)
- **คอลัมน์ Metadata:** 3 คอลัมน์ (`faultNumber`, `simulationRun`, `sample`) รวมทั้งหมด 55 คอลัมน์
- **Missing Values:** 0 ทุกไฟล์
- **การป้องกัน Data Leakage:** การแบ่ง Train / Val / Test ต้องแบ่งระดับ `simulationRun` (เช่น Run 1–350 สำหรับ Train, Run 351–425 สำหรับ Val, Run 426–500 สำหรับ Test) และ fit `StandardScaler` จาก `TEP_FaultFree_Training.csv` เฉพาะชุด Train เท่านั้น
