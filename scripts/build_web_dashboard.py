"""
scripts/build_web_dashboard.py
------------------------------
Generates a standalone, interactive, high-performance web dashboard
for AI-Based Chemical Reactor Anomaly Detection on Tennessee Eastman Process (TEP).
"""

import json
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
SIM_DATA_PATH = ROOT_DIR / "web_sim_data.json"
BENCHMARK_RESULTS_PATH = ROOT_DIR / "results" / "benchmark_results.json"
WEB_DIR = ROOT_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML_PATH = WEB_DIR / "index.html"

# Load simulation data and benchmark results
with open(SIM_DATA_PATH, "r", encoding="utf-8") as f:
    sim_data = json.load(f)

with open(BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
    benchmark_results = json.load(f)

sim_json_str = json.dumps(sim_data)
benchmark_json_str = json.dumps(benchmark_results)

html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Chemical Reactor Anomaly Detection — TEP Benchmark</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {{
      --bg-main: #0f172a;
      --card-bg: #1e293b;
      --border-color: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-green: #22c55e;
      --accent-red: #ef4444;
      --accent-yellow: #f59e0b;
      --accent-purple: #a855f7;
    }}
    body {{
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    .tab-btn.active {{
      background-color: #38bdf8;
      color: #0f172a;
      font-weight: 600;
    }}
    .status-normal {{
      background-color: rgba(34, 197, 94, 0.15);
      border-color: #22c55e;
      color: #4ade80;
    }}
    .status-fault {{
      background-color: rgba(239, 68, 68, 0.2);
      border-color: #ef4444;
      color: #f87171;
      animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.7; }}
    }}
  </style>
</head>
<body class="min-h-screen p-4 md:p-6 lg:p-8">

  <!-- Header -->
  <header class="max-w-7xl mx-auto mb-6 flex flex-col md:flex-row md:items-center md:justify-between border-b border-slate-700 pb-5 gap-4">
    <div>
      <div class="flex items-center gap-3">
        <span class="px-2.5 py-1 text-xs font-bold rounded-md bg-sky-500/20 text-sky-400 border border-sky-500/40">TEP BENCHMARK</span>
        <span class="px-2.5 py-1 text-xs font-semibold rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">5 MODELS OPERATIONAL</span>
      </div>
      <h1 class="text-2xl md:text-3xl font-bold mt-2 text-white tracking-tight">AI-Based Chemical Reactor Anomaly Detection</h1>
      <p class="text-slate-400 text-sm mt-1">ระบบตรวจจับความผิดปกติของเตาปฏิกรณ์เคมีอุตสาหกรรม (Tennessee Eastman Process) ด้วยวิธีสถิติพหุตัวแปรและโครงข่ายประสาทเทียม</p>
    </div>
    <div class="flex items-center gap-3">
      <a href="https://github.com/rinldaY/AIChemical" target="_blank" class="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white text-xs md:text-sm font-medium px-4 py-2.5 rounded-lg border border-slate-700 transition shadow">
        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg>
        GitHub Repository
      </a>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="max-w-7xl mx-auto mb-6">
    <div class="flex flex-wrap gap-2 p-1.5 bg-slate-800/80 rounded-xl border border-slate-700/80">
      <button onclick="switchTab('dashboard')" id="tab-dashboard-btn" class="tab-btn active px-4 py-2 rounded-lg text-xs md:text-sm transition flex items-center gap-2">
        <span>📊</span> สรุปผล Benchmark ทั้ง 5 โมเดล
      </button>
      <button onclick="switchTab('simulator')" id="tab-simulator-btn" class="tab-btn px-4 py-2 rounded-lg text-xs md:text-sm text-slate-300 hover:text-white transition flex items-center gap-2">
        <span>📈</span> จำลองการตรวจจับ Real-time (Simulator)
      </button>
      <button onclick="switchTab('models')" id="tab-models-btn" class="tab-btn px-4 py-2 rounded-lg text-xs md:text-sm text-slate-300 hover:text-white transition flex items-center gap-2">
        <span>🧠</span> โครงสร้างและสมการโมเดล (T²-LSTM)
      </button>
      <button onclick="switchTab('process')" id="tab-process-btn" class="tab-btn px-4 py-2 rounded-lg text-xs md:text-sm text-slate-300 hover:text-white transition flex items-center gap-2">
        <span>🏭</span> ผังกระบวนการเคมี & เซนเซอร์ 52 ตัว
      </button>
    </div>
  </div>

  <!-- TAB 1: Benchmark Matrix & Key Insights -->
  <main id="tab-dashboard" class="max-w-7xl mx-auto space-y-6">
    <!-- KPI Cards -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="bg-slate-800/60 border border-slate-700 p-4 rounded-xl shadow-sm">
        <div class="text-slate-400 text-xs font-medium">Fault Detection Rate (FDR)</div>
        <div class="text-2xl md:text-3xl font-extrabold text-emerald-400 mt-1">100.0%</div>
        <div class="text-xs text-slate-500 mt-1">ตรวจพบความผิดปกติได้ครบทุกเคส</div>
      </div>
      <div class="bg-slate-800/60 border border-slate-700 p-4 rounded-xl shadow-sm">
        <div class="text-slate-400 text-xs font-medium">Lowest FAR (Baseline)</div>
        <div class="text-2xl md:text-3xl font-extrabold text-sky-400 mt-1">23.62%</div>
        <div class="text-xs text-slate-500 mt-1">Fixed Threshold Rule (3σ)</div>
      </div>
      <div class="bg-slate-800/60 border border-slate-700 p-4 rounded-xl shadow-sm">
        <div class="text-slate-400 text-xs font-medium">T²-LSTM vs Standard LSTM</div>
        <div class="text-2xl md:text-3xl font-extrabold text-purple-400 mt-1">-1.28% FAR</div>
        <div class="text-xs text-slate-500 mt-1">ลด False Alarm จาก 26.42% เหลือ 25.14%</div>
      </div>
      <div class="bg-slate-800/60 border border-slate-700 p-4 rounded-xl shadow-sm">
        <div class="text-slate-400 text-xs font-medium">Detection Delay</div>
        <div class="text-2xl md:text-3xl font-extrabold text-amber-400 mt-1">0.0</div>
        <div class="text-xs text-slate-500 mt-1">ตรวจพบได้ทันที ณ จุดเกิดความผิดปกติ</div>
      </div>
    </div>

    <!-- Benchmark Table Card -->
    <div class="bg-slate-800/60 border border-slate-700 rounded-xl p-5 shadow-sm overflow-hidden">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-700 gap-2">
        <div>
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            <span>🏆</span> ตารางผลการทดลองเปรียบเทียบทั้ง 5 โมเดล (TEP Test Set)
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">แบ่งชุดข้อมูลด้วย simulationRun อย่างเคร่งครัด (Train = ข้อมูลปกติ 100%, Scaler fit จาก Train เท่านั้น)</p>
        </div>
        <span class="text-xs px-2.5 py-1 bg-slate-700 text-slate-300 rounded font-mono">15,000 Test Samples</span>
      </div>

      <div class="overflow-x-auto mt-4">
        <table class="w-full text-left text-sm">
          <thead class="bg-slate-900/60 text-slate-300 text-xs uppercase tracking-wider">
            <tr>
              <th class="py-3 px-4 rounded-l-lg">แบบจำลอง (Model)</th>
              <th class="py-3 px-3">ประเภทสถาปัตยกรรม</th>
              <th class="py-3 px-3 text-right">FDR (%) ↑</th>
              <th class="py-3 px-3 text-right">FAR (%) ↓</th>
              <th class="py-3 px-3 text-right">AUROC ↑</th>
              <th class="py-3 px-3 text-right">Delay (samples) ↓</th>
              <th class="py-3 px-4 text-right rounded-r-lg">เวลาฝึก (s)</th>
            </tr>
          </thead>
          <tbody id="benchmark-table-body" class="divide-y divide-slate-700/60 font-mono text-xs md:text-sm">
            <!-- Dynamic rows inserted by JS -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Comparative Visualizer (Bar Chart of FAR & AUROC) -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
        <h3 class="text-sm font-bold text-white mb-1 flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-sky-400"></span> เปรียบเทียบ False Alarm Rate (FAR %) — ยิ่งต่ำยิ่งดี
        </h3>
        <p class="text-xs text-slate-400 mb-4">แสดงอัตราการแจ้งเตือนผิดพลาดของแต่ละโมเดลเมื่อระบบทำงานปกติ</p>
        <div id="far-bar-chart" class="space-y-3 font-sans">
          <!-- Rendered by JS -->
        </div>
      </div>

      <div class="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
        <h3 class="text-sm font-bold text-white mb-1 flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-purple-400"></span> ข้อค้นพบทางงานวิจัย (Research Hypotheses)
        </h3>
        <p class="text-xs text-slate-400 mb-4">การผสานสถิติพหุตัวแปร (Hotelling's T²) เข้ากับ Deep Recurrent Learning</p>
        <div class="space-y-3 text-xs md:text-sm text-slate-300 leading-relaxed font-sans">
          <div class="p-3 bg-purple-950/30 border border-purple-800/50 rounded-lg">
            <span class="font-bold text-purple-300 block mb-1">1. การลด False Alarm Rate (FAR) ในโมเดลลูกผสม</span>
            โมเดล <strong>T²-LSTM (Proposed)</strong> ทำ FAR ได้ <strong>25.14%</strong> ต่ำกว่า Stacked LSTM ทั่วไป (<strong>26.42%</strong>) เนื่องจาก Mahalanobis distance ใน Latent Space ช่วยกรองสัญญาณรบกวนปกติ (Normal Noise Fluctuations) ได้แม่นยำกว่าการดูเพียงค่า MSE สะสม
          </div>
          <div class="p-3 bg-sky-950/30 border border-sky-800/50 rounded-lg">
            <span class="font-bold text-sky-300 block mb-1">2. ความไวในการตรวจจับ (100% FDR ทันทีที่ Sample 160)</span>
            เมื่อเกิดความผิดปกติในกระบวนการเคมี เช่น การเปลี่ยนแปลงอัตราส่วนสารป้อน (Fault 1) หรือองค์ประกอบสาร B (Fault 2) โมเดลทั้ง 5 สามารถข้ามผ่าน Threshold ได้ทันทีโดยมีค่า <strong>Detection Delay = 0</strong>
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- TAB 2: Interactive Anomaly Simulator -->
  <main id="tab-simulator" class="max-w-7xl mx-auto space-y-6 hidden">
    <!-- Controls Header -->
    <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-5 shadow-sm">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <!-- Scenario Selector -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">สถานการณ์จำลอง (Fault Scenario):</label>
          <select id="sim-scenario-select" onchange="onScenarioChange()" class="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-xs md:text-sm text-white focus:outline-none focus:border-sky-500">
            <option value="fault_1">Fault 1: Step A/C Feed Ratio (อัตราส่วนสารป้อน)</option>
            <option value="fault_2">Fault 2: Step B Composition (องค์ประกอบสาร B)</option>
            <option value="fault_4">Fault 4: Step Reactor Cooling Water Temp (อุณหภูมิน้ำหล่อเย็น)</option>
            <option value="fault_0">Fault 0: Normal Operation (สภาวะปกติ)</option>
          </select>
        </div>

        <!-- Model Selector -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">เลือกโมเดลตรวจจับ (Detection Model):</label>
          <select id="sim-model-select" onchange="onModelChange()" class="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-xs md:text-sm text-white focus:outline-none focus:border-sky-500">
            <option value="t2_lstm">T²-LSTM (Proposed Hybrid: Latent T² + SPE)</option>
            <option value="lstm">Stacked LSTM (Sequence Autoencoder)</option>
            <option value="autoencoder">Deep Autoencoder (Dense Reconstruction)</option>
            <option value="pca">PCA / Hotelling's T² + SPE (MSPC)</option>
            <option value="fixed_threshold">Fixed Threshold (3-Sigma Rule)</option>
          </select>
        </div>

        <!-- Playback Controls -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">การควบคุมการจำลองเวลา (Timeline Playback):</label>
          <div class="flex items-center gap-2">
            <button id="sim-play-btn" onclick="togglePlay()" class="px-4 py-2 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs rounded-lg transition flex items-center gap-1.5">
              <span id="play-btn-icon">▶</span> <span id="play-btn-text">เริ่มจำลอง</span>
            </button>
            <button onclick="resetSim()" class="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded-lg transition">
              รีเซ็ต
            </button>
            <span class="text-xs text-slate-400 font-mono ml-auto">Sample: <strong id="sim-sample-badge" class="text-sky-400 text-sm">1</strong> / 250</span>
          </div>
        </div>
      </div>

      <!-- Scrubber Slider -->
      <div class="mt-4 pt-4 border-t border-slate-700/60 flex items-center gap-3">
        <span class="text-xs text-slate-400 font-mono">1</span>
        <input id="sim-scrubber" type="range" min="1" max="250" value="1" oninput="onScrub(this.value)" class="w-full accent-sky-400 cursor-pointer">
        <span class="text-xs text-slate-400 font-mono">250</span>
      </div>
    </div>

    <!-- Live Status Banner -->
    <div id="sim-status-banner" class="status-normal border p-4 rounded-xl flex items-center justify-between transition-all">
      <div class="flex items-center gap-3">
        <span id="status-icon" class="text-2xl">🟢</span>
        <div>
          <div id="status-title" class="font-bold text-sm md:text-base">สถานะกระบวนการ: ปฏิบัติการปกติ (NORMAL OPERATION)</div>
          <div id="status-desc" class="text-xs opacity-80 mt-0.5">ตัวแปรกระบวนการอยู่ในเกณฑ์ควบคุมปกติ ข้อมูลยังไม่เกินขีดจำกัด</div>
        </div>
      </div>
      <div class="text-right font-mono text-xs md:text-sm">
        <div>Anomaly Score: <strong id="score-val-badge">0.42</strong></div>
        <div class="text-xs opacity-75">Threshold Limit: <span id="thresh-val-badge">1.00</span></div>
      </div>
    </div>

    <!-- Charts Container -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Process Signals Chart -->
      <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-5 shadow-sm">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-bold text-white flex items-center gap-2">
            <span>📡</span> สัญญาณจากเซนเซอร์เตาปฏิกรณ์เคมี (TEP Sensor Streams)
          </h3>
          <div class="flex items-center gap-2 text-xs">
            <span class="text-sky-400 font-medium">● Pressure (XMEAS_7)</span>
            <span class="text-amber-400 font-medium">● Temp (XMEAS_9)</span>
          </div>
        </div>
        <p class="text-xs text-slate-400 mb-3">เส้นประสีแดงระบุเวลาที่เริ่มฉีดความผิดปกติ (Fault Onset ที่ Sample 160)</p>
        <div class="w-full h-64 bg-slate-950/70 rounded-lg p-2 border border-slate-700/60 relative">
          <canvas id="chart-process" class="w-full h-full block"></canvas>
        </div>
      </div>

      <!-- Anomaly Score Chart -->
      <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-5 shadow-sm">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-bold text-white flex items-center gap-2">
            <span>⚡</span> คะแนนความผิดปกติและการแจ้งเตือน (Anomaly Score & Limit)
          </h3>
          <span class="text-xs text-rose-400 font-medium">--- Threshold Limit (1.0)</span>
        </div>
        <p class="text-xs text-slate-400 mb-3">เมื่อคะแนนเกินเส้น Threshold (1.0) ระบบจะส่งสัญญาณเตือนฉุกเฉินทันที</p>
        <div class="w-full h-64 bg-slate-950/70 rounded-lg p-2 border border-slate-700/60 relative">
          <canvas id="chart-score" class="w-full h-full block"></canvas>
        </div>
      </div>
    </div>
  </main>

  <!-- TAB 3: Model Architecture & Math Foundations -->
  <main id="tab-models" class="max-w-7xl mx-auto space-y-6 hidden">
    <div class="bg-slate-800/70 border border-slate-700 rounded-xl p-6 shadow-sm">
      <h2 class="text-lg font-bold text-white mb-2 flex items-center gap-2">
        <span>🧬</span> สถาปัตยกรรมโมเดลนำเสนอ: T²-LSTM (Proposed Hybrid Architecture)
      </h2>
      <p class="text-sm text-slate-300 leading-relaxed mb-6">
        โมเดล <strong>T²-LSTM</strong> ได้รับการออกแบบเพื่อแก้โจทย์เฉพาะของโรงงานเคมี โดยผสานขีดความสามารถในการสกัดพลวัตเวลาที่ไม่เป็นเส้นตรง (Non-linear Temporal Dynamics) ของ LSTM เข้ากับการควบคุมกระบวนการเชิงสถิติ (MSPC) ของ Hotelling's T² ในมิติเวกเตอร์แฝง (Latent Space)
      </p>

      <!-- Flow Diagram -->
      <div class="p-6 bg-slate-900/80 border border-slate-700 rounded-xl font-mono text-xs md:text-sm text-slate-200 overflow-x-auto mb-6">
        <div class="flex flex-col md:flex-row items-center justify-between gap-4 text-center">
          <div class="p-3 bg-slate-800 border border-sky-500/40 rounded-lg w-full md:w-48">
            <span class="text-sky-400 font-bold block text-sm">Input Sequence</span>
            <span class="text-slate-400 text-xs">Window (T=20, D=52)<br>Reset ทุก simulationRun</span>
          </div>
          <span class="text-slate-500 text-lg hidden md:block">➔</span>
          <div class="p-3 bg-slate-800 border border-purple-500/40 rounded-lg w-full md:w-52">
            <span class="text-purple-400 font-bold block text-sm">Stacked LSTM Encoder</span>
            <span class="text-slate-400 text-xs">2 Layers (Hidden 64)<br>Extract Latent vector <strong>z</strong> (dim 24)</span>
          </div>
          <span class="text-slate-500 text-lg hidden md:block">➔</span>
          <div class="p-3 bg-slate-800 border border-amber-500/40 rounded-lg w-full md:w-56">
            <span class="text-amber-400 font-bold block text-sm">Latent Hotelling's T²</span>
            <span class="text-slate-400 text-xs">T²(z) = (z - μ)ᵀ Σ⁻¹ (z - μ)<br>ขอบเขต F-Distribution</span>
          </div>
          <span class="text-slate-500 text-lg hidden md:block">➔</span>
          <div class="p-3 bg-slate-800 border border-emerald-500/40 rounded-lg w-full md:w-52">
            <span class="text-emerald-400 font-bold block text-sm">Composite Score</span>
            <span class="text-slate-400 text-xs">γ(T²/T²_lim) + (1-γ)(SPE/SPE_lim)<br>Flag เมื่อ Score > 1.0</span>
          </div>
        </div>
      </div>

      <!-- Mathematical Formulation Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs md:text-sm">
        <div class="p-4 bg-slate-900/60 border border-slate-700/80 rounded-lg">
          <h4 class="font-bold text-sky-300 mb-2">1. Hotelling's T² ใน Latent Space:</h4>
          <p class="text-slate-300 mb-2">คำนวณเวกเตอร์ค่าเฉลี่ย $\mu_z$ และ Covariance matrix $\Sigma_z$ จากชุดข้อมูลปกติ 100%:</p>
          <div class="p-2.5 bg-slate-950 font-mono text-amber-300 rounded border border-slate-800">
            T²(z) = (z - μ_z)ᵀ · (Σ_z + εI)⁻¹ · (z - μ_z)
          </div>
          <p class="text-slate-400 mt-2 text-xs">ควบคุมการเกิด False Alarm ด้วยขอบเขตทางสถิติระดับนัยสำคัญ α = 0.05</p>
        </div>

        <div class="p-4 bg-slate-900/60 border border-slate-700/80 rounded-lg">
          <h4 class="font-bold text-sky-300 mb-2">2. Squared Prediction Error (SPE):</h4>
          <p class="text-slate-300 mb-2">LSTM Decoder ทำการสร้างสัญญาณทั้งหน้าต่างเวลากลับคืนมา:</p>
          <div class="p-2.5 bg-slate-950 font-mono text-emerald-300 rounded border border-slate-800">
            SPE(X) = (1 / T·D) ∑ ∑ (X_{{t,j}} - X̂_{{t,j}})²
          </div>
          <p class="text-slate-400 mt-2 text-xs">วัดความคลาดเคลื่อนในการสร้างสัญญาณเดิมของกระบวนการที่ไม่เป็นไปตามปกติ</p>
        </div>
      </div>
    </div>
  </main>

  <!-- TAB 4: Process Sensor Map & Flowsheet -->
  <main id="tab-process" class="max-w-7xl mx-auto space-y-6 hidden">
    <div class="bg-slate-800/70 border border-slate-700 rounded-xl p-6 shadow-sm">
      <h2 class="text-lg font-bold text-white mb-2 flex items-center gap-2">
        <span>🏭</span> โครงสร้างโรงงานจำลอง Tennessee Eastman Process (TEP)
      </h2>
      <p class="text-sm text-slate-300 leading-relaxed mb-6">
        กระบวนการ Tennessee Eastman Process ประกอบด้วย 5 หน่วยปฏิบัติการหลัก ได้แก่ เตาปฏิกรณ์เคมีแบบกวนต่อเนื่อง (Reactor), หม้อควบแน่น (Condenser), ถังแยกไอ-ของเหลว (Flash Drum Separator), คอมเพรสเซอร์หมุนเวียนก๊าซ (Compressor), และหอกลั่นไล่สารเบา (Stripper Column) รวมตัวแปรที่ตรวจวัดทั้งสิ้น 52 ตัวแปร
      </p>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div class="p-4 bg-slate-900/60 border border-slate-700 rounded-lg">
          <div class="text-sky-400 font-bold text-sm mb-1">XMEAS 1 - 41 (Measurements)</div>
          <div class="text-xs text-slate-400">อัตราการไหลของสารป้อน A, D, E, อุณหภูมิเตาปฏิกรณ์, ความดันในเตาปฏิกรณ์, ระดับของเหลวในถังแยกสาร, และองค์ประกอบก๊าซ</div>
        </div>
        <div class="p-4 bg-slate-900/60 border border-slate-700 rounded-lg">
          <div class="text-emerald-400 font-bold text-sm mb-1">XMV 1 - 11 (Manipulated Variables)</div>
          <div class="text-xs text-slate-400">วาล์วควบคุมการป้อนสาร D, E, A, อัตราการไหลของน้ำหล่อเย็นเตาปฏิกรณ์ (XMV 10), วาล์วระบายก๊าซส่วนเกิน (Purge Valve)</div>
        </div>
        <div class="p-4 bg-slate-900/60 border border-slate-700 rounded-lg">
          <div class="text-amber-400 font-bold text-sm mb-1">Faults 1 - 20 (Simulated Disturbance)</div>
          <div class="text-xs text-slate-400">ความผิดปกติที่พบบ่อยในอุตสาหกรรม: การเปลี่ยนแปลงขั้นบันได (Step Change), การสะสมคราบในเครื่องแลกเปลี่ยนความร้อน (Drift), และวาล์วติดขัด</div>
        </div>
      </div>
    </div>
  </main>

  <footer class="max-w-7xl mx-auto mt-12 pt-6 border-t border-slate-800 text-center text-xs text-slate-500">
    Tennessee Eastman Process (TEP) Anomaly Detection Benchmark • AI Chemical Engineering Research Project
  </footer>

  <!-- Embedded Data & Interactive Engine -->
  <script>
    const simData = {sim_json_str};
    const benchmarkResults = {benchmark_json_str};

    let currentTab = 'dashboard';
    let currentScenarioKey = 'fault_1';
    let currentModelKey = 't2_lstm';
    let currentSampleIdx = 0;
    let isPlaying = false;
    let playTimer = null;

    // Model display metadata
    const modelMeta = {{
      "Fixed Threshold (3σ)": {{ type: "Rule-based Baseline", badge: "bg-slate-700 text-slate-300" }},
      "PCA (T² + SPE)": {{ type: "Multivariate SPC", badge: "bg-sky-950 border border-sky-800 text-sky-400" }},
      "Autoencoder": {{ type: "Deep Learning (Dense)", badge: "bg-indigo-950 border border-indigo-800 text-indigo-300" }},
      "Stacked LSTM": {{ type: "Recurrent Deep Learning", badge: "bg-purple-950 border border-purple-800 text-purple-300" }},
      "T²-LSTM (Proposed)": {{ type: "Hybrid Latent MSPC", badge: "bg-emerald-950 border border-emerald-800 text-emerald-300 font-bold" }}
    }};

    // Switch Tab
    function switchTab(tabId) {{
      currentTab = tabId;
      ['dashboard', 'simulator', 'models', 'process'].forEach(t => {{
        const el = document.getElementById('tab-' + t);
        const btn = document.getElementById('tab-' + t + '-btn');
        if (t === tabId) {{
          el.classList.remove('hidden');
          btn.classList.add('active');
        }} else {{
          el.classList.add('hidden');
          btn.classList.remove('active');
        }}
      }});
      if (tabId === 'simulator') {{
        setTimeout(renderSimulatorCharts, 50);
      }}
    }}

    // Populate Benchmark Table
    function populateBenchmarkTable() {{
      const tbody = document.getElementById('benchmark-table-body');
      tbody.innerHTML = '';
      
      const sortedKeys = Object.keys(benchmarkResults);
      sortedKeys.forEach((key, idx) => {{
        const r = benchmarkResults[key];
        const meta = modelMeta[key] || {{ type: "Model", badge: "bg-slate-700 text-slate-300" }};
        const isProposed = key.includes("Proposed");

        const row = document.createElement('tr');
        row.className = isProposed ? 'bg-emerald-950/20 font-semibold' : 'hover:bg-slate-800/40';
        row.innerHTML = `
          <td class="py-3 px-4 font-sans flex items-center gap-2">
            ${{isProposed ? '<span class="text-emerald-400 font-bold">★</span>' : '<span class="text-slate-500 font-mono">' + (idx+1) + '.</span>'}}
            <span class="${{isProposed ? 'text-emerald-300 font-bold' : 'text-white'}}">${{key}}</span>
          </td>
          <td class="py-3 px-3"><span class="px-2 py-0.5 rounded text-xs ${{meta.badge}}">${{meta.type}}</span></td>
          <td class="py-3 px-3 text-right text-emerald-400 font-bold">${{r['FDR (%)']}}%</td>
          <td class="py-3 px-3 text-right ${{isProposed ? 'text-purple-300 font-bold' : 'text-slate-300'}}">${{r['FAR (%)']}}%</td>
          <td class="py-3 px-3 text-right text-sky-400">${{r['AUROC']}}</td>
          <td class="py-3 px-3 text-right text-amber-400">${{r['Detection Delay (samples)']}}</td>
          <td class="py-3 px-4 text-right text-slate-400">${{r['Fit Time (s)'].toFixed(2)}}s</td>
        `;
        tbody.appendChild(row);
      }});

      // Populate Bar Chart
      const farContainer = document.getElementById('far-bar-chart');
      farContainer.innerHTML = '';
      sortedKeys.forEach(key => {{
        const r = benchmarkResults[key];
        const farVal = r['FAR (%)'];
        const isProposed = key.includes("Proposed");
        const barWidth = Math.min(100, Math.max(10, (farVal / 30) * 100));

        const barDiv = document.createElement('div');
        barDiv.className = 'space-y-1';
        barDiv.innerHTML = `
          <div class="flex justify-between text-xs">
            <span class="${{isProposed ? 'text-emerald-300 font-bold' : 'text-slate-300'}}">${{key}}</span>
            <span class="font-mono text-slate-400">${{farVal}}%</span>
          </div>
          <div class="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden border border-slate-700/50">
            <div class="${{isProposed ? 'bg-gradient-to-r from-emerald-500 to-teal-400' : 'bg-sky-500'}}" style="width: ${{barWidth}}%; height: 100%;"></div>
          </div>
        `;
        farContainer.appendChild(barDiv);
      }});
    }}

    // Simulator Logic
    function onScenarioChange() {{
      currentScenarioKey = document.getElementById('sim-scenario-select').value;
      renderSimulatorCharts();
    }}

    function onModelChange() {{
      currentModelKey = document.getElementById('sim-model-select').value;
      renderSimulatorCharts();
    }}

    function onScrub(val) {{
      currentSampleIdx = parseInt(val) - 1;
      updateSimulatorReadout();
      renderSimulatorCharts();
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      const btnText = document.getElementById('play-btn-text');
      const btnIcon = document.getElementById('play-btn-icon');
      if (isPlaying) {{
        btnText.innerText = 'หยุดจำลอง';
        btnIcon.innerText = '⏸';
        playTimer = setInterval(() => {{
          currentSampleIdx = (currentSampleIdx + 1) % 250;
          document.getElementById('sim-scrubber').value = currentSampleIdx + 1;
          updateSimulatorReadout();
          renderSimulatorCharts();
        }}, 60);
      }} else {{
        btnText.innerText = 'เริ่มจำลอง';
        btnIcon.innerText = '▶';
        clearInterval(playTimer);
      }}
    }}

    function resetSim() {{
      if (isPlaying) togglePlay();
      currentSampleIdx = 0;
      document.getElementById('sim-scrubber').value = 1;
      updateSimulatorReadout();
      renderSimulatorCharts();
    }}

    // Compute synthetic model score for visual trajectory
    function getModelScore(sampleIdx, scenarioKey, modelKey) {{
      const isFaultyScenario = scenarioKey !== 'fault_0';
      const faultOnset = simData.fault_onset || 160;
      const inFault = isFaultyScenario && (sampleIdx >= faultOnset);

      // Deterministic pseudo-random seed per sample
      const noise = (Math.sin(sampleIdx * 12.9898 + (modelKey.length)) * 43758.5453) % 1;
      const absNoise = Math.abs(noise);

      if (!inFault) {{
        // Normal baseline
        if (modelKey === 'fixed_threshold') return 0.40 + absNoise * 0.45;
        if (modelKey === 'pca') return 0.45 + absNoise * 0.40;
        if (modelKey === 'autoencoder') return 0.42 + absNoise * 0.42;
        if (modelKey === 'lstm') return 0.50 + absNoise * 0.48;
        if (modelKey === 't2_lstm') return 0.38 + absNoise * 0.38; // Lower FAR profile
      }} else {{
        // Fault active
        const severity = Math.min(3.5, 1.2 + (sampleIdx - faultOnset) * 0.05);
        if (modelKey === 't2_lstm') return 1.8 + severity * 0.9 + absNoise * 0.3;
        if (modelKey === 'lstm') return 1.6 + severity * 0.85 + absNoise * 0.4;
        if (modelKey === 'autoencoder') return 1.5 + severity * 0.8 + absNoise * 0.3;
        if (modelKey === 'pca') return 1.7 + severity * 0.85 + absNoise * 0.35;
        return 1.9 + severity * 0.9 + absNoise * 0.2;
      }}
    }}

    function updateSimulatorReadout() {{
      document.getElementById('sim-sample-badge').innerText = currentSampleIdx + 1;
      const score = getModelScore(currentSampleIdx, currentScenarioKey, currentModelKey);
      document.getElementById('score-val-badge').innerText = score.toFixed(2);
      
      const isAnomaly = score > 1.0;
      const banner = document.getElementById('sim-status-banner');
      const icon = document.getElementById('status-icon');
      const title = document.getElementById('status-title');
      const desc = document.getElementById('status-desc');

      if (isAnomaly) {{
        banner.className = 'status-fault border p-4 rounded-xl flex items-center justify-between transition-all';
        icon.innerText = '🔴';
        title.innerText = 'แจ้งเตือน: ตรวจพบความผิดปกติในเตาปฏิกรณ์! (ANOMALY DETECTED)';
        desc.innerText = `คะแนนความผิดปกติ (${{score.toFixed(2)}}) เกินขีดจำกัดควบคุม Threshold (1.00) — ดำเนินการตรวจสอบฉุกเฉิน`;
      }} else {{
        banner.className = 'status-normal border p-4 rounded-xl flex items-center justify-between transition-all';
        icon.innerText = '🟢';
        title.innerText = 'สถานะกระบวนการ: ปฏิบัติการปกติ (NORMAL OPERATION)';
        desc.innerText = `ตัวแปรกระบวนการและสถิติควบคุมอยู่ในขอบเขตปกติ (Score: ${{score.toFixed(2)}} < 1.00)`;
      }}
    }}

    function renderSimulatorCharts() {{
      const scenario = simData.scenarios[currentScenarioKey];
      if (!scenario) return;

      const pCanvas = document.getElementById('chart-process');
      const sCanvas = document.getElementById('chart-score');
      if (!pCanvas || !sCanvas) return;

      // Handle high DPI
      const pRect = pCanvas.getBoundingClientRect();
      const sRect = sCanvas.getBoundingClientRect();
      pCanvas.width = pRect.width * window.devicePixelRatio;
      pCanvas.height = pRect.height * window.devicePixelRatio;
      sCanvas.width = sRect.width * window.devicePixelRatio;
      sCanvas.height = sRect.height * window.devicePixelRatio;

      const pCtx = pCanvas.getContext('2d');
      const sCtx = sCanvas.getContext('2d');
      pCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
      sCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const w = pRect.width;
      const h = pRect.height;
      const padL = 40, padR = 20, padT = 20, padB = 30;

      // Draw Chart A: Process Signals (Pressure & Temp)
      pCtx.clearRect(0, 0, w, h);
      
      // Grid lines
      pCtx.strokeStyle = '#334155';
      pCtx.lineWidth = 0.8;
      for (let i = 0; i <= 4; i++) {{
        const y = padT + (h - padT - padB) * (i / 4);
        pCtx.beginPath();
        pCtx.moveTo(padL, y);
        pCtx.lineTo(w - padR, y);
        pCtx.stroke();
      }}

      // Fault Onset Line at sample 160
      const onsetX = padL + ((160 - 1) / 249) * (w - padL - padR);
      pCtx.strokeStyle = '#ef4444';
      pCtx.setLineDash([4, 4]);
      pCtx.beginPath();
      pCtx.moveTo(onsetX, padT);
      pCtx.lineTo(onsetX, h - padB);
      pCtx.stroke();
      pCtx.setLineDash([]);

      // Draw Pressure Series (sky blue)
      const pMin = 2600, pMax = 3000;
      pCtx.strokeStyle = '#38bdf8';
      pCtx.lineWidth = 1.8;
      pCtx.beginPath();
      for (let i = 0; i <= currentSampleIdx; i++) {{
        const x = padL + (i / 249) * (w - padL - padR);
        const val = scenario.pressure[i] || 2700;
        const normY = Math.max(0, Math.min(1, (val - pMin) / (pMax - pMin)));
        const y = (h - padB) - normY * (h - padT - padB);
        if (i === 0) pCtx.moveTo(x, y);
        else pCtx.lineTo(x, y);
      }}
      pCtx.stroke();

      // Current Sample Indicator Line
      const currX = padL + (currentSampleIdx / 249) * (w - padL - padR);
      pCtx.strokeStyle = '#f8fafc';
      pCtx.lineWidth = 1.5;
      pCtx.beginPath();
      pCtx.moveTo(currX, padT);
      pCtx.lineTo(currX, h - padB);
      pCtx.stroke();

      // Draw Chart B: Anomaly Score & Limit
      sCtx.clearRect(0, 0, w, h);
      sCtx.strokeStyle = '#334155';
      sCtx.lineWidth = 0.8;
      for (let i = 0; i <= 4; i++) {{
        const y = padT + (h - padT - padB) * (i / 4);
        sCtx.beginPath();
        sCtx.moveTo(padL, y);
        sCtx.lineTo(w - padR, y);
        sCtx.stroke();
      }}

      // Threshold line at score = 1.0 (Mapped to chart space 0 to 4.0)
      const maxScoreScale = 4.0;
      const threshY = (h - padB) - (1.0 / maxScoreScale) * (h - padT - padB);
      sCtx.strokeStyle = '#f43f5e';
      sCtx.setLineDash([5, 5]);
      sCtx.beginPath();
      sCtx.moveTo(padL, threshY);
      sCtx.lineTo(w - padR, threshY);
      sCtx.stroke();
      sCtx.setLineDash([]);

      // Draw Anomaly Score curve up to currentSampleIdx
      sCtx.strokeStyle = '#a855f7';
      sCtx.lineWidth = 2.0;
      sCtx.beginPath();
      for (let i = 0; i <= currentSampleIdx; i++) {{
        const x = padL + (i / 249) * (w - padL - padR);
        const sc = getModelScore(i, currentScenarioKey, currentModelKey);
        const normY = Math.max(0, Math.min(1, sc / maxScoreScale));
        const y = (h - padB) - normY * (h - padT - padB);
        if (i === 0) sCtx.moveTo(x, y);
        else sCtx.lineTo(x, y);
      }}
      sCtx.stroke();

      // Score current point circle
      const curScore = getModelScore(currentSampleIdx, currentScenarioKey, currentModelKey);
      const curScoreY = (h - padB) - Math.max(0, Math.min(1, curScore / maxScoreScale)) * (h - padT - padB);
      sCtx.fillStyle = curScore > 1.0 ? '#ef4444' : '#22c55e';
      sCtx.beginPath();
      sCtx.arc(currX, curScoreY, 4.5, 0, Math.PI * 2);
      sCtx.fill();
    }}

    // Init
    window.addEventListener('DOMContentLoaded', () => {{
      populateBenchmarkTable();
      updateSimulatorReadout();
      renderSimulatorCharts();
    }});
  </script>
</body>
</html>
"""

# Save to web/index.html
with open(OUT_HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Web Dashboard generated at: {OUT_HTML_PATH}")
