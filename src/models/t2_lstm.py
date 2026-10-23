"""
src/models/t2_lstm.py
---------------------
T²-LSTM: Hybrid Deep Learning and Statistical Process Control Model
for Chemical Reactor Anomaly Detection

หลักการของโมเดลนำเสนอ (Proposed Method):
    1. ผสานจุดเด่นของ Deep Recurrent Neural Network (LSTM) เข้ากับ
       การควบคุมกระบวนการเชิงสถิติพหุตัวแปร (Multivariate Statistical Process Control: MSPC)
    2. LSTM Encoder: สกัด Non-linear Temporal Dynamics จากลำดับสัญญาณเคมี
       ไปสู่ Latent Representation (z)
    3. LSTM Decoder: ทำการ Reconstruct ลำดับสัญญาณเดิมกลับคืนมา
    4. Hotelling's T² ใน Latent Space:
       - คำนวณ Mean Vector (mu_z) และ Covariance Matrix (Sigma_z) ของเวกเตอร์แฝง
         จากเฉพาะข้อมูลปกติ (Normal Training Set)
       - วัดระยะทาง Mahalanobis Distance: T²(z) = (z - mu_z)^T Sigma_z^{-1} (z - mu_z)
    5. Squared Prediction Error (SPE) ใน Residual Space:
       - วัดความคลาดเคลื่อนในการ Reconstruct สัญญาณ (MSE)
    6. Composite Anomaly Score:
       - ผสาน T² และ SPE แบบถ่วงน้ำหนักมาตรฐาน (Standardized Combined Metric):
         Score = gamma * (T² / T²_limit) + (1 - gamma) * (SPE / SPE_limit)
       - ช่วยลด False Alarm Rate (FAR) ขณะที่ยังคง Fault Detection Rate (FDR) ในระดับสูง
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class T2LSTMNet(nn.Module):
    """
    PyTorch Neural Network สำหรับ T²-LSTM

    Args:
        input_dim:    มิติของฟีเจอร์ (default: 52 สำหรับ TEP)
        latent_dim:   มิติของ Latent Feature Space (default: 32)
        hidden_dim:   ขนาด hidden units ใน LSTM (default: 64)
        num_layers:   จำนวนชั้นของ Stacked LSTM (default: 2)
        dropout:      Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int = 52,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0

        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Latent projection: บีบอัด Context Vector สู่ Latent Space ที่มีขนาดกะทัดรัด
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh(),
        )

        # Decoder projection: ขยาย Latent กลับสู่ Hidden Dim
        self.decoder_proj = nn.Linear(latent_dim, hidden_dim)

        # LSTM Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Reconstructed output
        self.output_linear = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """แปลง sequence ไปยัง latent representation vector z: (Batch, Latent)"""
        _, (h_n, _) = self.encoder(x)
        ctx = h_n[-1]  # (Batch, hidden_dim)
        z = self.latent_proj(ctx)  # (Batch, latent_dim)
        return z

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """สร้างลำดับสัญญาณกลับคืนจาก latent vector z"""
        h_dec = self.decoder_proj(z)  # (Batch, hidden_dim)
        repeated = h_dec.unsqueeze(1).repeat(1, seq_len, 1)  # (Batch, seq_len, hidden_dim)
        dec_out, _ = self.decoder(repeated)
        return self.output_linear(dec_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        Returns:
            reconstructed: (Batch, Seq_len, Features)
            z: Latent Vector (Batch, Latent_dim)
        """
        seq_len = x.size(1)
        z = self.encode(x)
        reconstructed = self.decode(z, seq_len)
        return reconstructed, z


class T2LSTMDetector:
    """
    T²-LSTM Anomaly Detector สำหรับตรวจจับความผิดปกติใน Tennessee Eastman Process

    Args:
        input_dim:     มิติของข้อมูลขาเข้า (52 features)
        latent_dim:    มิติของ Latent Space (default: 32)
        hidden_dim:    ขนาด LSTM Hidden State (default: 64)
        num_layers:    จำนวนชั้น LSTM (default: 2)
        gamma:         น้ำหนักการผสาน (weight สำหรับ T² เทียบกับ SPE, default: 0.5)
        significance:  ระดับนัยสำคัญทางสถิติ alpha สำหรับขีดจำกัดควบคุม (default: 0.05 หรือ 95%)
        random_seed:   Random seed สำหรับการทดลอง (default: 42)
    """

    def __init__(
        self,
        input_dim: int = 52,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        gamma: float = 0.5,
        significance: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gamma = gamma
        self.significance = significance
        self.random_seed = random_seed

        self.model: Optional[T2LSTMNet] = None
        self.latent_mean_: Optional[np.ndarray] = None
        self.latent_cov_inv_: Optional[np.ndarray] = None
        self.t2_limit_: float = 1.0
        self.spe_limit_: float = 1.0
        self.threshold_: float = 1.0

        self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.is_fitted: bool = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def _set_seeds(self) -> None:
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)

    def fit(
        self,
        X_train_normal: np.ndarray,
        X_val_normal: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> "T2LSTMDetector":
        """
        ฝึกโครงข่าย T²-LSTM และคำนวณสถิติพหุตัวแปรบนชุดข้อมูลปกติ

        Args:
            X_train_normal: Array 3 มิติ (N_windows, Seq_len, Features)
            X_val_normal:   Array 3 มิติ สำหรับคำนวณ Threshold (ปกติเท่านั้น)
            epochs:         จำนวนรอบฝึก
            batch_size:     ขนาด Batch
            lr:             Learning Rate
            device:         'cuda' หรือ 'cpu'
        """
        self._set_seeds()

        if X_train_normal.ndim != 3:
            raise ValueError(f"X_train_normal ต้องมี 3 มิติ ได้รับ shape {X_train_normal.shape}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.info(f"เริ่มการฝึก T²-LSTM บนอุปกรณ์: {device}")

        X_tr = np.asarray(X_train_normal, dtype=np.float32)
        n_samples, seq_len, n_features = X_tr.shape

        if n_features != self.input_dim:
            self.input_dim = n_features

        self.model = T2LSTMNet(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(device)

        dataset = TensorDataset(torch.from_numpy(X_tr))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.history_ = {"train_loss": [], "val_loss": []}

        # 1. ฝึกเครือข่าย LSTM Autoencoder
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            avg_train_loss = epoch_loss / n_samples
            self.history_["train_loss"].append(avg_train_loss)

            if X_val_normal is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.from_numpy(X_val_normal.astype(np.float32)).to(device)
                    v_recon, _ = self.model(X_v)
                    v_loss = criterion(v_recon, X_v).item()
                    self.history_["val_loss"].append(v_loss)

            if epoch % 5 == 0 or epoch == epochs:
                val_msg = f", val_loss: {self.history_['val_loss'][-1]:.6f}" if X_val_normal is not None else ""
                self._logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] - loss: {avg_train_loss:.6f}{val_msg}")

        # 2. สกัด Latent Features บนข้อมูลฝึกปกติเพื่อคำนวณ Hotelling's T² Parameters
        self.model.eval()
        latent_vectors = []
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (bx,) in eval_loader:
                bx = bx.to(device)
                z = self.model.encode(bx)
                latent_vectors.append(z.cpu().numpy())

        Z_train = np.concatenate(latent_vectors, axis=0)  # (N, latent_dim)

        self.latent_mean_ = np.mean(Z_train, axis=0)
        cov = np.cov(Z_train, rowvar=False)

        # ป้องกัน Singular Matrix ด้วย Ridge Regularization (L2 Shrinkage)
        reg = 1e-4 * np.trace(cov) / self.latent_dim * np.eye(self.latent_dim)
        cov_reg = cov + reg

        try:
            self.latent_cov_inv_ = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            self.latent_cov_inv_ = np.linalg.pinv(cov_reg)

        # 3. คำนวณขีดจำกัดทางสถิติ (Control Limits) สำหรับ T² และ SPE
        # T² Limit จาก F-Distribution: T²_alpha = (p*(n-1)/(n-p)) * F_{alpha}(p, n-p)
        p = self.latent_dim
        n = n_samples
        if n > p:
            f_val = stats.f.ppf(1.0 - self.significance, p, n - p)
            self.t2_limit_ = float(p * (n - 1) / (n - p) * f_val)
        else:
            self.t2_limit_ = float(stats.chi2.ppf(1.0 - self.significance, p))

        # SPE Limit จาก Normal Validation (หรือ Train)
        ref_data = X_val_normal if X_val_normal is not None else X_train_normal
        spe_ref = self._compute_spe(ref_data, device=device)
        self.spe_limit_ = float(np.percentile(spe_ref, (1.0 - self.significance) * 100.0))
        if self.spe_limit_ <= 0:
            self.spe_limit_ = 1.0

        # ตั้งเกณฑ์ Composite Threshold เริ่มต้นที่ 1.0 (เมื่อค่ามาตรฐาน > 1.0 คือเกิน limit)
        self.threshold_ = 1.0
        self.is_fitted = True

        self._logger.info(
            f"Fit T²-LSTM สำเร็จ: T² Limit = {self.t2_limit_:.4f}, SPE Limit = {self.spe_limit_:.6f}"
        )
        return self

    def _compute_spe(self, X: np.ndarray, batch_size: int = 256, device: Optional[str] = None) -> np.ndarray:
        """คำนวณ Squared Prediction Error (SPE)"""
        if self.model is None:
            raise RuntimeError("โมเดลยังไม่ได้รับการฝึก")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.eval()
        self.model.to(device)

        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        spe_list = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(device)
                recon, _ = self.model(bx)
                # MSE เฉลี่ยข้ามมิติเวลาและฟีเจอร์
                spe = torch.mean((bx - recon) ** 2, dim=(1, 2)).cpu().numpy()
                spe_list.append(spe)

        return np.concatenate(spe_list)

    def _compute_t2(self, X: np.ndarray, batch_size: int = 256, device: Optional[str] = None) -> np.ndarray:
        """คำนวณ Hotelling's T² ใน Latent Space"""
        if self.model is None or self.latent_mean_ is None or self.latent_cov_inv_ is None:
            raise RuntimeError("โมเดลยังไม่ได้รับการฝึก")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.eval()
        self.model.to(device)

        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        z_list = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(device)
                z = self.model.encode(bx)
                z_list.append(z.cpu().numpy())

        Z = np.concatenate(z_list, axis=0)  # (N, latent_dim)
        diff = Z - self.latent_mean_
        # Mahalanobis distance squared: sum_j diff_ij * (diff @ cov_inv)_ij
        t2_scores = np.sum((diff @ self.latent_cov_inv_) * diff, axis=1)
        return t2_scores

    def score(
        self,
        X: np.ndarray,
        return_components: bool = False,
        device: Optional[str] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        คำนวณ Composite Anomaly Score:
        Score = gamma * (T² / T²_limit) + (1 - gamma) * (SPE / SPE_limit)

        Args:
            X: Array 3 มิติ (N_samples, Seq_len, Features)
            return_components: หากเป็น True จะส่งคืน (combined_score, t2_score, spe_score)

        Returns:
            composite_score หรือ tuple ของ (score, t2, spe)
        """
        if not self.is_fitted:
            raise RuntimeError("โมเดลยังไม่ได้รับการฝึก กรุณาเรียก .fit() ก่อน")

        t2_scores = self._compute_t2(X, device=device)
        spe_scores = self._compute_spe(X, device=device)

        norm_t2 = t2_scores / max(self.t2_limit_, 1e-8)
        norm_spe = spe_scores / max(self.spe_limit_, 1e-8)

        composite = self.gamma * norm_t2 + (1.0 - self.gamma) * norm_spe

        if return_components:
            return composite, t2_scores, spe_scores
        return composite

    def predict(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """ทำนายความผิดปกติ (0 = ปกติ, 1 = ผิดปกติ) ตาม Composite Threshold"""
        composite_scores = self.score(X, device=device)
        return (composite_scores > self.threshold_).astype(int)

    def tune_threshold_on_val(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_far: float = 0.05,
        device: Optional[str] = None,
    ) -> float:
        """
        ปรับจูน Composite Threshold บนชุด Validation เพื่อให้ FAR สอดคล้องกับ target_far
        """
        scores = self.score(X_val, device=device)
        normal_scores = scores[y_val == 0]

        if len(normal_scores) == 0:
            raise ValueError("ชุด Validation ต้องมีข้อมูลปกติ (y=0) เพื่อจูน Threshold")

        pct = (1.0 - target_far) * 100.0
        self.threshold_ = float(np.percentile(normal_scores, pct))
        self._logger.info(f"ปรับจูน T²-LSTM Threshold สำเร็จ: {self.threshold_:.4f} (Target FAR: {target_far:.1%})")
        return self.threshold_

    def save(self, filepath: Union[str, Path]) -> None:
        """บันทึกโมเดลและพารามิเตอร์ทั้งหมดลงไฟล์"""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("ไม่สามารถบันทึกโมเดลที่ยังไม่ได้ฝึก")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "gamma": self.gamma,
            "significance": self.significance,
            "random_seed": self.random_seed,
            "latent_mean_": self.latent_mean_,
            "latent_cov_inv_": self.latent_cov_inv_,
            "t2_limit_": self.t2_limit_,
            "spe_limit_": self.spe_limit_,
            "threshold_": self.threshold_,
            "history_": self.history_,
        }
        torch.save(checkpoint, path)
        self._logger.info(f"บันทึก T²-LSTM Checkpoint ไปที่: {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[str] = None) -> "T2LSTMDetector":
        """โหลดโมเดล T²-LSTM จากไฟล์ checkpoint"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=device)

        instance = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            gamma=checkpoint["gamma"],
            significance=checkpoint["significance"],
            random_seed=checkpoint["random_seed"],
        )
        instance.latent_mean_ = checkpoint["latent_mean_"]
        instance.latent_cov_inv_ = checkpoint["latent_cov_inv_"]
        instance.t2_limit_ = checkpoint["t2_limit_"]
        instance.spe_limit_ = checkpoint["spe_limit_"]
        instance.threshold_ = checkpoint["threshold_"]
        instance.history_ = checkpoint["history_"]

        instance.model = T2LSTMNet(
            input_dim=instance.input_dim,
            latent_dim=instance.latent_dim,
            hidden_dim=instance.hidden_dim,
            num_layers=instance.num_layers,
        ).to(device)

        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.is_fitted = True
        return instance
