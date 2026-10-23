"""
src/models/stacked_lstm.py
--------------------------
Stacked LSTM Sequence Autoencoder for Unsupervised Anomaly Detection

หลักการ:
    - โมเดลโครงข่ายประสาทแบบ Sequential สำหรับตรวจจับความผิดปกติในกระบวนการเคมี
    - ใช้ Stacked LSTM Encoder เพื่อแปลง Time-series Window (Batch, Seq_len, Features)
      ไปเป็น Hidden Representation
    - ใช้ Stacked LSTM Decoder เพื่อ Reconstruct ลำดับสัญญาณเดิมกลับคืนมา
    - Anomaly Score: Reconstruction Error (MSE) ของหน้าต่างเวลา
    - Threshold: กำหนดจาก percentile ของคะแนนบนชุด Validation ข้อมูลปกติ
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class StackedLSTMNet(nn.Module):
    """
    PyTorch Neural Network Module สำหรับ Stacked LSTM Autoencoder

    Args:
        input_dim:    มิติของฟีเจอร์ขาเข้า (default: 52 สำหรับ TEP)
        hidden_dim:   ขนาดของ LSTM hidden units (default: 64)
        num_layers:   จำนวนชั้นของ Stacked LSTM (default: 2)
        dropout:      Dropout probability ระหว่าง LSTM layers (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int = 52,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0

        # Encoder: (Batch, Seq, Features) -> (Batch, Seq, Hidden) & (h_n, c_n)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Decoder: (Batch, Seq, Hidden) -> (Batch, Seq, Hidden)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Output projection back to feature dimension
        self.output_linear = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        แปลง Time-series Window ไปยัง Context Vector
        Returns:
            context_vector: (Batch, Hidden) จากชั้นบนสุดของ h_n
            latent_seq: (Batch, Seq, Hidden)
        """
        out, (h_n, c_n) = self.encoder(x)
        # h_n shape: (num_layers, batch, hidden_dim) -> ชั้นสุดท้าย h_n[-1]
        context_vector = h_n[-1]
        return context_vector, out

    def decode(self, context_vector: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        สร้างลำดับข้อมูลย้อนกลับจาก Context Vector
        """
        # ขยาย context_vector ซ้ำตามความยาว seq_len: (Batch, Seq_len, Hidden)
        repeated = context_vector.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(repeated)
        reconstruction = self.output_linear(dec_out)
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> encode -> decode -> x_reconstructed"""
        seq_len = x.size(1)
        context_vector, _ = self.encode(x)
        return self.decode(context_vector, seq_len)


class StackedLSTMDetector:
    """
    Stacked LSTM Anomaly Detector สำหรับ TEP Time-Series

    Args:
        input_dim:     มิติของข้อมูลขาเข้า (52 features)
        hidden_dim:    ขนาด Hidden state (64)
        num_layers:    จำนวนชั้น LSTM (2)
        dropout:       Dropout rate (0.1)
        threshold_pct: Percentile สำหรับตั้ง threshold จาก Val error (default: 95.0)
        random_seed:   Random seed สำหรับ reproducibility (default: 42)
    """

    def __init__(
        self,
        input_dim: int = 52,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        threshold_pct: float = 95.0,
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.threshold_pct = threshold_pct
        self.random_seed = random_seed

        self.model: Optional[StackedLSTMNet] = None
        self.threshold_: float = 0.0
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
    ) -> "StackedLSTMDetector":
        """
        ฝึก Stacked LSTM บนเฉพาะ Window ข้อมูลปกติ (Train Set)

        Args:
            X_train_normal: Array 3 มิติ (N_windows, Seq_len, N_features)
            X_val_normal:   Array 3 มิติ สำหรับคำนวณ Threshold (ปกติเท่านั้น)
            epochs:         จำนวนรอบการฝึก
            batch_size:     ขนาด Batch
            lr:             Learning Rate
            device:         'cuda' หรือ 'cpu' (None = auto-detect)
        """
        self._set_seeds()

        if X_train_normal.ndim != 3:
            raise ValueError(f"X_train_normal ต้องมี 3 มิติ (N, Seq, Features) แต่ได้ shape {X_train_normal.shape}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.info(f"เริ่มการฝึก Stacked LSTM บนอุปกรณ์: {device}")

        X_tr = np.asarray(X_train_normal, dtype=np.float32)
        n_samples, seq_len, n_features = X_tr.shape

        if n_features != self.input_dim:
            self.input_dim = n_features

        self.model = StackedLSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)

        dataset = TensorDataset(torch.from_numpy(X_tr))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.history_ = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
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
                    v_recon = self.model(X_v)
                    v_loss = criterion(v_recon, X_v).item()
                    self.history_["val_loss"].append(v_loss)

            if epoch % 5 == 0 or epoch == epochs:
                val_msg = f", val_loss: {self.history_['val_loss'][-1]:.6f}" if X_val_normal is not None else ""
                self._logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] - train_loss: {avg_train_loss:.6f}{val_msg}")

        # กำหนด Threshold
        if X_val_normal is not None:
            self._logger.info(f"กำลังคำนวณ Threshold ที่ Percentile {self.threshold_pct}% จากชุด Val ปกติ")
            val_scores = self.score(X_val_normal, device=device)
            self.threshold_ = float(np.percentile(val_scores, self.threshold_pct))
        else:
            self._logger.warning("ไม่ได้ระบุ X_val_normal จึงคำนวณ Threshold จากชุด Train ปกติ")
            train_scores = self.score(X_train_normal, device=device)
            self.threshold_ = float(np.percentile(train_scores, self.threshold_pct))

        self.is_fitted = True
        self._logger.info(f"ฝึก Stacked LSTM สำเร็จ - กำหนด Threshold = {self.threshold_:.6f}")
        return self

    def score(self, X: np.ndarray, batch_size: int = 256, device: Optional[str] = None) -> np.ndarray:
        """
        คำนวณ Anomaly Score (Reconstruction MSE ต่อ sample หน้าต่างเวลา)

        Args:
            X: Array 3 มิติ (N_samples, Seq_len, N_features)
        Returns:
            scores: Array 1 มิติ (N_samples,)
        """
        if self.model is None:
            raise RuntimeError("โมเดลยังไม่ได้รับการฝึก กรุณาเรียก .fit() ก่อน")
        if X.ndim != 3:
            raise ValueError(f"X ต้องมี 3 มิติ แต่ได้รับ shape {X.shape}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.eval()
        self.model.to(device)

        X_arr = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X_arr))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        scores_list = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                recon = self.model(batch_x)
                # MSE เฉลี่ยข้าม Seq_len และ Features
                batch_scores = torch.mean((batch_x - recon) ** 2, dim=(1, 2)).cpu().numpy()
                scores_list.append(batch_scores)

        return np.concatenate(scores_list)

    def predict(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """ทำนายความผิดปกติ (0 = ปกติ, 1 = ผิดปกติ)"""
        scores = self.score(X, device=device)
        return (scores > self.threshold_).astype(int)

    def tune_threshold_on_val(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_far: float = 0.05,
        device: Optional[str] = None,
    ) -> float:
        """
        ปรับจูน Threshold บนชุด Validation เพื่อให้ False Alarm Rate (FAR) ใกล้เคียง target_far
        """
        val_scores = self.score(X_val, device=device)
        normal_scores = val_scores[y_val == 0]

        if len(normal_scores) == 0:
            raise ValueError("ชุด Validation ต้องมีข้อมูลปกติ (y=0) เพื่อจูน Threshold")

        pct = (1.0 - target_far) * 100.0
        self.threshold_ = float(np.percentile(normal_scores, pct))
        self._logger.info(f"ปรับจูน Threshold สำเร็จ: {self.threshold_:.6f} (เป้าหมาย FAR <= {target_far:.1%})")
        return self.threshold_

    def save(self, filepath: Union[str, Path]) -> None:
        """บันทึกโมเดลและพารามิเตอร์"""
        if self.model is None:
            raise RuntimeError("ไม่สามารถบันทึกโมเดลที่ยังไม่ได้ฝึก")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "threshold_pct": self.threshold_pct,
            "threshold_": self.threshold_,
            "random_seed": self.random_seed,
            "history_": self.history_,
        }
        torch.save(checkpoint, path)
        self._logger.info(f"บันทึก Stacked LSTM สำเร็จที่: {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[str] = None) -> "StackedLSTMDetector":
        """โหลดโมเดลจากไฟล์ checkpoint"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=device)

        instance = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            threshold_pct=checkpoint["threshold_pct"],
            random_seed=checkpoint["random_seed"],
        )
        instance.threshold_ = checkpoint["threshold_"]
        instance.history_ = checkpoint["history_"]

        instance.model = StackedLSTMNet(
            input_dim=instance.input_dim,
            hidden_dim=instance.hidden_dim,
            num_layers=instance.num_layers,
            dropout=instance.dropout,
        ).to(device)

        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.is_fitted = True
        return instance
