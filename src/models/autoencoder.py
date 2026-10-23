"""
src/models/autoencoder.py
-------------------------
Deep Autoencoder for Unsupervised Anomaly Detection

หลักการ:
    - ฝึกโมเดลบีบอัดและคืนค่า (Reconstruct) ข้อมูลโดยใช้เฉพาะข้อมูลปกติ (Normal Data)
    - โครงสร้าง: Fully-Connected Encoder-Decoder พร้อม Bottleneck Latent Space
    - Anomaly Score: Reconstruction Error (MSE ต่อ sample)
      score = (1 / d) * sum_{j=1}^d (x_j - x_hat_j)^2
    - Threshold: กำหนดจาก percentile ของ reconstruction error บนชุด Validation ข้อมูลปกติ
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class AutoencoderNet(nn.Module):
    """
    PyTorch Neural Network Module สำหรับ Autoencoder

    Args:
        input_dim:    มิติของข้อมูลขาเข้า (default: 52 สำหรับ TEP)
        latent_dim:   มิติของ Bottleneck Latent Space (default: 8)
        hidden_dims:  ขนาดของ hidden layers ใน encoder (default: [32, 16])
    """

    def __init__(
        self,
        input_dim: int = 52,
        latent_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 1. สร้าง Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 2. สร้าง Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """แปลงข้อมูลไปยัง latent representation"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """แปลง latent representation กลับเป็น reconstructed features"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> latent -> x_hat"""
        z = self.encode(x)
        return self.decode(z)


class AutoencoderDetector:
    """
    Autoencoder Anomaly Detector สำหรับ TEP Benchmark

    Args:
        input_dim:     มิติของข้อมูลขาเข้า (52 features)
        latent_dim:    มิติของ Bottleneck (8)
        hidden_dims:   มิติ hidden layers ใน encoder ([32, 16])
        threshold_pct: Percentile สำหรับตั้ง threshold จาก Val error (default: 95.0)
        random_seed:   Random seed สำหรับ reproducibility (default: 42)
    """

    def __init__(
        self,
        input_dim: int = 52,
        latent_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        threshold_pct: float = 95.0,
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 16]
        self.threshold_pct = threshold_pct
        self.random_seed = random_seed

        self.model: Optional[AutoencoderNet] = None
        self.threshold_: float = 0.0
        self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.is_fitted: bool = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def _set_seeds(self) -> None:
        """กำหนด random seeds เพื่อความสามารถในการทำซ้ำได้ (Reproducibility)"""
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)

    def fit(
        self,
        X_train_normal: np.ndarray,
        X_val_normal: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> "AutoencoderDetector":
        """
        ฝึก Autoencoder บนเฉพาะชุดข้อมูลปกติ (Train Set)

        Args:
            X_train_normal: Array 2 มิติ (N_samples, N_features) ของชุดฝึกปกติ
            X_val_normal:   Array 2 มิติ (N_samples, N_features) ของ Val ปกติ สำหรับตั้ง threshold
            epochs:         จำนวนรอบการฝึก (default: 20)
            batch_size:     ขนาด batch (default: 256)
            lr:             Learning rate (default: 1e-3)
            device:         'cuda' หรือ 'cpu' (None = auto-detect)

        Returns:
            self
        """
        self._set_seeds()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.info(f"เริ่มการฝึก Autoencoder บนอุปกรณ์: {device}")

        X_tr = np.asarray(X_train_normal, dtype=np.float32)
        n_samples, n_features = X_tr.shape

        if n_features != self.input_dim:
            self.input_dim = n_features

        self.model = AutoencoderNet(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
        ).to(device)

        train_dataset = TensorDataset(torch.from_numpy(X_tr))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.history_ = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            avg_train_loss = epoch_loss / n_samples
            self.history_["train_loss"].append(avg_train_loss)

            # คำนวณ Val loss หากมี
            if X_val_normal is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.from_numpy(X_val_normal.astype(np.float32)).to(device)
                    v_rec = self.model(X_v)
                    v_loss = criterion(v_rec, X_v).item()
                    self.history_["val_loss"].append(v_loss)

            if epoch % 5 == 0 or epoch == epochs:
                val_info = f" | Val Loss: {self.history_['val_loss'][-1]:.6f}" if X_val_normal is not None else ""
                self._logger.info(f"Epoch {epoch:02d}/{epochs} - Train Loss: {avg_train_loss:.6f}{val_info}")

        # ตั้ง Anomaly Threshold จาก Reconstruction Error บน Validation Normal Set
        self.is_fitted = True
        eval_data = X_val_normal if X_val_normal is not None else X_tr
        scores = self.score(eval_data)
        self.threshold_ = float(np.percentile(scores, self.threshold_pct))

        self._logger.info(
            f"Autoencoder fit สำเร็จ: Anomaly Threshold = {self.threshold_:.6f} "
            f"(percentile {self.threshold_pct}%)"
        )
        return self

    def score(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        คำนวณคะแนนความผิดปกติ (Reconstruction Error / MSE ต่อ sample)

        Args:
            X: Array 2 มิติ (N_samples, N_features)

        Returns:
            np.ndarray: Array 1 มิติ (N_samples,) ของคะแนน MSE
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("ต้องเรียก .fit() ก่อนคำนวณคะแนน")

        device = next(self.model.parameters()).device
        self.model.eval()

        X_arr = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X_arr))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        scores_list: List[np.ndarray] = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                rec = self.model(batch_x)
                # MSE ต่อ sample: mean((x - rec)^2, axis=1)
                batch_mse = torch.mean((batch_x - rec) ** 2, dim=1).cpu().numpy()
                scores_list.append(batch_mse)

        return np.concatenate(scores_list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ทำนายความผิดปกติ (0=Normal, 1=Fault)

        Args:
            X: Array 2 มิติ (N_samples, N_features)

        Returns:
            np.ndarray: Array 1 มิติของผลทำนาย (0 หรือ 1)
        """
        scores = self.score(X)
        return (scores > self.threshold_).astype(int)

    def save(self, filepath: Union[str, Path]) -> None:
        """บันทึกโมเดลและพารามิเตอร์ลงไฟล์ checkpoint"""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("ไม่สามารถบันทึกโมเดลที่ยังไม่ได้ fit")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "threshold_": self.threshold_,
            "threshold_pct": self.threshold_pct,
            "history_": self.history_,
            "random_seed": self.random_seed,
        }
        torch.save(checkpoint, path)
        self._logger.info(f"บันทึก Autoencoder checkpoint ไปที่: {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[str] = None) -> "AutoencoderDetector":
        """โหลดโมเดลจากไฟล์ checkpoint"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(filepath, map_location=device)
        detector = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            threshold_pct=checkpoint["threshold_pct"],
            random_seed=checkpoint.get("random_seed", 42),
        )

        detector.model = AutoencoderNet(
            input_dim=detector.input_dim,
            latent_dim=detector.latent_dim,
            hidden_dims=detector.hidden_dims,
        ).to(device)

        detector.model.load_state_dict(checkpoint["model_state_dict"])
        detector.threshold_ = checkpoint["threshold_"]
        detector.history_ = checkpoint.get("history_", {})
        detector.is_fitted = True
        return detector
