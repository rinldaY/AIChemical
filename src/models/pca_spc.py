"""
src/models/pca_spc.py
---------------------
Multivariate Statistical Process Control (MSPC) ด้วย PCA

สถิติที่คำนวณ:
    1. Hotelling's T²:
       - วัดความผันแปรใน Principal Component Subspace (PCS)
       - T² = x^T P Lambda^{-1} P^T x = sum_{i=1}^l (t_i^2 / lambda_i)
       - ขอบเขตควบคุม (Control Limit) คำนวณจาก F-Distribution:
         T²_limit = [l(n^2 - 1) / (n(n - l))] * F_alpha(l, n - l)

    2. Squared Prediction Error (SPE / Q-statistic):
       - วัดความผันแปรใน Residual Subspace (RS)
       - SPE = ||x - x_hat||² = x^T (I - P P^T) x
       - ขอบเขตควบคุม Jackson-Mudholkar Approximation จากผลรวมค่าไอเกนที่เหลือ

    3. รวมผลการตรวจจับ:
       - แจ้งเตือนเมื่อ T² เกินเกณฑ์ หรือ SPE เกินเกณฑ์
       - Normalized Anomaly Score: max(T² / T²_limit, SPE / SPE_limit)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class PCASPCModel:
    """
    PCA-based Multivariate Statistical Process Control (Hotelling's T² + SPE)

    Args:
        n_components:        จำนวน components (int) หรือ สัดส่วน variance (float เช่น 0.85)
        alpha:               ระดับนัยสำคัญสำหรับขอบเขตควบคุม (default: 0.05 คือ 95% confidence)
        feature_names:       รายชื่อฟีเจอร์ (optional)
    """

    def __init__(
        self,
        n_components: Union[int, float] = 0.85,
        alpha: float = 0.05,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha ต้องอยู่ระหว่าง 0 และ 1 (ได้รับ {alpha})")

        self.n_components = n_components
        self.alpha = float(alpha)
        self.feature_names = feature_names

        self.pca: Optional[PCA] = None
        self.n_components_: int = 0
        self.t2_limit_: float = 0.0
        self.spe_limit_: float = 0.0
        self.eigenvalues_: Optional[np.ndarray] = None
        self.residual_eigenvalues_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        self._logger = logging.getLogger(self.__class__.__name__)

    def fit(self, X_train_normal: np.ndarray) -> "PCASPCModel":
        """
        Fit PCA บนข้อมูลปกติ (X_train_normal ต้องผ่านการ Standardize มาแล้ว)
        และคำนวณขอบเขตควบคุม T²_limit และ SPE_limit ตามทฤษฎีสถิติ

        Args:
            X_train_normal: Array 2 มิติ (N_samples, N_features)

        Returns:
            self
        """
        X = np.asarray(X_train_normal, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X_train_normal ต้องมี 2 มิติ แต่ได้ shape {X.shape}")
        n_samples, n_features = X.shape

        if n_samples <= n_features:
            raise ValueError(
                f"จำนวนตัวอย่าง ({n_samples}) ต้องมากกว่าจำนวนฟีเจอร์ ({n_features}) เพื่อคำนวณ covariance"
            )

        # 1. Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        self.n_components_ = self.pca.n_components_
        self.eigenvalues_ = self.pca.explained_variance_

        # 2. คำนวณ Hotelling's T² Control Limit ผ่าน F-distribution
        l = self.n_components_
        n = n_samples
        f_crit = stats.f.ppf(1.0 - self.alpha, l, n - l)
        self.t2_limit_ = float((l * (n**2 - 1) / (n * (n - l))) * f_crit)

        # 3. คำนวณค่าไอเกนที่เหลือสำหรับ SPE (Jackson-Mudholkar Approximation)
        # คำนวณ eigenvalues ทั้งหมดของ covariance matrix
        cov_matrix = np.cov(X, rowvar=False)
        all_eigenvalues = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]
        # ค่าไอเกนส่วนที่เหลือ (Residual eigenvalues)
        self.residual_eigenvalues_ = np.maximum(all_eigenvalues[l:], 1e-12)

        theta1 = float(np.sum(self.residual_eigenvalues_))
        theta2 = float(np.sum(self.residual_eigenvalues_ ** 2))
        theta3 = float(np.sum(self.residual_eigenvalues_ ** 3))

        if theta1 > 0 and theta2 > 0:
            h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * (theta2 ** 2))
            if abs(h0) < 1e-4:
                h0 = 1e-4
            z_crit = stats.norm.ppf(1.0 - self.alpha)
            term1 = (z_crit * np.sqrt(2.0 * theta2 * (h0 ** 2))) / theta1
            term2 = (theta2 * h0 * (h0 - 1.0)) / (theta1 ** 2)
            bracket = term1 + 1.0 + term2
            if bracket > 0:
                self.spe_limit_ = float(theta1 * (bracket ** (1.0 / h0)))
            else:
                self.spe_limit_ = float(np.percentile(self._compute_spe_raw(X), (1.0 - self.alpha) * 100))
        else:
            # Fallback หากไม่มี residual variance (ใช้ empirical percentile)
            self.spe_limit_ = float(np.percentile(self._compute_spe_raw(X), (1.0 - self.alpha) * 100))

        self.is_fitted = True
        self._logger.info(
            f"Fit PCASPC สำเร็จ: ใช้ {self.n_components_}/{n_features} components | "
            f"T²_limit = {self.t2_limit_:.4f} | SPE_limit = {self.spe_limit_:.4f} (alpha={self.alpha})"
        )
        return self

    def _compute_t2_raw(self, X: np.ndarray) -> np.ndarray:
        """คำนวณ Hotelling's T² statistic"""
        if self.pca is None or self.eigenvalues_ is None:
            raise RuntimeError("โมเดลยังไม่ได้ fit")
        # scores = X * P
        scores = self.pca.transform(X)
        # T² = sum (score_i² / lambda_i)
        t2 = np.sum((scores ** 2) / self.eigenvalues_, axis=1)
        return t2

    def _compute_spe_raw(self, X: np.ndarray) -> np.ndarray:
        """คำนวณ SPE (Squared Prediction Error / Q-statistic)"""
        if self.pca is None:
            raise RuntimeError("โมเดลยังไม่ได้ fit")
        X_hat = self.pca.inverse_transform(self.pca.transform(X))
        residuals = X - X_hat
        spe = np.sum(residuals ** 2, axis=1)
        return spe

    def compute_statistics(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        คำนวณทั้งค่า T² และ SPE สำหรับชุดข้อมูล X

        Args:
            X: Array 2 มิติ (N_samples, N_features)

        Returns:
            (T²_values, SPE_values)
        """
        if not self.is_fitted:
            raise RuntimeError("ต้องเรียก .fit() ก่อนคำนวณสถิติ")
        X_arr = np.asarray(X, dtype=float)
        return self._compute_t2_raw(X_arr), self._compute_spe_raw(X_arr)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        คำนวณคะแนนความผิดปกติรวมแบบปรับสเกล (Combined Normalized Anomaly Score)

        Score = max(T² / T²_limit, SPE / SPE_limit)
        หาก score > 1.0 แสดงว่ามีสถิติใดสถิติหนึ่งเกินขอบเขตควบคุม

        Args:
            X: Array 2 มิติ (N_samples, N_features)

        Returns:
            np.ndarray: Array 1 มิติของคะแนนความผิดปกติ
        """
        t2, spe = self.compute_statistics(X)
        norm_t2 = t2 / max(self.t2_limit_, 1e-6)
        norm_spe = spe / max(self.spe_limit_, 1e-6)
        return np.maximum(norm_t2, norm_spe)

    def predict(self, X: np.ndarray, decision: str = "combined") -> np.ndarray:
        """
        ทำนายความผิดปกติ (0=Normal, 1=Fault)

        Args:
            X:        Array 2 มิติ (N_samples, N_features)
            decision: วิธีตัดสินใจ ('combined' คือ T² เกิน หรือ SPE เกิน, 't2_only', 'spe_only')

        Returns:
            np.ndarray: Array 1 มิติ (0 หรือ 1)
        """
        t2, spe = self.compute_statistics(X)

        if decision == "combined":
            is_anomaly = (t2 > self.t2_limit_) | (spe > self.spe_limit_)
        elif decision == "t2_only":
            is_anomaly = (t2 > self.t2_limit_)
        elif decision == "spe_only":
            is_anomaly = (spe > self.spe_limit_)
        else:
            raise ValueError(f"ไม่รู้จัก decision mode '{decision}' (ต้องเป็น combined, t2_only หรือ spe_only)")

        return is_anomaly.astype(int)

    def tune_limits_on_val(self, X_val: np.ndarray, target_far: float = 0.05) -> Tuple[float, float]:
        """
        ปรับจูน T²_limit และ SPE_limit บนข้อมูลปกติในชุด Validation ให้ตรงกับเป้าหมาย FAR

        Args:
            X_val:      ข้อมูลปกติในชุด Validation (y_val == 0)
            target_far: เป้าหมาย False Alarm Rate (default: 0.05)

        Returns:
            (new_t2_limit, new_spe_limit)
        """
        t2, spe = self.compute_statistics(X_val)
        # ปรับขอบเขตตาม empirical percentile
        pct = (1.0 - (target_far / 2.0)) * 100.0
        self.t2_limit_ = float(np.percentile(t2, pct))
        self.spe_limit_ = float(np.percentile(spe, pct))
        self._logger.info(
            f"ปรับจูนขอบเขตควบคุมบน Val สำเร็จ: T²_limit = {self.t2_limit_:.4f}, "
            f"SPE_limit = {self.spe_limit_:.4f} (เป้าหมาย FAR = {target_far*100:.1f}%)"
        )
        return self.t2_limit_, self.spe_limit_
