"""
src/utils/logging_utils.py
--------------------------
Utility functions สำหรับ logging และ reproducibility

Functions:
    setup_logger  — สร้าง logger พร้อม console และ file handler
    seed_everything — กำหนด random seeds สำหรับ numpy, random, torch (optional)
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> logging.Logger:
    """
    สร้าง logger ที่ส่งออกทั้ง console และไฟล์ (optional)

    Args:
        name:    ชื่อ logger (ควรใช้ __name__ ของ module ที่เรียก)
        log_dir: โฟลเดอร์สำหรับบันทึก log file
                 None = ไม่บันทึกไฟล์ (console เท่านั้น)
        level:   ระดับ logging เช่น logging.INFO, logging.DEBUG
        fmt:     รูปแบบ log message

    Returns:
        logging.Logger: logger ที่พร้อมใช้งาน

    Example:
        >>> logger = setup_logger(__name__, log_dir="logs")
        >>> logger.info("Started experiment")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ป้องกัน duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{name.replace('.', '_')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def seed_everything(seed: int = 42, use_torch: bool = False) -> None:
    """
    กำหนด random seeds เพื่อให้ผลการทดลอง reproducible

    Args:
        seed:      ค่า seed (default: 42 ตาม default_config.yaml)
        use_torch: True = กำหนด seed ให้ PyTorch ด้วย
                   False = กำหนดเฉพาะ Python random และ numpy

    Note:
        - PyTorch ต้องติดตั้งก่อนจึงจะใช้ use_torch=True ได้
        - ใน Google Colab ควรใช้ use_torch=True เสมอเมื่อฝึกโมเดล

    Example:
        >>> seed_everything(42, use_torch=False)
        >>> seed_everything(42, use_torch=True)  # บน Colab เมื่อมี PyTorch
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if use_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            logging.getLogger(__name__).warning(
                "PyTorch ไม่พบในระบบ — ข้าม torch seed (ติดตั้งบน Colab เพื่อใช้งาน)"
            )
