import torch
import platform
import os
import logging

def get_optimal_device() -> torch.device:
    """
    İşletim sistemini ve mevcut donanımı analiz ederek en uygun cihazı döndürür.
    Mac (MPS) için float32 zorlaması yapar.
    """
    device = torch.device("cpu")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"🟢 CUDA Device Found: {torch.cuda.get_device_name(0)}")
    
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        device = torch.device("mps")
        # Mac silikon işlemcilerde bfloat16 hatalara yol açabilir, float32 zorluyoruz.
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' 
        logging.info("🟠 Apple Metal (MPS) Device Found. Using strict float32 context.")
    
    else:
        logging.warning("⚪ No accelerator found. Using CPU.")

    return device