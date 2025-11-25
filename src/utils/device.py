import torch
import platform
import logging

def get_optimal_device(force_cpu: bool = False) -> torch.device:
    """
    Mevcut donanımı analiz eder ve en uygun PyTorch cihazını döndürür.
    Mac (Apple Silicon) için MPS, NVIDIA için CUDA, aksi halde CPU seçer.
    
    Args:
        force_cpu (bool): True ise donanım hızlandırmayı yoksayar.
    
    Returns:
        torch.device: Seçilen cihaz.
    """
    if force_cpu:
        logging.info("⚠️ Device forced to CPU.")
        return torch.device("cpu")

    system = platform.system()
    
    # 1. Mac OS (Apple Silicon) Kontrolü
    if system == "Darwin" and torch.backends.mps.is_available():
        logging.info("🚀 Apple Silicon (MPS) detected. Using Metal Performance Shaders.")
        # KRİTİK: MPS şu an bazı operasyonlarda float64 desteklemiyor, float32'ye zorlamak gerekebilir.
        # Bu ayar genellikle model tanımında dtype=torch.float32 ile yapılır ama burada logluyoruz.
        return torch.device("mps")
    
    # 2. NVIDIA CUDA Kontrolü
    elif torch.cuda.is_available():
        logging.info(f"🚀 NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    
    # 3. Fallback
    else:
        logging.warning("⚠️ No GPU acceleration detected. Running on CPU (Slow).")
        return torch.device("cpu")