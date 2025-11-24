import random
import numpy as np
import torch
import os

def set_global_seed(seed: int, deterministic: bool = False):
    """
    Python, NumPy ve PyTorch için global seed'i sabitler.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # İleri düzey determinizm (Hızı düşürebilir, debug için açılabilir)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
    print(f"🔒 Global Seed Set to: {seed} (Deterministic: {deterministic})")