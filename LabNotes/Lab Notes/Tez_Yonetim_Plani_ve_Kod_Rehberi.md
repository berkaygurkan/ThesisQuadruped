
# A. GELİŞMİŞ KLASÖR YAPISI

```text
thesis_repo/
├── setup.py                            # IsaacLab'e linklemek için (pip -e .)
├── requirements.txt                    # Ortam bağımlılıkları
├── configs/                            # ANA KONFİGÜRASYONLAR (YAML)
│   ├── config.yaml                     # Global ayarlar (WandB, Debug, Default Device)
│   ├── exported/                       # Her run'da otomatik dump edilen configler (run_id.json)
│   └── phase_3_1/                      # Faz ayarları
├── data/
│   ├── checkpoints/                    # Format: [tarih]_[faz]_[seed]/checkpoint.pt
│   └── logs/                           # Tensorboard/WandB
├── src/                                # CORE ENGINE (Değişmez Motor)
│   ├── __init__.py
│   ├── utils/
│   │   ├── device.py                   # Cihaz ve dtype yönetimi (Mac float32 fix)
│   │   ├── logger.py                   # Hiyerarşik loglama
│   │   └── seeding.py                  # Global seed ve Git Hash alıcı
│   ├── algos/
│   │   ├── trainer.py                  # BaseTrainer (ABC) ve PPOTrainer
│   │   └── meta_learner.py             # Meta algoritmalar
│   ├── envs/                           # ORTAM YÖNETİMİ
│   │   ├── interface.py                # MetaTaskWrapper (ABC) - SÖZLEŞME
│   │   └── wrappers.py                 # Gym ve Isaac implementasyonları
│   └── networks/                       # Policy, Value, Encoder Ağları
├── scripts/                            # ÜRETİM VE BATCH İŞLEMLER
│   ├── utils/
│   │   ├── sweep.py                    # WandB Hyperparam Sweep
│   │   ├── run_tests.py                # Pytest / Smoke Tests
│   │   └── eval_batch.py               # Çoklu seed değerlendirme
│   ├── phase_3_1/                      # IsaacLab scriptleri
│   └── phase_3_2/
└── notebooks/                          # PROTOTİP (GLUE CODE)
    ├── Phase_3_1_0_Reacher.ipynb       # Config -> Train -> Plot akışı
    └── sanity_checks.ipynb             # Env/Agent shape ve logic kontrolleri


# B. Versiyon ve Seed Yönetimi (src/utils/seeding.py)

import subprocess
import torch
import numpy as np
import random

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        return "UNKNOWN_VER"

def set_seed(seed: int, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

# C . Ortam Arayüzü
Tüm wrapperlar bu soyut sınıftan (ABC) türetilmek zorundadır. Kritik: IsaacLab n_envs > 1 döndürürken, Gym n_envs=1 olabilir. Trainer buna göre yazılmalıdır.

from abc import ABC, abstractmethod
import torch

class MetaTaskWrapper(ABC):
    """
    Tüm Meta-RL ortamları için zorunlu arayüz.
    """
    @property
    def is_vectorized(self):
        """Ortamın vektörel (IsaacLab) olup olmadığını döner. Default: False"""
        return False

    @abstractmethod
    def reset(self, task=None):
        """
        Args: task (dict, optional): Belirli bir göreve zorlar. None ise rastgele sample.
        Returns: obs (n_envs, obs_dim), info
        """
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def sample_task(self):
        """Yeni bir fiziksel parametre seti çeker (örn: friction, mass)."""
        pass

    @abstractmethod
    def get_task(self):
        """Loglama için mevcut task bilgisini döndürür."""
        pass



# D. Checkpoint İçeriği

Model dosyası (.pt) sadece ağırlıkları değil, deneyin kimliğini de taşımalıdır.

state = {
    'policy_state_dict': agent.policy.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'normalizer_state': env.obs_rms,  # Varsa
    'rng_state': { ... },             # Random number generator states
    'config': cfg_dict,               # Hangi ayarlarla eğitildi?
    'git_commit_hash': get_git_hash(),# Hangi kod sürümüydü?
    'run_id': run_id,
    'global_step': step
}

3. DEĞERLENDİRME VE RAPORLAMA

Protokol:

Task Split: Eğitim görevleri (Train Tasks) ve daha önce hiç görülmemiş arızalar (OOD - Out of Distribution Test Tasks) ayrılmalıdır.

İstatistik: Her deney en az 3 farklı seed ile tekrarlanmalıdır.

Raporlama: Sonuçlar Mean Reward ± Std Dev (%95 Confidence Interval) formatında sunulur.

Araç: src.utils.reporting modülü sonuçları Pandas DataFrame'e alır ve df.to_latex() ile tez formatına çevirir.

4. DONANIM YÖNETİMİ (src/utils/device.py)

Mac kullanıcıları için float32 zorlaması hayati önem taşır.

import torch
import platform
import os
import logging

def get_optimal_device() -> torch.device:
    if platform.system() == "Darwin":
        os.environ['MUJOCO_GL'] = 'glfw'
        # MPS float64 desteklemez, float32'ye zorla
        torch.set_default_dtype(torch.float32)
        return torch.device("mps")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")