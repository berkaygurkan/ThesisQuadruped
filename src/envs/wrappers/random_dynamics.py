import gymnasium as gym
import numpy as np
from typing import Optional

class RandomDampingWrapper(gym.Wrapper):
    """
    MuJoCo Reacher-v5 ortamı için Domain Randomization Wrapper'ı.
    
    Meta-RL (RL^2) testi için her epizot başında eklem sönümlemesini (damping)
    belirlenen aralıkta rastgele değiştirir.
    
    Args:
        env (gym.Env): Paketlenmiş ortam.
        min_damping (float): Minimum damping katsayısı.
        max_damping (float): Maksimum damping katsayısı.
    """
    def __init__(self, env: gym.Env, min_damping: float, max_damping: float):
        super().__init__(env)
        self.min_d = min_damping
        self.max_d = max_damping
        
        # Gymnasium/MuJoCo modeline erişim kontrolü (Sanity Check)
        # Reacher-v5'te model verisi env.unwrapped.model altındadır.
        if not hasattr(self.env.unwrapped, 'model'):
            raise AttributeError("Bu wrapper sadece MuJoCo tabanlı ortamlarla çalışır!")
            
        self._current_damping = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # 1. RNG Seed'i ayarla (Reproducibility için kritik)
        super().reset(seed=seed)
        
        # 2. Yeni fizik parametrelerini belirle (Context Sampling)
        # Reacher modelinde genellikle 2 eklem vardır. Her ikisine de uyguluyoruz.
        # np_random Gymnasium'un kendi seed-safe random generator'ıdır.
        new_damping = self.np_random.uniform(self.min_d, self.max_d)
        
        # 3. MuJoCo Modeline Müdahale (Simülasyon Fiziğini Değiştir)
        # dof_damping: Degree of freedom damping coefficients
        self.env.unwrapped.model.dof_damping[:] = new_damping
        self._current_damping = new_damping
        
        # 4. Ortamı sıfırla ve gözlemi döndür
        return self.env.reset(seed=seed, options=options)

    def get_context(self) -> float:
        """Raporlama ve debug için mevcut sönümleme değerini döndürür."""
        return self._current_damping