import gymnasium as gym
import numpy as np

class PreviousActionWrapper(gym.Wrapper):
    """
    Gözlem uzayına bir önceki adımda uygulanan aksiyonu (a_{t-1}) ekler.
    Meta-RL ajanının 'System Identification' yapabilmesi için kritiktir.
    
    Yeni Gözlem: [State_t, Action_{t-1}]
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 1. Mevcut uzayları al
        obs_space = env.observation_space
        act_space = env.action_space
        
        # 2. Yeni gözlem boyutu = Eski Gözlem + Aksiyon Boyutu
        # Reacher (11) + Action (2) = 13 boyutlu input
        low = np.concatenate([obs_space.low, act_space.low])
        high = np.concatenate([obs_space.high, act_space.high])
        
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
        # İlk adım için boş aksiyon buffer'ı
        self.prev_action = np.zeros(act_space.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Reset anında önceki aksiyon sıfırlanır
        self.prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        # Input'u birleştir: [Obs, 0.0, 0.0]
        new_obs = np.concatenate([obs, self.prev_action]).astype(np.float32)
        return new_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # a_{t-1}'i mevcut gözleme ekle
        new_obs = np.concatenate([obs, self.prev_action]).astype(np.float32)
        
        # Bir sonraki adım için bugünkü aksiyonu sakla
        self.prev_action = action.astype(np.float32)
        
        return new_obs, reward, terminated, truncated, info