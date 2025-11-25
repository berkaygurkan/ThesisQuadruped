import gymnasium as gym
import numpy as np

class Float32ObservationWrapper(gym.ObservationWrapper):
    """
    Mac (MPS) uyumluluğu için kritik Wrapper.
    Gymnasium'un varsayılan float64 gözlemlerini float32'ye indirger.
    """
    def __init__(self, env):
        super().__init__(env)
        # Gözlem uzayının (Observation Space) veri tipini güncelle
        # Bu, SB3'ün tensörleri otomatik olarak float32 başlatmasını sağlar.
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=old_space.low,
            high=old_space.high,
            shape=old_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        # Gelen veriyi float32'ye cast et
        return np.array(observation, dtype=np.float32)