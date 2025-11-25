import os
import yaml
import json
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import RecurrentPPO

from src.utils.device import get_optimal_device
# Yeni Wrapperlar
from src.envs.wrappers.dynamics import RandomDampingWrapper
from src.envs.wrappers.observations import PreviousActionWrapper

# ... (Float32Wrapper sınıfı aynı kalacak) ...

class ScientificTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = get_optimal_device()
        self.run_id = f"{self.config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Loglama yolları... (Önceki kodla aynı)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.log_dir = os.path.join(base_dir, "data", "logs", self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Config Dump (Reproducibility)
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    def _make_env(self, seed, rank):
        """Wrapper Sıralaması Çok Önemli!"""
        def _init():
            env = gym.make(self.config['env']['id'])
            
            # 1. Donanım Uyumluluğu (En iç katman)
            if self.device.type == "mps":
                # Float32Wrapper burada tanımlı olmalı
                pass 

            # 2. Fizik (Dynamics)
            env = RandomDampingWrapper(
                env, 
                min_damping=self.config['env']['wrappers'][0]['args']['min_damping'],
                max_damping=self.config['env']['wrappers'][0]['args']['max_damping']
            )
            
            # 3. Gözlem Zenginleştirme (Previous Action)
            # DİKKAT: Bu wrapper float32 dönüşümünden sonra gelmeli ki type bozulmasın.
            env = PreviousActionWrapper(env)
            
            env.reset(seed=seed + rank)
            return env
        return _init

    def train(self):
        print(f"🚀 Starting Training: {self.run_id}")
        seeds = self.config['training']['seeds']
        
        for seed in seeds:
            set_random_seed(seed)
            env = SubprocVecEnv([self._make_env(seed, i) for i in range(self.config['env']['n_envs'])])
            env = VecMonitor(env, filename=os.path.join(self.log_dir, f"seed_{seed}_monitor.csv"))

            # --- ANALİST DÜZELTMESİ ---
            # Hardcoded değerler silindi. Config'den kwargs aktarımı yapılıyor.
            policy_kwargs = self.config['hyperparameters']['policy_kwargs']
            
            print(f"   -> Policy Kwargs Loaded: {policy_kwargs}") # Teyit Baskısı

            model = RecurrentPPO(
                policy=self.config['hyperparameters']['policy_type'],
                env=env,
                verbose=1,
                device=self.device,
                tensorboard_log=self.log_dir,
                learning_rate=self.config['hyperparameters']['learning_rate'],
                n_steps=self.config['hyperparameters']['n_steps'],
                batch_size=self.config['hyperparameters']['batch_size'],
                gamma=self.config['hyperparameters']['gamma'],
                gae_lambda=self.config['hyperparameters']['gae_lambda'],
                ent_coef=self.config['hyperparameters']['ent_coef'],
                policy_kwargs=policy_kwargs # <--- KRİTİK DÜZELTME BURADA
            )

            model.learn(
                total_timesteps=self.config['training']['total_timesteps'], 
                tb_log_name=f"LSTM_ActionInput_seed_{seed}",
                progress_bar=True
            )
            
            model.save(os.path.join(self.log_dir, f"model_seed_{seed}"))
            env.close()