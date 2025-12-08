"""
GÖREV: Faz 3.1.0 - Velocity Tracking PPO (BLINDNESS FIX)
DURUM:
  1. HATA: Robot hedef hızı görmüyordu (Obs içinde yoktu).
  2. ÇÖZÜM: 'generated_commands' terimi Observation Manager'a eklendi.
  3. ARTIK: Robot "Gitmem gereken hız bu" diyebilecek.
"""

import argparse
import os
import torch
import numpy as np
from datetime import datetime

# --- 1. IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096, help="Ortam sayısı")
parser.add_argument("--seed", type=int, default=42, help="Seed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Importlar ---
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

# --- 3. Ortam Konfigürasyonu ---

@configclass
class AntCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False, 
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.0),   
            lin_vel_y=(0.0, 0.0),   
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class AntTrackingEnvCfg(AntEnvCfg):
    def __init__(self):
        super().__init__()
        
        # 1. Komut Enjeksiyonu
        self.commands = AntCommandsCfg()
        
        # 2. [KRİTİK DÜZELTME] GÖZLEM UZAYINA KOMUT EKLEME
        # Mevcut policy grubuna 'velocity_commands' terimini ekliyoruz.
        # Bu sayede robot hedef hızı (vx, vy, w) görebilecek.
        if hasattr(self.observations, "policy"):
            self.observations.policy.velocity_commands = ObsTerm(
                func=mdp.generated_commands, 
                params={"command_name": "base_velocity"}
            )
        
        # 3. Simülasyon Ayarları
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = "cuda:0"
        
        # 4. ÖDÜL AYARLARI (Yumuşatılmış - Öğrenmeyi Kolaylaştırmak İçin)
        # Robotun önce hareket etmeyi öğrenmesi için cezaları hafiflettik.
        
        # Eski çöp ödülleri sil
        for key in ["progress", "alive", "energy", "joint_pos_limits", "track_lin_vel_xy_exp", "lin_vel_z_l2", "action_rate_l2"]:
            if hasattr(self.rewards, key): setattr(self.rewards, key, None)

        # Pozitifler
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=2.0, 
            params={"std": 0.5, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=1.0,
            params={"std": 0.5, "command_name": "base_velocity"},
        )
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.5) # Orta seviye alive

        # Negatifler (Cezalar Hafifletildi)
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1) # Çok az ceza
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001) # Çok az ceza
        self.rewards.energy = RewTerm(func=mdp.action_l2, weight=-0.001)

# --- 4. Eğitim Fonksiyonu ---
def main():
    run_name = f"PPO_BlindFix_Ant_{datetime.now().strftime('%Y%m%d_%H%M')}"
    log_dir = os.path.join("logs", "tracking_blindfix", run_name)
    model_dir = os.path.join("models", "tracking_blindfix", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"[INFO] Blindness Fix Mode | Envs: {args_cli.num_envs}")
    
    try:
        env_cfg = AntTrackingEnvCfg()
        env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[HATA] {e}")
        return

    # KONTROL: Gözlem Uzayı Büyüdü mü?
    # Eskiden 60 idi. Şimdi komutlar (3 float) eklendiği için 63 olmalı.
    print("-" * 50)
    print(f"[CHECK] Yeni Gözlem Boyutu: {env.observation_space['policy'].shape}")
    print("-" * 50)

    env = Sb3VecEnvWrapper(env)
    
    # Normalizasyon
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda",
        n_steps=128,             # Dengeli horizon
        batch_size=32768,       
        n_epochs=5,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
    )

    print(f"[INFO] Eğitim Başlıyor... Hedef: 50M Adım")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=int(5_000_000 / args_cli.num_envs),
        save_path=model_dir,
        name_prefix="ant_blind"
    )

    try:
        model.learn(
            total_timesteps=150_000_000, 
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        model.save(os.path.join(model_dir, "final_model"))
        env.save(os.path.join(model_dir, "vec_normalize.pkl"))
        print("[BAŞARILI] Eğitim Tamamlandı.")

    except KeyboardInterrupt:
        print("[İPTAL] Kaydediliyor...")
        model.save(os.path.join(model_dir, "interrupted_model"))
        env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()