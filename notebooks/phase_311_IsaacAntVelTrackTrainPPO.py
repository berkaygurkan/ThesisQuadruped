"""
GÖREV: Faz 3.1.0 - Velocity Tracking PPO (MAX PERFORMANCE PRODUCTION)
DURUM:
  1. Ortam Sayısı: 4096 (Varsayılan)
  2. Batch Size: 32768 (GPU Doyurma Modu)
  3. Hedef: 50M Adım
  4. Robust Komut Sistemi: Aktif
"""

import argparse
import os
import torch
import numpy as np
from datetime import datetime

# --- 1. IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Velocity Tracking PPO Max Perf")
# [AYAR] Varsayılanı 4096 yaptık. GPU VRAM yetmezse 2048'e düşürün.
parser.add_argument("--num_envs", type=int, default=4096, help="Ortam sayısı")
parser.add_argument("--seed", type=int, default=42, help="Seed")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Headless modda başlat (Performans için şart)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Importlar ---
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# --- 3. Ortam Konfigürasyonu ---

@configclass
class AntCommandsCfg:
    """Robot için hız komutlarını tanımlar."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False, # Performans için KAPALI
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.0),   # Hız limitini artırdık (Daha agresif koşu)
            lin_vel_y=(0.0, 0.0),   
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class AntTrackingEnvCfg(AntEnvCfg):
    def __init__(self):
        super().__init__()
        
        # Komut Enjeksiyonu
        self.commands = AntCommandsCfg()
        
        # Simülasyon Ayarları
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = "cuda:0"
        
        # Ödül Fonksiyonu (Agresif Eğitim)
        if hasattr(self.rewards, "progress"): 
            self.rewards.progress = None
            
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=2.0, # Ödül ağırlığını artırdık
            params={"std": 0.5, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=1.0,
            params={"std": 0.5, "command_name": "base_velocity"},
        )
        if hasattr(self.rewards, "energy"): 
            self.rewards.energy.weight = -0.0002 # Enerji cezası çok düşük

# --- 4. Eğitim Fonksiyonu ---
def main():
    run_name = f"PPO_MaxPerf_Ant_{datetime.now().strftime('%Y%m%d_%H%M')}"
    log_dir = os.path.join("logs", "tracking_production", run_name)
    model_dir = os.path.join("models", "tracking_production", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"[INFO] Yüksek Performans Modu: {args_cli.num_envs} Envs | GPU: RTX 4070 Ti")
    
    try:
        env_cfg = AntTrackingEnvCfg()
        env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[KRİTİK HATA] {e}")
        return

    # --- Hızlı Teyit ---
    if hasattr(env, "command_manager"):
        # List/Dict uyumluluğu
        terms = env.command_manager.active_terms
        cmds = list(terms.keys()) if isinstance(terms, dict) else ["Term Count: " + str(len(terms))]
        print(f"[SYSTEM CHECK] Komut Sistemi Aktif: {cmds}")
    
    # SB3 Wrapper
    env = Sb3VecEnvWrapper(env)

    # --- HİPERPARAMETRELER (MAX PERF) ---
    # Buffer Size = 4096 * 24 = 98,304
    # Batch Size = 32,768 (Buffer'ı 3 kerede tüketir)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda",
        n_steps=24,             # Rollout uzunluğu (Kısa tutuldu ki VRAM şişmesin)
        batch_size=32768,       # GPU için optimize
        n_epochs=5,             # Her veriyle 5 tur eğitim
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2
    )

    print(f"[INFO] Eğitim Başlıyor... Hedef: 50 MİLYON Adım")
    print("-" * 50)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=int(5_000_000 / args_cli.num_envs), # Her 5M adımda bir kaydet
        save_path=model_dir,
        name_prefix="ant_prod"
    )

    try:
        model.learn(
            total_timesteps=500_000_000, 
            callback=checkpoint_callback,
            progress_bar=True
        )
        model.save(os.path.join(model_dir, "final_model"))
        print("[BAŞARILI] Üretim eğitimi tamamlandı.")

    except KeyboardInterrupt:
        print("[İPTAL] Eğitim durduruldu. Model kaydediliyor...")
        model.save(os.path.join(model_dir, "interrupted_model"))
    
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()