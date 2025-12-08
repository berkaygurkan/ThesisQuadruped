

"""
GÖREV: Faz 3.3 - FINAL COMPARISON (SINGLE SESSION FIX)
DURUM:
  1. Simülasyon KİLİTLENMESİ çözüldü (Ortam kapatılıp açılmıyor).
  2. Aynı ortam üzerinde önce PPO, sonra LSTM deneniyor.
  3. Kütle sabit (3.0x).
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# --- AYARLAR ---

# --- AYARLAR (LÜTFEN YOLLARI KONTROL EDİN) ---
# Baseline (PPO) Model Klasörü
BASELINE_DIR = "models/tracking_blindfix/PPO_BlindFix_Ant_20251208_2020"

# Meta-RL (LSTM) Model Klasörü
METARL_DIR = "models/meta_rl/MetaRL_LSTM_Ant_20251208_2238"

TEST_MASS_SCALE = 3.0
TARGET_VELOCITY = 1.5
TEST_STEPS = 500

# --- 1. IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Test sayısı")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Importlar ---
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    pass

try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

# --- 3. Ortam Konfigürasyonu ---
@configclass
class AntCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000.0, 1000.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(TARGET_VELOCITY, TARGET_VELOCITY), 
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class AntTestEnvCfg(AntEnvCfg):
    def __init__(self):
        super().__init__()
        self.commands = AntCommandsCfg()
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = "cuda:0"
        
        # KÜTLE ZORLAMA
        self.events.force_heavy_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (TEST_MASS_SCALE, TEST_MASS_SCALE),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        
        # Gözlem Düzeltmesi
        if hasattr(self.observations, "policy"):
            self.observations.policy.velocity_commands = ObsTerm(
                func=mdp.generated_commands, 
                params={"command_name": "base_velocity"}
            )
        if hasattr(self.rewards, "progress"): self.rewards.progress = None

# --- 4. Tekil Test Fonksiyonu ---
def evaluate_agent(base_env, model_dir, model_class, label):
    print(f"\n>>> TEST BAŞLIYOR: {label}")
    
    zip_path = os.path.join(model_dir, "final_model.zip")
    norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    
    if not os.path.exists(zip_path):
        print(f"[HATA] Model yok: {zip_path}")
        return []

    # 1. Normalizasyonu Yükle (Wrapper Oluştur)
    # base_env'i sarmalıyoruz. Her test için taze wrapper.
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, Sb3VecEnvWrapper(base_env))
        env.training = False
        env.norm_reward = False
        print("[BİLGİ] VecNormalize yüklendi.")
    else:
        env = Sb3VecEnvWrapper(base_env) # Normalizasyon yoksa düz sar
        print("[UYARI] Normalizasyon dosyası bulunamadı!")

    # 2. Modeli Yükle
    print(f"[MODEL] Yükleniyor...")
    model = model_class.load(zip_path, env=env)
    
    # 3. Koştur
    velocities = []
    obs = env.reset() # Bu sırada kütle 3.0x olacak (Config Event sayesinde)
    
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    for i in range(TEST_STEPS):
        if model_class == RecurrentPPO:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        obs, _, _, _ = env.step(action)
        episode_starts = np.zeros((env.num_envs,), dtype=bool)
        
        # Gerçek veriyi base_env'den çek
        vx = base_env.scene["robot"].data.root_lin_vel_b[0, 0].item()
        velocities.append(vx)
        
        if i % 100 == 0:
            print(f"[{label}] Adım {i}: {vx:.2f} m/s")
            
    return velocities

# --- 5. Ana Akış ---
def main():
    print(f"[INFO] Ortam Kuruluyor (Tek Seferlik)... Kütle: {TEST_MASS_SCALE}x")
    
    try:
        env_cfg = AntTestEnvCfg()
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[KRİTİK HATA] {e}")
        return

    # --- TEST 1: BASELINE (PPO) ---
    ppo_vels = evaluate_agent(base_env, BASELINE_DIR, PPO, "Baseline PPO")
    
    # --- TEST 2: META-RL (LSTM) ---
    # Ortamı kapatmadan sadece ajan değiştiriyoruz
    lstm_vels = evaluate_agent(base_env, METARL_DIR, RecurrentPPO, "Meta-RL LSTM")

    # --- GRAFİK ---
    if ppo_vels and lstm_vels:
        plt.figure(figsize=(10, 6))
        plt.axhline(y=TARGET_VELOCITY, color='green', linestyle='--', label='Hedef (1.5 m/s)', linewidth=2)
        
        plt.plot(ppo_vels, label='Baseline PPO', color='red', alpha=0.6)
        plt.plot(lstm_vels, label='Meta-RL LSTM', color='blue', linewidth=2)
        
        plt.title(f"Ağır Yük Testi ({TEST_MASS_SCALE}x Kütle)")
        plt.xlabel("Adım")
        plt.ylabel("Hız (m/s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 2.0)
        
        plt.savefig("comparison_single_session.png")
        print("\n[SONUÇ] Grafik kaydedildi: comparison_single_session.png")
    
    base_env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()