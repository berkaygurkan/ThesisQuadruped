"""
GÖREV: Faz 3.2.1 - Meta-RL (API FIX: SceneEntityCfg)
DURUM:
  1. 'randomize_mass' parametre hatası GİDERİLDİ.
  2. Eski 'asset_name' yerine yeni 'asset_cfg' standardı kullanıldı.
  3. Kütle randomizasyonu aktif.
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
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
# [YENİ] Varlık referansı için gerekli
from isaaclab.managers import SceneEntityCfg 

try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("Lütfen 'pip install sb3-contrib' komutunu çalıştırın.")

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
class AntMetaEnvCfg(AntEnvCfg):
    def __init__(self):
        super().__init__()
        
        self.commands = AntCommandsCfg()
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = "cuda:0"
        
        # --- [KRİTİK DÜZELTME] KÜTLE RANDOMİZASYONU (API UPDATE) ---
        # Eski yöntem: asset_name="robot", body_names=".*"
        # Yeni yöntem: asset_cfg=SceneEntityCfg("robot", body_names=".*")
        
        self.events.randomize_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"), # DÜZELTİLDİ
                "mass_distribution_params": (0.5, 2.5),
                "operation": "scale",
                "recompute_inertia": True, # Atalet momentini de güncelle
            },
        )
        
        self.events.reset_mass_randomization = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"), # DÜZELTİLDİ
                "mass_distribution_params": (0.5, 2.5),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )

        # Gözlem (Komut Ekleme)
        if hasattr(self.observations, "policy"):
            self.observations.policy.velocity_commands = ObsTerm(
                func=mdp.generated_commands, 
                params={"command_name": "base_velocity"}
            )

        # Ödül Fonksiyonu (Baseline ile Aynı)
        for key in ["progress", "alive", "energy", "joint_pos_limits", "track_lin_vel_xy_exp", "lin_vel_z_l2", "action_rate_l2"]:
            if hasattr(self.rewards, key): setattr(self.rewards, key, None)

        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp, weight=2.0, 
            params={"std": 0.5, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp, weight=1.0,
            params={"std": 0.5, "command_name": "base_velocity"},
        )
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.5)
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
        self.rewards.energy = RewTerm(func=mdp.action_l2, weight=-0.001)

# --- 4. Eğitim Fonksiyonu ---
def main():
    run_name = f"MetaRL_LSTM_Ant_{datetime.now().strftime('%Y%m%d_%H%M')}"
    log_dir = os.path.join("logs", "meta_rl", run_name)
    model_dir = os.path.join("models", "meta_rl", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"[INFO] Meta-RL (LSTM) | Kütle Randomizasyonu: [0.5x, 2.5x]")
    
    try:
        env_cfg = AntMetaEnvCfg()
        env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[KRİTİK HATA] Ortam oluşturulamadı: {e}")
        return

    env = Sb3VecEnvWrapper(env)
    
    # Normalizasyon (Baseline ile aynı)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    # --- RECURRENTPPO (LSTM) ---
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        lstm_hidden_size=256,
        n_lstm_layers=1,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda",
        n_steps=128,            
        batch_size=65536,       
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=policy_kwargs
    )

    print(f"[INFO] Eğitim Başlıyor... Hedef: 50M Adım")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=int(5_000_000 / args_cli.num_envs),
        save_path=model_dir,
        name_prefix="meta_ant"
    )

    try:
        model.learn(
            total_timesteps=150_000_000, 
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        model.save(os.path.join(model_dir, "final_model"))
        env.save(os.path.join(model_dir, "vec_normalize.pkl"))
        print("[BAŞARILI] Meta-RL Modeli Kaydedildi.")

    except KeyboardInterrupt:
        print("[İPTAL] Kaydediliyor...")
        model.save(os.path.join(model_dir, "interrupted_model"))
        env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()