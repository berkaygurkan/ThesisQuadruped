""""
GÖREV: Faz 3.9 - CONSTANT HEAVY LOAD (3x MASS) + VELOCITY TRACKING (1.5 m/s)
AÇIKLAMA:
  - Robot simülasyonun başından itibaren 3x kütle (sanal kuvvet) altındadır.
  - Hedef hız 1.5 m/s olarak belirlenmiştir.
  - Grafik, ani değişim yerine "ağır yük altında hedefi yakalama" performansını ölçer.
"""
import time
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from stable_baselines3.common.utils import set_random_seed
from sklearn.decomposition import PCA 

# --- AYARLAR ---
BASELINE_DIR = "models/tracking_blindfix/PPO_BlindFix_Ant_20251208_2020"
METARL_DIR = "models/meta_rl/MetaRL_LSTM_Ant_20251208_2238"

TARGET_VELOCITY = 1.5      # HEDEF HIZ GÜNCELLENDİ
TOTAL_STEPS = 700
LOAD_FACTOR = 3.0          # 3x KÜTLE (NOMINAL * 3)
SMOOTHING_WINDOW = 40      # Grafik yumuşatma penceresi
SEED = 42

# --- 1. IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Importlar ---
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
try:
    from sb3_contrib import RecurrentPPO
except ImportError: pass

try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

# --- 3. Ortam ---
@configclass
class AntCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000.0, 1000.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(TARGET_VELOCITY, TARGET_VELOCITY), # Sabit 1.5 m/s
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class AntForceTestEnvCfg(AntEnvCfg):
    def __init__(self):
        super().__init__()
        self.commands = AntCommandsCfg()
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = "cuda:0"
        
        if hasattr(self.observations, "policy"):
            self.observations.policy.velocity_commands = ObsTerm(
                func=mdp.generated_commands, 
                params={"command_name": "base_velocity"}
            )
        # Progress reward'ı iptal edip sadece takibe odaklanmasını sağlıyoruz
        if hasattr(self.rewards, "progress"): self.rewards.progress = None

# --- 4. Sanal Yük Fonksiyonu ---
def apply_virtual_force(env, scale_factor):
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    device = base_env.device
    
    masses = robot.root_physx_view.get_masses()
    total_mass = torch.sum(masses, dim=-1)
    
    # F = m * (Scale - 1) * g  (Ekstra Yerçekimi Kuvveti)
    gravity = -9.81
    added_force_z = total_mass * (scale_factor - 1.0) * gravity
    
    forces = torch.zeros((base_env.num_envs, 1, 3), device=device)
    forces[:, 0, 2] = added_force_z 
    
    torques = torch.zeros((base_env.num_envs, 1, 3), device=device)
    body_ids = torch.tensor([0], device=device) 
    
    robot.set_external_force_and_torque(forces, torques, body_ids=body_ids)

# --- 5. Test Motoru ---
def evaluate_agent(base_env, model_dir, model_class, label):
    print(f"Test ediliyor: {label} (Target: {TARGET_VELOCITY} m/s, Load: {LOAD_FACTOR}x)")
    zip_path = os.path.join(model_dir, "final_model.zip")
    norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    
    if not os.path.exists(zip_path): return [], []

    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, Sb3VecEnvWrapper(base_env))
        env.training = False
        env.norm_reward = False
    else:
        env = Sb3VecEnvWrapper(base_env)

    model = model_class.load(zip_path, env=env)
    
    velocities = []
    lstm_history = [] 
    
    obs = env.reset()
    
    try:
        base_env.scene["robot"].reset()
    except: pass

    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    for i in range(TOTAL_STEPS):
        # [DEĞİŞİKLİK] Yük en baştan itibaren sürekli uygulanıyor
        apply_virtual_force(env, LOAD_FACTOR)

        if model_class == RecurrentPPO:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            if lstm_states is not None:
                h_state = lstm_states[0][-1][0]
                hidden_vector = h_state.cpu().numpy() if isinstance(h_state, torch.Tensor) else h_state
                lstm_history.append(hidden_vector)
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        obs, _, _, _ = env.step(action)
        
        base_env.render()
        
        episode_starts = np.zeros((env.num_envs,), dtype=bool)
        
        vx = base_env.scene["robot"].data.root_lin_vel_b[0, 0].item()
        velocities.append(vx)
            
    return velocities, np.array(lstm_history)

# --- 6. Hesaplama ve Çizimler ---
def plot_tracking_performance(ppo_data, meta_data):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300
    })

    # Rolling Mean
    ppo_smooth = pd.Series(ppo_data).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
    meta_smooth = pd.Series(meta_data).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Lines
    ax.plot(ppo_smooth, color="#000000", linestyle='--', linewidth=2.0, label='PPO (Baseline)')
    ax.plot(meta_smooth, color="#000000", linestyle='-', linewidth=2.0, label='Meta-RL (LSTM)')
    
    # Target Line
    ax.axhline(y=TARGET_VELOCITY, color='black', linestyle='-.', linewidth=2, label=f'Target ({TARGET_VELOCITY} m/s)')
    
    # Error Area (Meta-RL vs Target)
    ax.fill_between(range(len(meta_smooth)), meta_smooth, TARGET_VELOCITY, color='#D32F2F', alpha=0.1)

    # RMSE Calculation (Last 200 steps - Steady State)
    last_steps = 200
    if len(meta_smooth) > last_steps:
        meta_rmse = np.sqrt(np.mean((meta_smooth[-last_steps:] - TARGET_VELOCITY)**2))
        ppo_rmse = np.sqrt(np.mean((ppo_smooth[-last_steps:] - TARGET_VELOCITY)**2))
        
        info_text = (
            f"Steady-State RMSE (Last {last_steps} steps):\n"
            f"• Meta-RL: {meta_rmse:.3f}\n"
            f"• PPO: {ppo_rmse:.3f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.15, info_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    ax.set_xlabel('Simulation Steps')
    ax.set_ylabel('Linear Velocity (m/s)')
    ax.set_title(f'Velocity Tracking under Heavy Load ({LOAD_FACTOR}x Mass)')
    ax.set_ylim(0.0, TARGET_VELOCITY * 1.3)
    ax.set_xlim(0, TOTAL_STEPS)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    plt.savefig("academic_constant_load_tracking.png")
    plt.savefig("academic_constant_load_tracking.pdf")
    print("\n[SONUÇ] Hız takip grafiği kaydedildi: academic_constant_load_tracking.png")

def plot_latent_space_trajectory(lstm_data):
    if len(lstm_data) == 0: return

    print("[INFO] Latent Space PCA analizi yapılıyor...")
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(lstm_data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color mapping based on time (Progress)
    sc = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], 
               c=np.arange(len(reduced_data)), cmap='plasma', alpha=0.7, s=30)
    
    ax.scatter(reduced_data[0, 0], reduced_data[0, 1], c='green', marker='^', s=150, edgecolors='black', label='Start')
    ax.scatter(reduced_data[-1, 0], reduced_data[-1, 1], c='black', marker='X', s=150, edgecolors='white', label='End')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Time Steps')
    
    ax.set_title(f'Latent Space Evolution ({LOAD_FACTOR}x Mass)', fontsize=14)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("latent_space_constant_load.png")

# --- 7. Ana Akış ---
def main():
    print(f"[INFO] Global Seed Ayarlanıyor: {SEED}")
    set_random_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[INFO] Ortam Kuruluyor: Target={TARGET_VELOCITY} m/s, Mass={LOAD_FACTOR}x")
    try:
        env_cfg = AntForceTestEnvCfg()
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[HATA] {e}")
        return

    # Veri Toplama
    ppo_vels, _ = evaluate_agent(base_env, BASELINE_DIR, PPO, "PPO")
    lstm_vels, lstm_history = evaluate_agent(base_env, METARL_DIR, RecurrentPPO, "Meta-RL")
    
    base_env.close()

    # Grafikler
    if len(ppo_vels) > 0 and len(lstm_vels) > 0:
        plot_tracking_performance(ppo_vels, lstm_vels)
    
    if len(lstm_history) > 0:
        plot_latent_space_trajectory(lstm_history)
    
    simulation_app.close()

if __name__ == "__main__":
    main()