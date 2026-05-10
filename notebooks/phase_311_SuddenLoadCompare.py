"""
GÖREV: Faz 3.8 - SUDDEN LOAD + LATENT SPACE (FIXED & FINAL)
REVİZYONLAR:
  1. 'numpy.ndarray' object has no attribute 'cpu' hatası GİDERİLDİ.
  2. Latent Space (z_t) analizi aktif.
  3. Akademik Hız Grafiği aktif.
"""
import time # En tepeye eklemeyi unutmayın
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

TARGET_VELOCITY = 1.0
TOTAL_STEPS = 700
LOAD_STEP = 350
LOAD_FACTOR = 6.0      # 6x Gravity
SMOOTHING_WINDOW = 60
RECOVERY_PCT = 0.90
SEED = 42              # Sabit Seed

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
            lin_vel_x=(TARGET_VELOCITY, TARGET_VELOCITY), 
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
        if hasattr(self.rewards, "progress"): self.rewards.progress = None

# --- 4. Sanal Yük Fonksiyonu ---
def apply_virtual_force(env, scale_factor):
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    device = base_env.device
    
    masses = robot.root_physx_view.get_masses()
    total_mass = torch.sum(masses, dim=-1)
    
    gravity = -9.81
    added_force_z = total_mass * (scale_factor - 1.0) * gravity
    
    forces = torch.zeros((base_env.num_envs, 1, 3), device=device)
    forces[:, 0, 2] = added_force_z 
    
    torques = torch.zeros((base_env.num_envs, 1, 3), device=device)
    body_ids = torch.tensor([0], device=device) 
    
    robot.set_external_force_and_torque(forces, torques, body_ids=body_ids)

# --- 5. Test Motoru (RENDER EKLENDİ) ---
# --- 5. Test Motoru (VIDEO FIX: RENDER + SLEEP) ---
def evaluate_agent(base_env, model_dir, model_class, label):
    print(f"Test ediliyor: {label}")
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
    
    # Başlangıç Reset
    try:
        rs = base_env.scene["robot"].data.default_root_state.clone()
        rs[:, 2] += 0.5
        base_env.scene["robot"].write_root_state_to_sim(rs)
        base_env.scene["robot"].reset()
    except: pass

    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    # --- DÖNGÜ BAŞLIYOR ---
    for i in range(TOTAL_STEPS):
        if i >= LOAD_STEP:
            apply_virtual_force(env, LOAD_FACTOR)

        if model_class == RecurrentPPO:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            if lstm_states is not None:
                h_state = lstm_states[0][-1][0]
                if isinstance(h_state, torch.Tensor):
                    hidden_vector = h_state.cpu().numpy()
                else:
                    hidden_vector = h_state
                lstm_history.append(hidden_vector)
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        obs, _, _, _ = env.step(action)
        
        # --- [KRİTİK EKLEME] ---
        # 1. Görüntüyü çiz (Render)
        base_env.render()
        
        # 2. Hızı Yavaşlat (Real-time Simulation)
        # 0.02 saniye bekleme (50 FPS), böylece simülasyon fırlayıp gitmez.
        # Kaydedici her kareyi yakalamaya fırsat bulur.
        #time.sleep(0.02) 
        # -----------------------

        episode_starts = np.zeros((env.num_envs,), dtype=bool)
        
        vx = base_env.scene["robot"].data.root_lin_vel_b[0, 0].item()
        velocities.append(vx)
            
    return velocities, np.array(lstm_history)

# --- 6. Hesaplama ve Çizimler ---
def calculate_recovery_step(velocities, load_step, target, threshold_pct):
    threshold = target * threshold_pct
    post_load = velocities[load_step:]
    for i, v in enumerate(post_load):
        if v >= threshold:
            return i + load_step 
    return None

def plot_academic_results(ppo_data, meta_data):
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

    ppo_smooth = pd.Series(ppo_data).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()
    meta_smooth = pd.Series(meta_data).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ppo_smooth, color="#000000", linestyle='--', linewidth=2.0, label='PPO (Baseline)')
    ax.plot(meta_smooth, color="#000000", linestyle='-', linewidth=2.0, label='Meta-RL (LSTM)')
    
    ax.axvline(x=LOAD_STEP, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axhline(y=TARGET_VELOCITY, color='green', linestyle=':', linewidth=2, label='Target Velocity')
    
    ax.axvspan(LOAD_STEP, TOTAL_STEPS, color='gray', alpha=0.1)
    ax.text(LOAD_STEP + 10, 0.2, "Sudden Load Zone", fontsize=10, style='italic')

    meta_recovery_idx = calculate_recovery_step(meta_smooth, LOAD_STEP + 30, TARGET_VELOCITY, RECOVERY_PCT)

    if meta_recovery_idx:
        t_adapt = meta_recovery_idx - LOAD_STEP
        val = meta_smooth[meta_recovery_idx]
        ax.scatter(meta_recovery_idx, val, color='#1F77B4', s=100, zorder=5, marker='o')
        
        ax.annotate(f'Adaptation: {t_adapt} steps', 
                    xy=(meta_recovery_idx, val), 
                    xytext=(meta_recovery_idx - 260, val - 0.5),
                    arrowprops=dict(facecolor='#1F77B4', shrink=0.05, width=1.5),
                    fontsize=12, fontweight='bold', color='#1F77B4')

    ax.set_xlabel('Simulation Steps')
    ax.set_ylabel('Linear Velocity (m/s)')
    ax.set_title(f'Adaptation to Sudden Load Increase ({LOAD_FACTOR}x Gravity)')
    ax.set_ylim(0.0, 1.5)
    ax.set_xlim(0, TOTAL_STEPS)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    plt.savefig("academic_sudden_load_v4.png")
    plt.savefig("academic_sudden_load_v4.pdf")
    print("\n[SONUÇ] Hız grafiği kaydedildi: academic_sudden_load_v4.png")

def plot_latent_space(lstm_data):
    if len(lstm_data) == 0:
        print("[UYARI] Latent veri yok, çizim atlanıyor.")
        return

    print("[INFO] Latent Space (z_t) PCA analizi yapılıyor...")
    
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(lstm_data)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    steps = np.arange(len(reduced_data))
    
    normal_indices = steps < LOAD_STEP
    ax.scatter(reduced_data[normal_indices, 0], reduced_data[normal_indices, 1], 
               c=steps[normal_indices], cmap='Blues', alpha=0.6, label='Normal Walking', s=20)
    
    heavy_indices = steps >= LOAD_STEP
    ax.scatter(reduced_data[heavy_indices, 0], reduced_data[heavy_indices, 1], 
               c=steps[heavy_indices], cmap='Reds', alpha=0.6, label='Heavy Load (Force)', s=20)
    
    # İşaretçiler
    ax.scatter(reduced_data[0, 0], reduced_data[0, 1], 
               c='green', marker='^', s=150, edgecolors='black', label='Start')
    
    ax.scatter(reduced_data[LOAD_STEP, 0], reduced_data[LOAD_STEP, 1], 
               c='yellow', marker='*', s=300, edgecolors='black', zorder=10, label='Load Impact')
    
    ax.scatter(reduced_data[-1, 0], reduced_data[-1, 1], 
               c='black', marker='X', s=150, edgecolors='white', label='End')

    # Oklar
    for i in range(0, len(reduced_data)-1, 40):
        ax.annotate("", xy=reduced_data[i+5], xytext=reduced_data[i],
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4))

    ax.set_title(f'Latent Space ($z_t$) Dynamics under {LOAD_FACTOR}x Gravity', fontsize=14)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("latent_space_trajectory.png")
    plt.savefig("latent_space_trajectory.pdf")
    print("[SONUÇ] Latent Space grafiği kaydedildi: latent_space_trajectory.png")

# --- 7. Ana Akış (Otomatik Kayıt İçin Temizlendi) ---
def main():
    # SEED Sabitleme
    print(f"[INFO] Global Seed Ayarlanıyor: {SEED}")
    set_random_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[INFO] Ortam Kuruluyor... Otomatik Kayıt Modu")
    try:
        env_cfg = AntForceTestEnvCfg()
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[HATA] {e}")
        return

    # --- Manuel Bekleme (Wait Loop) KALDIRILDI ---
    # --anim_recording parametreleri kullanıldığı için simülasyon
    # başlar başlamaz kayıt da otomatik başlayacak.
    
    print("Simülasyon başlıyor... (Kayıt parametreleri aktif)")

    # Veri Toplama
    ppo_vels, _ = evaluate_agent(base_env, BASELINE_DIR, PPO, "PPO")
    lstm_vels, lstm_history = evaluate_agent(base_env, METARL_DIR, RecurrentPPO, "Meta-RL")
    
    base_env.close()

    # Grafikler
    if len(ppo_vels) > 0 and len(lstm_vels) > 0:
        plot_academic_results(ppo_vels, lstm_vels)
    
    if len(lstm_history) > 0:
        plot_latent_space(lstm_history)
    
    simulation_app.close()

if __name__ == "__main__":
    main()