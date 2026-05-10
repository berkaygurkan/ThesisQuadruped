"""
GÖREV: FAZ 4.0 - ROBUSTNESS ANALYSIS (LARGE SCALE STATISTICAL TEST)
SENARYO:
  1. Standard Test: Nominal ağırlık (1.0x).
  2. OOD Test: Aşırı ağırlık (5.0x) - Başlangıçtan itibaren.
ÇIKTI:
  - Ortalama Ödül ve Standart Sapma karşılaştırmalı Bar Chart.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm # İlerleme çubuğu için
from stable_baselines3.common.utils import set_random_seed

# --- AYARLAR ---
BASELINE_DIR = "models/tracking_blindfix/PPO_BlindFix_Ant_20251208_2020"
METARL_DIR = "models/meta_rl/MetaRL_LSTM_Ant_20251208_2238"

NUM_TEST_EPISODES = 100   # Kaç epizot test edilecek (İstatistiksel güç için en az 50-100)
OOD_MASS_FACTOR = 5.0     # OOD Zorluğu (5 Katı Ağırlık)
SEED = 42

# --- 1. IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
# Hız için num_envs arttırıyoruz (Paralel test)
parser.add_argument("--num_envs", type=int, default=64) 
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
        debug_vis=False, # Hız için kapalı
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), # Sabit hız hedefi
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class AntStatsEnvCfg(AntEnvCfg):
    def __init__(self):
        super().__init__()
        self.commands = AntCommandsCfg()
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = "cuda:0"
        # Render kapalı (Hızlı test için)
        self.viewer = None 
        
        if hasattr(self.observations, "policy"):
            self.observations.policy.velocity_commands = ObsTerm(
                func=mdp.generated_commands, 
                params={"command_name": "base_velocity"}
            )

# --- 4. Sanal Yük Fonksiyonu ---
def apply_continuous_force(env, scale_factor):
    """Her adımda çağrılır, robotu sürekli ağır hissettirir."""
    if scale_factor <= 1.01: return # 1.0 ise işlem yapma (Optimisazyon)

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

# --- 5. İstatistiksel Test Motoru ---
def collect_statistics(base_env, model_dir, model_class, label, mass_factor):
    print(f"\n[{label}] İstatistikler Toplanıyor... (Mass: {mass_factor}x)")
    
    zip_path = os.path.join(model_dir, "final_model.zip")
    norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    
    if not os.path.exists(zip_path): 
        print(f"[HATA] Model bulunamadı: {zip_path}")
        return [], 0

    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, Sb3VecEnvWrapper(base_env))
        env.training = False
        env.norm_reward = False
    else:
        env = Sb3VecEnvWrapper(base_env)

    model = model_class.load(zip_path, env=env)
    
    # Değişkenler
    num_envs = env.num_envs
    episode_rewards = []
    current_rewards = np.zeros(num_envs)
    episode_counts = 0
    
    obs = env.reset()
    
    # LSTM Durumları
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    # İlerleme çubuğu (toplam epizot sayısına göre)
    pbar = tqdm(total=NUM_TEST_EPISODES, desc=f"{label} ({mass_factor}x)")
    
    while episode_counts < NUM_TEST_EPISODES:
        # 1. Ağırlık Uygula (SÜREKLİ - 0. Adımdan itibaren)
        apply_continuous_force(env, mass_factor)
        
        # 2. Model Tahmini
        if model_class == RecurrentPPO:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        # 3. Adım
        obs, rewards, dones, infos = env.step(action)
        episode_starts = dones
        
        # 4. Ödül Takibi
        current_rewards += rewards
        
        # Biten epizotları kaydet
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0
                episode_counts += 1
                pbar.update(1)
                
                # LSTM state sıfırlama (Meta-RL için önemli)
                if model_class == RecurrentPPO and lstm_states is not None:
                    # hidden ve cell state'leri o env için sıfırla (Basit yöntem: hepsini sıfırla veya karmaşık maskeleme)
                    # SB3 bunu otomatik yapabilir ama emin olmak için:
                    pass 

        if episode_counts >= NUM_TEST_EPISODES:
            break
            
    pbar.close()
    
    # İstatistikler
    mean_rew = np.mean(episode_rewards)
    std_rew = np.std(episode_rewards)
    print(f"   -> Mean Reward: {mean_rew:.2f} +/- {std_rew:.2f}")
    
    return episode_rewards, mean_rew, std_rew

# --- 6. Grafik Çizimi ---
def plot_robustness_bar_chart(results):
    """
    results = {
        'PPO_Standard': (mean, std),
        'PPO_OOD': (mean, std),
        'Meta_Standard': (mean, std),
        'Meta_OOD': (mean, std)
    }
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'figure.dpi': 300
    })

    labels = ['Standard Environment\n(1.0x Gravity)', f'OOD Environment\n({OOD_MASS_FACTOR}x Gravity)']
    
    ppo_means = [results['PPO_Standard'][0], results['PPO_OOD'][0]]
    ppo_stds = [results['PPO_Standard'][1], results['PPO_OOD'][1]]
    
    meta_means = [results['Meta_Standard'][0], results['Meta_OOD'][0]]
    meta_stds = [results['Meta_Standard'][1], results['Meta_OOD'][1]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Çubuklar
    rects1 = ax.bar(x - width/2, ppo_means, width, yerr=ppo_stds, label='PPO (Baseline)', 
                    color='#D62728', alpha=0.8, capsize=5, edgecolor='black')
    
    rects2 = ax.bar(x + width/2, meta_means, width, yerr=meta_stds, label='Meta-RL (Ours)', 
                    color='#1F77B4', alpha=0.9, capsize=5, edgecolor='black')

    # Süslemeler
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Robustness Analysis: Performance under OOD Mass')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Değerleri çubukların üstüne yazma
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig("academic_robustness_bar.png")
    plt.savefig("academic_robustness_bar.pdf")
    print("\n[GRAFİK] Kaydedildi: academic_robustness_bar.png")

# --- 7. Ana Akış ---
def main():
    print(f"[INFO] Global Seed: {SEED}")
    set_random_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[INFO] Ortam Kuruluyor... ({args_cli.num_envs} Envs Paralel)")
    try:
        env_cfg = AntStatsEnvCfg()
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[HATA] {e}")
        return

    results = {}

    # 1. PPO - Standard
    _, mean, std = collect_statistics(base_env, BASELINE_DIR, PPO, "PPO Baseline", 1.0)
    results['PPO_Standard'] = (mean, std)

    # 2. PPO - OOD (Heavy)
    _, mean, std = collect_statistics(base_env, BASELINE_DIR, PPO, "PPO Baseline", OOD_MASS_FACTOR)
    results['PPO_OOD'] = (mean, std)

    # 3. Meta-RL - Standard
    _, mean, std = collect_statistics(base_env, METARL_DIR, RecurrentPPO, "Meta-RL", 1.0)
    results['Meta_Standard'] = (mean, std)

    # 4. Meta-RL - OOD (Heavy)
    _, mean, std = collect_statistics(base_env, METARL_DIR, RecurrentPPO, "Meta-RL", OOD_MASS_FACTOR)
    results['Meta_OOD'] = (mean, std)

    base_env.close()
    simulation_app.close()

    # Grafik Çiz
    plot_robustness_bar_chart(results)

if __name__ == "__main__":
    main()