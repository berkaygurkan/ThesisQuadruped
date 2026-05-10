"""
GÖREV: AKADEMİK LEARNING CURVE (BLACK & WHITE)
ÖZELLİKLER:
  1. Renk yerine Çizgi Tipi (Solid vs Dashed) kullanılır.
  2. Gölgelendirme Gri tonlamalıdır.
  3. TensorBoard verilerini okur.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- AYARLAR ---
BASELINE_DIR = "logs/tracking_blindfix/PPO_BlindFix_Ant_20251208_2020/PPO_1"
METARL_DIR = "logs/meta_rl/MetaRL_LSTM_Ant_20251208_2238/RecurrentPPO_1"

SMOOTHING_WINDOW = 50 
SHADED_ERRORS = True   

# --- 1. Veri Yükleme (Değişmedi) ---
def load_tensorboard_data(log_dir):
    print(f"[ARANIYOR] {log_dir}...")
    event_files = [os.path.join(root, f) for root, _, files in os.walk(log_dir) for f in files if "events.out.tfevents" in f]
    
    if not event_files:
        print(f"[HATA] '{log_dir}' içinde event dosyası bulunamadı!")
        return None

    all_steps, all_rewards = [], []
    for event_file in event_files:
        try:
            # Sadece skalerleri yükle (Version Safe)
            ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
            ea.Reload()
            tags = ea.Tags()['scalars']
            
            target_tag = 'rollout/ep_rew_mean'
            if target_tag not in tags:
                # Alternatif etiketler
                target_tag = next((t for t in ['eval/mean_reward', 'train/reward'] if t in tags), None)
            
            if target_tag:
                events = ea.Scalars(target_tag)
                for event in events:
                    all_steps.append(event.step)
                    all_rewards.append(event.value)
        except Exception as e:
            print(f"    [HATA] {e}")

    if not all_steps: return None

    df = pd.DataFrame({'timesteps': all_steps, 'reward': all_rewards}).sort_values(by='timesteps')
    df['reward_smooth'] = df['reward'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    df['reward_std'] = df['reward'].rolling(window=SMOOTHING_WINDOW, min_periods=1).std()
    return df

# --- 2. Formatlayıcılar ve Çizim (SİYAH BEYAZ) ---
def millions_formatter(x, pos):
    return f'{x*1e-6:.0f}M' # .0f yaparak virgülden sonrasını attım (daha temiz görünür)

def plot_learning_curves_bw(ppo_data, meta_data):
    # Akademik Stil
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'lines.linewidth': 2.0
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- PPO (Baseline) ---
    # Stil: Siyah, Kesikli Çizgi (Dashed)
    if ppo_data is not None:
        ax.plot(ppo_data['timesteps'], ppo_data['reward_smooth'], 
                color='black', linestyle='--', label='PPO (Baseline)')
        
        if SHADED_ERRORS:
            r_mean = ppo_data['reward_smooth']
            r_std = ppo_data['reward_std'].fillna(0)
            # Gri Gölge
            ax.fill_between(ppo_data['timesteps'], 
                            r_mean - r_std, r_mean + r_std, 
                            color='gray', alpha=0.3)

    # --- Meta-RL (Ours) ---
    # Stil: Siyah, Düz Çizgi (Solid)
    if meta_data is not None:
        ax.plot(meta_data['timesteps'], meta_data['reward_smooth'], 
                color='black', linestyle='-', linewidth=2.5, label='Meta-RL (LSTM)')
        
        if SHADED_ERRORS:
            r_mean = meta_data['reward_smooth']
            r_std = meta_data['reward_std'].fillna(0)
            # Daha Koyu Gri Gölge (Fark belli olsun diye)
            ax.fill_between(meta_data['timesteps'], 
                            r_mean - r_std, r_mean + r_std, 
                            color='black', alpha=0.15)

    # Eksenler
    ax.set_xlabel('Total Timesteps (Environment Interactions)')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Training Performance Comparison')
    
    # X Ekseni Formatı (Milyon)
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    
    # Grid (Daha silik gri)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig("academic_learning_curve_bw.png")
    plt.savefig("academic_learning_curve_bw.pdf")
    print("\n[SONUÇ] Grafik kaydedildi: academic_learning_curve_bw.png ve .pdf")

# --- 3. Ana Akış ---
def main():
    print("TensorBoard verileri okunuyor...")
    ppo_df = load_tensorboard_data(BASELINE_DIR)
    meta_df = load_tensorboard_data(METARL_DIR)
    
    if ppo_df is None and meta_df is None:
        print("[KRİTİK] Veri bulunamadı.")
        return

    plot_learning_curves_bw(ppo_df, meta_df)

if __name__ == "__main__":
    main()