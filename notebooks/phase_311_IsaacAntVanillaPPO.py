"""
GÖREV: Baseline PPO Eğitimi (Standart Ant)
DURUM: Argument Parser Hatası Giderildi.
"""

import argparse
import os
from datetime import datetime

# --- 1. IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher

# Argümanları al
parser = argparse.ArgumentParser(description="Simple PPO Baseline")

# [DÜZELTME] --headless argümanını buradan SİLDİK. 
# IsaacLab (AppLauncher) bunu otomatik yönetir.
parser.add_argument("--num_envs", type=int, default=64, help="Ortam sayısı")
parser.add_argument("--seed", type=int, default=42, help="Seed")

# IsaacLab argümanlarını enjekte et (Headless burada eklenir)
AppLauncher.add_app_launcher_args(parser)

# Ayrıştır
args_cli = parser.parse_args()

# Uygulamayı başlat
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Importlar (Simülasyon başladıktan sonra) ---
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

# Import Güvenliği
try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# --- 3. Yapılandırma ve Eğitim ---
def main():
    # Run ID
    run_name = f"SimplePPO_Ant_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # Log Klasörleri
    # Tez standartlarına göre 'logs/' altında tutuyoruz
    log_dir = os.path.join("logs", "baseline", run_name)
    model_dir = os.path.join("models", "baseline", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Ortam Ayarları (Override)
    env_cfg = AntEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"[INFO] Ortam Yükleniyor: {args_cli.num_envs} Paralel Ortam | Cihaz: {env_cfg.sim.device}")
    
    try:
        env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[KRİTİK HATA] Ortam yüklenemedi: {e}")
        return

    # SB3 Wrapper
    env = Sb3VecEnvWrapper(env)

    # Model Tanımı (PPO - MLP)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=24,
        batch_size=32768, #int(64 * args_cli.num_envs / 4), # Batch size ölçekleme
        learning_rate=3e-4,
        device="cuda"
    )

    print(f"[INFO] Eğitim Başlıyor: {run_name}")
    print(f"[INFO] Loglar: {log_dir}")
    print("-" * 50)

    # Callback: Her 500.000 adımda bir checkpoint al
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args_cli.num_envs, 1), 
        save_path=model_dir,
        name_prefix="ppo_ant"
    )

    try:
        model.learn(
            total_timesteps=20_000_000, # 5M Adım
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Final Kayıt
        model.save(os.path.join(model_dir, "final_model"))
        print("[BAŞARILI] Eğitim tamamlandı ve kaydedildi.")

    except KeyboardInterrupt:
        print("\n[İPTAL] Eğitim kullanıcı tarafından durduruldu.")
        model.save(os.path.join(model_dir, "interrupted_model"))
    
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()