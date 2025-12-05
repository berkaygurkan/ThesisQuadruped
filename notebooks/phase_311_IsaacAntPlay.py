"""
GÖREV: Eğitilmiş PPO Modelini Oynatma (Play/Inference)
DURUM: SB3 VecEnv API uyumsuzluğu giderildi.
"""

import argparse
import os
import sys
import glob
import time
from datetime import datetime

# --- 1. IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher

# Argümanlar
parser = argparse.ArgumentParser(description="Play/Inference Script")
parser.add_argument("--num_envs", type=int, default=4, help="Görselleştirme için ortam sayısı")
parser.add_argument("--seed", type=int, default=42, help="Seed")

# IsaacLab argümanlarını ekle
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Uygulamayı başlat
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Importlar ---
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO

# Ortam Konfigürasyonu
try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

# --- 3. Model Bulma Yardımcısı ---
def get_latest_model_path(search_dir="models/baseline"):
    if not os.path.exists(search_dir):
        # Eğer baseline yoksa, interrupt edilmiş modellere bak (debug için)
        if os.path.exists("models"):
             search_dir = "models"
        else:
             raise FileNotFoundError(f"Model klasörü bulunamadı: {search_dir}")
    
    list_of_dirs = glob.glob(os.path.join(search_dir, "*"))
    if not list_of_dirs:
        raise FileNotFoundError(f"{search_dir} altında hiç model klasörü yok.")
    
    # En yeni klasörü bul
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    
    # Model dosyasını ara
    model_path = os.path.join(latest_dir, "final_model.zip")
    if not os.path.exists(model_path):
        # Checkpointlere bak
        checkpoints = glob.glob(os.path.join(latest_dir, "*.zip"))
        if checkpoints:
            model_path = max(checkpoints, key=os.path.getctime)
        else:
            # Belki direkt klasörün içindedir (basit kaydetme)
            possible_model = latest_dir + ".zip"
            if os.path.exists(possible_model):
                model_path = possible_model
            else:
                raise FileNotFoundError(f"Model bulunamadı: {latest_dir}")
            
    print(f"[OTOMATİK SEÇİM] Yüklenen Model: {model_path}")
    return model_path

# --- 4. Ana Oynatma Fonksiyonu ---
def main():
    try:
        model_path = get_latest_model_path()
    except Exception as e:
        print(f"[HATA] {e}")
        print("Lütfen önce 'phase_311_IsaacAntVanillaPPO.py' ile eğitim yapın.")
        return

    # Ortam Ayarları
    env_cfg = AntEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Ortam Yükleniyor (GUI)... Env: {args_cli.num_envs}")
    
    # Ortamı oluştur ve sarmala
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env) # SB3 uyumluluğu

    # Modeli Yükle
    print("[INFO] Model yükleniyor...")
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"[HATA] Model yükleme başarısız: {e}")
        env.close()
        return

    # --- 5. Simülasyon Döngüsü ---
    print("-" * 50)
    print("[INFO] Simülasyon Başlıyor. (Çıkış: Pencereyi Kapat veya Ctrl+C)")
    print("-" * 50)

    # [DÜZELTME 1] SB3 Wrapper reset() sadece obs döner, tuple dönmez!
    obs = env.reset()

    step_count = 0
    try:
        while simulation_app.is_running():
            # Model tahmini
            action, _ = model.predict(obs, deterministic=True)
            
            # [DÜZELTME 2] SB3 Wrapper step() 4 değer döner: obs, reward, done, info
            # (Gymnasium'un 5 değerini 4'e indirger)
            obs, reward, done, info = env.step(action)
            
            # İsteğe bağlı: Hız bilgisi yazdırma (Info içindeyse)
            step_count += 1
            if step_count % 100 == 0:
                # Ödül ortalamasını yazdır
                mean_reward = reward.mean().item()
                print(f"Step: {step_count} | Ortalama Ödül: {mean_reward:.4f}")

    except KeyboardInterrupt:
        print("\n[BİLGİ] Kullanıcı çıkışı.")
    except Exception as e:
        print(f"\n[HATA] Simülasyon hatası: {e}")
    finally:
        env.close()
        simulation_app.close()
        print("[INFO] Kapatıldı.")

if __name__ == "__main__":
    main()