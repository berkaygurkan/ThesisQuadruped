"""
GÖREV: IsaacLab & Ant Ortamı - Sistem Sağlık Kontrolü (Sanity Check)
TİP: Test Scripti (DÜZELTİLMİŞ IMPORT YOLLARI)
"""

import argparse
import os

# --- 1. ADIM: IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab Ant Sanity Check")
parser.add_argument("--num_envs", type=int, default=4, help="Ortam sayısı")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[INFO] 1. IsaacLab Simülasyon Uygulaması Başlatıldı.")

# --- 2. ADIM: Importlar (Simülasyon Başladıktan Sonra) ---
import torch
from isaaclab.envs import ManagerBasedRLEnv

# [KRİTİK DÜZELTME] Import yolu güncellendi: locomotion -> classic
try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
    print("[BAŞARILI] AntEnvCfg başarıyla import edildi.")
except ImportError:
    # Alternatif yol (Bazı versiyonlarda direct altında olabilir, kontrol ediyoruz)
    print("[UYARI] Klasik yol başarısız, alternatif deneniyor...")
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Donanım Hedefi: {device}")

    # Ant Konfigürasyonu
    env_cfg = AntEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    
    print("[INFO] 2. Ant Ortamı Yükleniyor...")
    try:
        env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[KRİTİK HATA] Ortam yüklenemedi: {e}")
        return

    # --- 3. ADIM: Sanity Check ---
    print("-" * 40)
    print("[SANITY CHECK] Başlıyor...")
    obs, _ = env.reset()
    print(f"✔ Reset Başarılı. Gözlem Şekli: {obs['policy'].shape}") 
    print("-" * 40)

    # --- 4. ADIM: Görsel Test ---
    print("[INFO] 4. Simülasyon Döngüsü Başlıyor (Çıkış için Ctrl+C veya Pencereyi Kapat)...")

    try:
        while simulation_app.is_running():
            actions = 2 * torch.rand(env.num_envs, env.action_space.shape[1], device=device) - 1
            obs, rew, terminated, truncated, info = env.step(actions)
            
    except KeyboardInterrupt:
        print("\n[INFO] Test kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[HATA] Bir sorun oluştu: {e}")
    finally:
        env.close()
        simulation_app.close()
        print("[INFO] Simülasyon kapatıldı.")

if __name__ == "__main__":
    main()