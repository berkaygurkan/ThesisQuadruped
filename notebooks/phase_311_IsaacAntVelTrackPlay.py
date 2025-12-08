"""
GÖREV: Play/Inference (BULLETPROOF COMMAND INJECTION)
DURUM:
  1. 'str' object hatası GİDERİLDİ.
  2. Komut nesnesi (_terms veya get_term) üzerinden güvenli çekiliyor.
  3. Model ismi ve tarihi belirgin şekilde raporlanıyor.
"""

import argparse
import os
import glob
import torch
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# --- 1. IsaacLab Başlatıcı ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4, help="Ortam sayısı")
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

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO

# Base Config
try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

# --- 3. CONFIG PARITY (Train Koduyla Birebir Aynı) ---
@configclass
class AntCommandsCfg:
    """Robot için hız komutlarını tanımlar."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True, # Play modunda okları görelim
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class AntTrackingEnvCfg(AntEnvCfg):
    def __init__(self):
        super().__init__()
        self.commands = AntCommandsCfg() # Komut sistemini ekle
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = "cuda:0"
        
        # Ödül yapısını da benzetelim (Hata almamak için)
        if hasattr(self.rewards, "progress"): self.rewards.progress = None

# --- 4. Akıllı Model Bulucu ---
def get_latest_model_path(root_dir="models"):
    if not os.path.exists(root_dir):
        # Belki direkt zip'tir
        if os.path.exists("models/final_model.zip"): return "models/final_model.zip"
        return None

    # Recursive arama (tracking_production, tracking_baseline vb.)
    all_zips = list(Path(root_dir).rglob("*.zip"))
    if not all_zips: return None
    
    # En yeniye göre sırala
    latest_file = max(all_zips, key=os.path.getmtime)
    return str(latest_file)

# --- 5. Ana Fonksiyon ---
def main():
    # Model Yolu Raporu
    try:
        model_path = get_latest_model_path()
        if not model_path:
            print("[HATA] Model dosyası bulunamadı.")
            return
        
        print("\n" + "="*60)
        print(f"🚀 [MODEL]: {model_path}")
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"📅 [TARİH]: {mod_time}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"[HATA] {e}")
        return

    # Ortamı Başlat
    try:
        env_cfg = AntTrackingEnvCfg()
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[ORTAM HATASI] {e}")
        return

    env = Sb3VecEnvWrapper(base_env)
    model = PPO.load(model_path, env=env)
    
    print("-" * 60)
    print("SİMÜLASYON BAŞLADI - VELOCITY TRACKING")
    print("-" * 60)

    # --- KOMUT NESNESİNİ BULMA (CERRAHİ YÖNTEM) ---
    cmd_mgr = base_env.command_manager
    velocity_term = None
    term_name = "base_velocity"

    # Plan A: get_term fonksiyonu
    if hasattr(cmd_mgr, "get_term"):
        try:
            velocity_term = cmd_mgr.get_term(term_name)
            print("[DEBUG] Plan A (get_term) Başarılı.")
        except: pass

    # Plan B: Gizli _terms sözlüğü (En garantisi)
    if velocity_term is None and hasattr(cmd_mgr, "_terms"):
        if term_name in cmd_mgr._terms:
            velocity_term = cmd_mgr._terms[term_name]
            print("[DEBUG] Plan B (_terms dict) Başarılı.")
    
    # Plan C: Active terms sözlük ise
    if velocity_term is None and isinstance(cmd_mgr.active_terms, dict):
        if term_name in cmd_mgr.active_terms:
            velocity_term = cmd_mgr.active_terms[term_name]
            print("[DEBUG] Plan C (active_terms dict) Başarılı.")

    # Plan D: Hiçbiri olmadıysa, active_terms listesinden ismi alıp _terms'den çek
    if velocity_term is None and isinstance(cmd_mgr.active_terms, list):
        # Listede isim varsa
        names = [str(x) for x in cmd_mgr.active_terms] # Garanti string
        print(f"[DEBUG] Bulunan Terim İsimleri: {names}")
        
        # 'vel' içeren ilk ismi bul
        found_name = next((s for s in names if "vel" in s), None)
        if found_name and hasattr(cmd_mgr, "_terms"):
             velocity_term = cmd_mgr._terms[found_name]
             print(f"[DEBUG] Plan D (List Search -> {found_name}) Başarılı.")

    if velocity_term is None:
        print("[KRİTİK HATA] Komut nesnesine erişilemedi! Hız sabitlenemeyecek.")
    else:
        print(f"[BİLGİ] Komut Enjeksiyonu Aktif. Nesne: {type(velocity_term)}")

    obs = env.reset()
    
    # HEDEF: 1.5 m/s (Test için sabitliyoruz)
    target_vel_tensor = torch.tensor([1.5, 0.0, 0.0], device=base_env.device).repeat(base_env.num_envs, 1)

    step_count = 0
    while simulation_app.is_running():
        # --- ENJEKSİYON ---
        if velocity_term is not None:
            # Bellek alanına yaz
            velocity_term.command[:] = target_vel_tensor

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if step_count % 20 == 0:
            robot_vel = base_env.scene["robot"].data.root_lin_vel_b
            vx = robot_vel[0, 0].item()
            
            # Hedef Teyidi (Okuma)
            t = -99.9
            if velocity_term is not None:
                t = velocity_term.command[0, 0].item()

            print(f"Robot 0 | Hız: {vx:.2f} m/s (Hedef: {t:.2f}) | Ödül: {reward[0]:.2f}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()