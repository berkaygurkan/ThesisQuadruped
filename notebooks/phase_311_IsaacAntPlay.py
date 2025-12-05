"""
GÖREV: Play/Inference (HATA KORUMALI & NATIVE ENJECTION)
DURUM: 'list' vs 'dict' hatası giderildi. Komut sistemi otomatik algılanır.
"""

import argparse
import os
import glob
import torch
import sys
import numpy as np
from datetime import datetime

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
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO

# Ortam
try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

# --- 3. Model Bulucu ---
def get_latest_model_path(search_dir="models/baseline"):
    if not os.path.exists(search_dir): search_dir = "models"
    if not os.path.exists(search_dir): 
        print(f"[UYARI] '{search_dir}' bulunamadı.")
        return None 

    list_of_dirs = glob.glob(os.path.join(search_dir, "*"))
    if not list_of_dirs: return None
    
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    
    model_path = os.path.join(latest_dir, "final_model.zip")
    if not os.path.exists(model_path):
        checkpoints = glob.glob(os.path.join(latest_dir, "*.zip"))
        if checkpoints: model_path = max(checkpoints, key=os.path.getctime)
        else: return latest_dir + ".zip"
    print(f"[MODEL] {model_path}")
    return model_path

# --- 4. Ana Fonksiyon ---
def main():
    model_path = get_latest_model_path()
    if not model_path:
        print("[HATA] Model bulunamadı. Lütfen önce eğitim yapın.")
        return

    # Ortam Kurulumu
    env_cfg = AntEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"
    
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(base_env)
    model = PPO.load(model_path, env=env)
    
    print("-" * 60)
    print("SİMÜLASYON BAŞLADI - ROBUST COMMAND MODE")
    print("-" * 60)

    # --- [CRITICAL FIX] KOMUT TERİMİNİ GÜVENLİ YAKALAMA ---
    cmd_mgr = base_env.command_manager
    active_terms = cmd_mgr.active_terms
    command_term = None # Hedef terim objesi
    
    # 1. Veri tipi kontrolü ve Terimi Yakalama
    if isinstance(active_terms, dict):
        # Sözlük ise (İsim -> Obje)
        if len(active_terms) > 0:
            term_name = list(active_terms.keys())[0]
            command_term = active_terms[term_name]
            print(f"[DEBUG] Komut Sistemi: DICT (Terim: {term_name})")
    
    elif isinstance(active_terms, list):
        # Liste ise (Obje Listesi)
        if len(active_terms) > 0:
            command_term = active_terms[0]
            print(f"[DEBUG] Komut Sistemi: LIST (Index: 0)")
            
    else:
        print(f"[UYARI] Bilinmeyen active_terms tipi: {type(active_terms)}")

    # 2. Hata Kontrolü
    if command_term is None:
        print("[HATA] Aktif komut terimi bulunamadı! Robot rastgele hareket edebilir.")
    else:
        print(f"[BİLGİ] Komut Enjeksiyonu Hazır. Hedef: 1.5 m/s")

    obs = env.reset()
    
    # Hedef Hız Vektörü (Vx=1.5, Vy=0, w=0)
    target_vel = torch.tensor([1.5, 0.0, 0.0], device=base_env.device).repeat(base_env.num_envs, 1)
    
    step_count = 0

    while simulation_app.is_running():
        # --- ENJEKSİYON (HER ADIMDA) ---
        if command_term is not None:
            # Buffer'a doğrudan yazıyoruz. Bu işlem kesindir.
            command_term.command[:] = target_vel

        # Model Tahmini
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # --- İZLEME ---
        if step_count % 20 == 0:
            # Gerçek Hız
            robot_vel = base_env.scene["robot"].data.root_lin_vel_b
            vx = robot_vel[0, 0].item()
            
            # Hedef Hız (Okuma)
            current_target = -1.0
            if command_term is not None:
                # Command tensor şekli [N, 3] varsayılır
                current_target = command_term.command[0, 0].item()

            print(f"Robot 0 | Hız: {vx:.2f} m/s (Hedef: {current_target:.2f}) | Ödül: {reward[0]:.2f}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()