"""
GÖREV: Play/Inference (DYNAMIC STEP TEST)
AMAÇ: Robotun farklı hız komutlarına (0.5, 2.5, 0.0) anlık tepkisini ölçmek.
"""

import argparse
import os
import torch
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
from isaaclab.managers import ObservationTermCfg as ObsTerm 

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Marker
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.shapes import ConeCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

try:
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
except ImportError:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.ant.ant_env_cfg import AntEnvCfg

# --- 3. CONFIG ---
@configclass
class AntCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.0), # Eğitimle uyumlu
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class AntTrackingEnvCfg(AntEnvCfg):
    def __init__(self):
        super().__init__()
        self.commands = AntCommandsCfg()
        self.scene.num_envs = args_cli.num_envs
        self.sim.device = "cuda:0"
        
        # Gözlem Düzeltmesi (Blind Fix)
        if hasattr(self.observations, "policy"):
            self.observations.policy.velocity_commands = ObsTerm(
                func=mdp.generated_commands, 
                params={"command_name": "base_velocity"}
            )
        if hasattr(self.rewards, "progress"): self.rewards.progress = None

# --- 4. Marker ---
def create_cone_marker_cfg(color, prim_path):
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={"pointer": ConeCfg(radius=0.10, height=0.4, visual_material=PreviewSurfaceCfg(diffuse_color=color))}
    )

# --- 5. Model Bulucu ---
def get_latest_run_dir(root_dir="models"):
    if not os.path.exists(root_dir): return None
    all_zips = list(Path(root_dir).rglob("final_model.zip"))
    if not all_zips: return None
    latest_file = max(all_zips, key=os.path.getmtime)
    return latest_file.parent

# --- 6. Ana Fonksiyon ---
def main():
    run_dir = get_latest_run_dir()
    if not run_dir:
        print("[HATA] Model bulunamadı.")
        return

    model_path = os.path.join(run_dir, "final_model.zip")
    norm_path = os.path.join(run_dir, "vec_normalize.pkl")

    print(f"🚀 [MODEL]: {model_path}")
    
    # Ortam
    try:
        env_cfg = AntTrackingEnvCfg()
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"[HATA] {e}")
        return

    env = Sb3VecEnvWrapper(base_env)

    # Normalizasyon
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"👓 [NORM]: Yüklendi.")

    model = PPO.load(model_path, env=env)

    # Markerlar
    vel_marker = VisualizationMarkers(create_cone_marker_cfg((1.0, 0.0, 0.0), "/Visuals/Velocity"))
    cmd_marker = VisualizationMarkers(create_cone_marker_cfg((0.0, 1.0, 0.0), "/Visuals/Command"))
    vel_marker.set_visibility(True)
    cmd_marker.set_visibility(True)

    # Komut Nesnesi
    cmd_mgr = base_env.command_manager
    command_term = None
    term_names = list(cmd_mgr.active_terms.keys()) if isinstance(cmd_mgr.active_terms, dict) else cmd_mgr.active_terms
    target_name = next((s for s in term_names if "vel" in s), term_names[0] if term_names else None)

    if target_name:
        if hasattr(cmd_mgr, "get_term"): command_term = cmd_mgr.get_term(target_name)
        elif hasattr(cmd_mgr, "_terms"): command_term = cmd_mgr._terms.get(target_name)
        elif isinstance(cmd_mgr.active_terms, dict): command_term = cmd_mgr.active_terms.get(target_name)

    print("-" * 60)
    print("TEST: DİNAMİK HIZ DEĞİŞİMİ (0.5 -> 2.5 -> 0.0 -> 1.0)")
    print("-" * 60)

    obs = env.reset()
    device = base_env.device
    
    # Kickstart
    try:
        root_state = base_env.scene["robot"].data.default_root_state.clone()
        root_state[:, 2] += 0.5
        base_env.scene["robot"].write_root_state_to_sim(root_state)
        base_env.scene["robot"].reset()
    except: pass

    # Görsel Hazırlık
    tilt_quat = quat_from_angle_axis(torch.tensor(-np.pi/2, device=device).repeat(base_env.num_envs), torch.tensor([0., 1., 0.], device=device).repeat(base_env.num_envs, 1))

    step_count = 0
    current_target_val = 0.0

    while simulation_app.is_running():
        # --- DİNAMİK HEDEF BELİRLEME (SENARYO) ---
        if step_count < 200:
            current_target_val = 0.5  # Yavaş
            stage = "Yavaş (0.5)"
        elif step_count < 400:
            current_target_val = 2.5  # Çok Hızlı
            stage = "Koşu (2.5)"
        elif step_count < 600:
            current_target_val = 0.0  # Dur
            stage = "Dur (0.0)"
        else:
            current_target_val = 1.0  # Normal
            stage = "Normal (1.0)"

        target_vel = torch.tensor([current_target_val, 0.0, 0.0], device=device).repeat(base_env.num_envs, 1)

        # Enjeksiyon
        if command_term: command_term.command[:] = target_vel

        # Model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Görselleştirme
        if step_count % 2 == 0:
             robot_pos = base_env.scene["robot"].data.root_pos_w.clone()
             try: vels = base_env.scene["robot"].data.root_lin_vel_w 
             except: vels = base_env.scene["robot"].data.root_lin_vel_b
             
             speed = torch.norm(vels[:, :2], dim=1)
             heading = torch.atan2(vels[:, 1], vels[:, 0])
             q_head = quat_from_angle_axis(heading, torch.tensor([0.,0.,1.], device=device).repeat(base_env.num_envs, 1))
             
             marker_pos = robot_pos.clone(); marker_pos[:, 2] += 0.8
             scales = torch.ones((base_env.num_envs, 3), device=device) * 0.4
             scales[:, 2] = speed * 0.3 + 0.1 
             vel_marker.visualize(marker_pos, quat_mul(q_head, tilt_quat), scales)

             scales_cmd = scales.clone()
             scales_cmd[:, 2] = current_target_val * 0.3 + 0.1
             marker_pos[:, 2] += 0.2
             cmd_marker.visualize(marker_pos, tilt_quat, scales_cmd)

        if step_count % 20 == 0:
            vx = base_env.scene["robot"].data.root_lin_vel_b[0, 0].item()
            print(f"[{stage}] Adım {step_count} | Hız: {vx:.2f} / {current_target_val:.2f} m/s")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()