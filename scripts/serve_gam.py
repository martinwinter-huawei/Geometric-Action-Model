#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OpenPI-compatible WebSocket policy server for Geometric-Action-Model (GAM).

Serves language-conditioned 7-DoF delta end-effector actions to RoboLab:
  observation -> [dx, dy, dz, droll, dpitch, dyaw, gripper]

Usage:
    # 1. Dummy mock policy (test pipeline & connection without checkpoint):
    python scripts/serve_gam.py --dummy --port 8000

    # 2. Real trained GAM checkpoint:
    python scripts/serve_gam.py --checkpoint checkpoints_hf/3da-libero-gam/spatial/gam.pt --port 8000
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_DA3_DIR = _REPO_ROOT / "Depth-Anything-3" / "src" / "depth_anything_3"
for d in (str(_SRC_DIR), str(_DA3_DIR), str(_REPO_ROOT)):
    if d not in sys.path:
        sys.path.insert(0, d)

os.environ.setdefault("DA3_ROOT", str(_REPO_ROOT))
os.environ.setdefault("DA3_BASE_CKPT", str(_REPO_ROOT / "checkpoints" / "track4world_da3.pth"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("gam-policy-server")


class DummyGAMPolicy:
    """Mock policy for fast pipeline & connection verification."""

    def __init__(self, action_horizon: int = 8):
        self.action_horizon = action_horizon
        logger.info("[DummyGAMPolicy] Initialized dummy policy (action_horizon=%d)", action_horizon)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        actions = np.zeros((self.action_horizon, 7), dtype=np.float32)
        actions[:, -1] = 1.0  # Open gripper
        return {"action": actions}


class GAMPolicyService:
    """Wraps Geometric-Action-Model (GAM) using native evaluation pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
        device: str = "cuda:0",
        history_horizon: str | int = "auto",
        rollout_decode_horizon: str | int = "full",
        use_ema: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = str(Path(checkpoint_path).resolve())

        # If config not specified, check for config.yaml in same folder
        if config_path is None:
            candidate_cfg = Path(self.checkpoint_path).parent / "config.yaml"
            if candidate_cfg.exists():
                config_path = str(candidate_cfg)

        logger.info("Loading GAM checkpoint from: %s (config=%s, device=%s)", self.checkpoint_path, config_path, self.device)
        self.policy, self.policy_info = self._load_policy(config_path, history_horizon, rollout_decode_horizon, use_ema)
        logger.info("GAM policy loaded successfully. Metadata: %s", self.policy_info)

    def _load_policy(self, config_path, history_horizon, rollout_decode_horizon, use_ema):
        from eval_libero_unified import load_policy

        args = argparse.Namespace(
            stage="auto",
            ckpt=self.checkpoint_path,
            config=config_path,
            action_stats_key=None,
            action_stats_json=None,
            decode_visuals=False,
            detailed_video=False,
            history_horizon=history_horizon,
            rollout_decode_horizon=rollout_decode_horizon,
            rotate_policy_input=False,
            proprio_orientation="auto",
            text_prompt_normalization="libero_hdf5_task_text",
            use_ema=use_ema,
        )

        cfg = OmegaConf.load(config_path) if config_path and os.path.exists(config_path) else OmegaConf.create({})
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        policy, policy_info = load_policy(args, cfg, ckpt, self.device)
        return policy, policy_info

    def reset_episode(self):
        """Reset internal history between evaluation episodes."""
        reset_fn = getattr(self.policy, "reset_episode", None)
        if callable(reset_fn):
            reset_fn()

    @torch.inference_mode()
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Inference entrypoint called by WebsocketPolicyServer."""
        agentview_img = obs.get("agentview_image") or obs.get("image")
        wrist_img = obs.get("robot0_eye_in_hand_image") or obs.get("wrist_image")
        prompt = obs.get("prompt", "")

        if agentview_img is None:
            raise ValueError("Missing 'image' or 'agentview_image' in observation.")

        if wrist_img is None:
            wrist_img = agentview_img

        # Proprioception extraction
        eef_pos = obs.get("robot0_eef_pos") or obs.get("eef_pos")
        eef_quat = obs.get("robot0_eef_quat") or obs.get("eef_quat")
        gripper_pos = obs.get("robot0_gripper_qpos") or obs.get("gripper_pos")

        if eef_pos is None:
            eef_pos = np.zeros(3, dtype=np.float32)
        if eef_quat is None:
            eef_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if gripper_pos is None:
            gripper_qpos = np.array([0.02, -0.02], dtype=np.float32)
        else:
            gripper_pos = np.asarray(gripper_pos, dtype=np.float32).flatten()
            if gripper_pos.size == 1:
                g = float(gripper_pos[0])
                gripper_qpos = np.array([g / 2.0, -g / 2.0], dtype=np.float32)
            elif gripper_pos.size >= 2:
                gripper_qpos = gripper_pos[:2]
            else:
                gripper_qpos = np.array([0.02, -0.02], dtype=np.float32)

        gam_obs = {
            "agentview_image": np.asarray(agentview_img, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.asarray(wrist_img, dtype=np.uint8),
            "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float32),
            "robot0_eef_quat": np.asarray(eef_quat, dtype=np.float32),
            "robot0_gripper_qpos": gripper_qpos,
        }

        # Query native GAM policy
        if bool(getattr(self.policy, "accepts_task_desc", False)):
            actions_tensor = self.policy(gam_obs, task_desc=prompt)
        else:
            actions_tensor = self.policy(gam_obs)

        actions_np = actions_tensor.detach().float().cpu().numpy()
        if actions_np.ndim == 3:
            actions_np = actions_np[0]

        return {"action": actions_np}


def serve():
    parser = argparse.ArgumentParser(description="Serve Geometric-Action-Model over WebSocket.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--config", type=str, default=None, help="Optional path to companion config.yaml.")
    parser.add_argument("--dummy", action="store_true", help="Run with mock/dummy policy for testing.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--horizon", type=int, default=8, help="Action chunk prediction horizon for dummy policy.")
    args = parser.parse_args()

    if args.dummy or args.checkpoint is None:
        if args.checkpoint is None and not args.dummy:
            logger.warning("No --checkpoint provided. Starting in --dummy mode.")
        service = DummyGAMPolicy(action_horizon=args.horizon)
    else:
        service = GAMPolicyService(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
        )

    try:
        from openpi_server.websocket_policy_server import WebsocketPolicyServer
    except ModuleNotFoundError:
        try:
            from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        except ModuleNotFoundError as e:
            logger.error("openpi-server is not installed. Install with .")
            raise e

    logger.info("Starting OpenPI WebSocket server on %s:%d...", args.host, args.port)
    server = WebsocketPolicyServer(policy=service, host=args.host, port=args.port, metadata={})
    server.serve_forever()


if __name__ == "__main__":
    serve()
