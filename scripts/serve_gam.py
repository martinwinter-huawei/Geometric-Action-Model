#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OpenPI-compatible WebSocket policy server for Geometric-Action-Model (GAM).

Serves GAM's canonical DROID action convention to RoboLab (policies/gam/client.py).

Request (one msgpack-numpy dict per policy call):
    exterior_image   (H, W, 3) uint8   exterior camera, view 0 (DROID exterior_2_left in training)
    wrist_image      (H, W, 3) uint8   wrist camera, view 1
    eef_pos          (3,)  float       end-effector position, robot-base frame [m]
    eef_quat_wxyz    (4,)  float       end-effector orientation, robot-base frame
    gripper_position (1,)  float       gripper closedness, 0 = open, 1 = closed
    prompt           str               task instruction

Response:
    action  (N, 7) float32  [dx, dy, dz, drot_x, drot_y, drot_z, gripper_close]
            Per-step deltas at 15 Hz: position delta [m] and rotation vector [rad],
            both on robot-base axes (target = R(drot) @ current), and gripper
            close in [0, 1]. This is the inverse of the `droid_target` transform
            in src/robot/data/pretraining.py.

Usage:
    # Pipeline test without a checkpoint:
    python scripts/serve_gam.py --dummy --port 8000

    # Pretrained GAM (OXE + MimicGen + RoboCasa mixture, includes DROID):
    python scripts/serve_gam.py \
        --checkpoint checkpoints_hf/3da-libero-gam/pretrained/pretrained-gam.pt --port 8000

The policy is stateless across calls (history horizon 1, no committed history), so
one server can serve several RoboLab envs and needs no per-episode reset.
"""

import argparse
import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
# Parent of the `depth_anything_3` package (it is not pip-installed in this venv).
_DA3_DIR = _REPO_ROOT / "Depth-Anything-3" / "src"
for d in (str(_SRC_DIR), str(_DA3_DIR), str(_REPO_ROOT)):
    if d not in sys.path:
        sys.path.insert(0, d)

os.environ.setdefault("DA3_ROOT", str(_REPO_ROOT))
os.environ.setdefault("DA3_BASE_CKPT", str(_REPO_ROOT / "checkpoints" / "track4world_da3.pth"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("gam-policy-server")

# Number of initial requests whose proprio / actions are logged in full, so a
# smoke test shows whether the live inputs sit inside the training statistics.
_DEBUG_LOG_CALLS = 3


class DummyGAMPolicy:
    """Mock policy for fast pipeline & connection verification."""

    def __init__(self, action_horizon: int = 8):
        self.action_horizon = action_horizon
        self.metadata = {"policy": "gam-dummy", "action_space": "droid_base_delta"}
        logger.info("[DummyGAMPolicy] Initialized dummy policy (action_horizon=%d)", action_horizon)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        # Zero deltas, gripper open.
        return {"action": np.zeros((self.action_horizon, 7), dtype=np.float32)}


class GAMPolicyService:
    """Wraps a GAM Stage-1 checkpoint behind the RoboLab request/response contract above."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
        device: str = "cuda:0",
        action_stats_key: str | None = "droid",
        use_ema: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = str(Path(checkpoint_path).resolve())

        # LIBERO suite checkpoints ship a config.yaml next to gam.pt; the pretrained
        # checkpoint carries its config inside (`ckpt["config"]`), which load_policy merges.
        if config_path is None:
            candidate_cfg = Path(self.checkpoint_path).parent / "config.yaml"
            if candidate_cfg.exists():
                config_path = str(candidate_cfg)

        logger.info(
            "Loading GAM checkpoint from: %s (config=%s, device=%s, action_stats_key=%s)",
            self.checkpoint_path, config_path or "<embedded>", self.device, action_stats_key,
        )
        self.policy, self.policy_info = self._load_policy(config_path, action_stats_key, use_ema)
        # Decode and return a single model step (one 8-action chunk) per call.
        self.policy.active_action_horizon = 1
        self._num_calls = 0
        self.metadata = {
            "policy": "gam",
            "checkpoint": self.checkpoint_path,
            "action_space": "droid_base_delta",
            "action_stats_key": str(self.policy_info.get("action_stats_key", action_stats_key)),
        }
        logger.info("GAM policy loaded. Metadata: %s", self.policy_info)

    def _load_policy(self, config_path, action_stats_key, use_ema):
        from eval_libero_unified import load_policy

        args = argparse.Namespace(
            stage="auto",
            ckpt=self.checkpoint_path,
            config=config_path,
            action_stats_key=action_stats_key,
            action_stats_json=None,
            decode_visuals=False,
            detailed_video=False,
            # Stateless: the observation window is the current frame only, matching
            # the closed-loop evals in the released configs (history_horizon: 1).
            history_horizon=1,
            rollout_decode_horizon=1,
            rotate_policy_input=False,
            # DROID proprio is xyz + extrinsic-XYZ Euler (droid_state transform).
            proprio_orientation="rpy",
            # OXE/DROID instructions are trained verbatim, not LIBERO-normalized.
            text_prompt_normalization="raw_task_text",
            use_ema=use_ema,
        )

        cfg = OmegaConf.load(config_path) if config_path and os.path.exists(config_path) else OmegaConf.create({})
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False, mmap=True)

        policy, policy_info = load_policy(args, cfg, ckpt, self.device)
        return policy, policy_info

    @staticmethod
    def _to_gam_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Map the RoboLab request onto the robosuite-style keys GAM's live converter reads."""
        missing = [k for k in ("exterior_image", "wrist_image", "eef_pos", "eef_quat_wxyz", "gripper_position")
                   if k not in obs]
        if missing:
            raise KeyError(f"Request is missing {missing}; got keys {sorted(obs)}")

        quat_wxyz = np.asarray(obs["eef_quat_wxyz"], dtype=np.float32).reshape(4)
        closedness = float(np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1)[0])
        return {
            "agentview_image": np.asarray(obs["exterior_image"], dtype=np.uint8),
            "robot0_eye_in_hand_image": np.asarray(obs["wrist_image"], dtype=np.uint8),
            "robot0_eef_pos": np.asarray(obs["eef_pos"], dtype=np.float32).reshape(3),
            # GAM's converter expects xyzw.
            "robot0_eef_quat": quat_wxyz[[1, 2, 3, 0]],
            # The converter reads gripper width as qpos[0] - qpos[1]; DROID's proprio
            # slot is closedness in [0, 1], so encode it as [closedness, 0].
            "robot0_gripper_qpos": np.array([closedness, 0.0], dtype=np.float32),
        }

    @torch.inference_mode()
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        gam_obs = self._to_gam_obs(obs)
        prompt = str(obs.get("prompt", ""))

        actions = self.policy(gam_obs, task_desc=prompt)
        actions_np = actions.detach().float().cpu().numpy().reshape(-1, 7)
        if not np.all(np.isfinite(actions_np)):
            logger.warning("Non-finite GAM actions; replacing with zeros.")
            actions_np = np.nan_to_num(actions_np, nan=0.0, posinf=0.0, neginf=0.0)

        if self._num_calls < _DEBUG_LOG_CALLS:
            debug = getattr(self.policy, "last_debug", {}) or {}
            proprio = debug.get("proprio_raw")
            logger.info(
                "call %d prompt=%r proprio[xyz,rpy,grip]=%s",
                self._num_calls, prompt,
                np.round(proprio.numpy(), 3).tolist() if proprio is not None else None,
            )
            logger.info(
                "call %d actions[0]=%s |dpos|max=%.4f |drot|max=%.4f",
                self._num_calls, np.round(actions_np[0], 4).tolist(),
                float(np.abs(actions_np[:, :3]).max()), float(np.abs(actions_np[:, 3:6]).max()),
            )
        self._num_calls += 1
        return {"action": actions_np.astype(np.float32)}


def serve_forever(service, host: str, port: int) -> None:
    """Minimal OpenPI WebSocket protocol: send metadata on connect, then obs -> action."""
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve

    lock = threading.Lock()  # one GPU model; serialize requests across connections

    def handler(conn) -> None:
        packer = msgpack_numpy.Packer()
        peer = conn.remote_address  # unavailable once the socket is closed
        conn.send(packer.pack(service.metadata))
        logger.info("Client connected: %s", peer)
        while True:
            try:
                obs = msgpack_numpy.unpackb(conn.recv())
            except Exception:  # connection closed
                logger.info("Client disconnected: %s", peer)
                return
            try:
                with lock:
                    result = service.infer(obs)
                conn.send(packer.pack(result))
            except Exception:
                # A str frame is the OpenPI client's error channel.
                conn.send(traceback.format_exc())
                logger.exception("Inference failed")
                conn.close()
                return

    with serve(handler, host, port, compression=None, max_size=None) as server:
        logger.info("Serving on ws://%s:%d", host, port)
        server.serve_forever()


def serve():
    parser = argparse.ArgumentParser(description="Serve Geometric-Action-Model over WebSocket.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config.yaml; default: sidecar config.yaml, else the checkpoint's embedded config.")
    parser.add_argument("--action-stats-key", type=str, default="droid",
                        help="Per-dataset action/proprio statistics to (de)normalize with (default: droid). "
                             "'auto' lets the loader choose from the checkpoint config.")
    parser.add_argument("--dummy", action="store_true", help="Run with mock/dummy policy for testing.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--horizon", type=int, default=8, help="Action chunk length for the dummy policy.")
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
            action_stats_key=None if args.action_stats_key == "auto" else args.action_stats_key,
        )

    serve_forever(service, args.host, args.port)


if __name__ == "__main__":
    serve()
