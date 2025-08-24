## ACT-So100-IsaacLab — Model Testing Guide

This repository contains a test script to run trained policies (LeRobot/SmolVLA, robomimic, or plain PyTorch checkpoints) on the SO_100 task in Isaac Lab (e.g., `Template-So-100-CubeLift-v0`): `scripts/skrl/test_ACT.py`.

### Setup (Quick Start)
- **Dependencies**: Installed Isaac Sim / Isaac Lab and a working Python environment.
- **Install package** (optional but recommended):
```bash
pip install -e source/SO_100
```

### Test your own model
The example below tests a LeRobot/SmolVLA-style pretrained model (a folder containing `model.safetensors` and `train_config.json`) using the viewport camera.

```bash
python3 scripts/skrl/test_ACT.py \
  --task Template-So-100-CubeLift-v0 \
  --checkpoint /home/beable/lerobot/outputs/train/2025-08-18/02-38-09_act/checkpoints/080000/pretrained_model/ \
  --dataset_root /home/beable/IsaacLab-SO_100/dataset \
  --enable_cameras \
  --camera_source viewport \
  --camera_format rgb \
  --action_map abs2default \
  --arm_scales 3.1,0.8,2.1,0.8,1.5 \
  --axis_signs 1,1,1,1,1 \
  --binarize_gripper \
  --normalize_state_limits \
  --rate_limit 0.02 \
  --rate_limit_gripper 0.15 \
  --replan_every 16 \
  --dump_model_input_step 0 \
  --dump_model_input_dir outputs/model_input_rgb
```

Alternatives:
- **robomimic checkpoint** (auto-detected): `--checkpoint /path/to/robomimic/ckpt.pth`
- **Plain PyTorch checkpoint** (`.pt/.pth`): `--checkpoint /path/to/your_model.pth`

### Key flags (brief)
- **--checkpoint**: One of the following:
  - LeRobot/SmolVLA folder (contains `model.safetensors` + `train_config.json`)
  - robomimic `.pth`
  - Plain PyTorch `.pt/.pth`
- **--dataset_root**: Required for LeRobot models (reads dataset metadata). Example: `/home/beable/IsaacLab-SO_100/dataset`.
- **--enable_cameras**: Required for vision models.
- **--camera_source**: `sensor` (default) or `viewport`.
- **--camera_format**: `rgb` (default) or `bgr`.
- **--action_map**: `none | abs2norm | delta2norm | abs2default`. Typically `abs2default` works well.
- **--arm_scale / --arm_scales**: Action normalization scales (single value or comma-separated for 5 joints).
- **--axis_signs**: Axis signs, e.g., `1,1,1,1,1`.
- **--binarize_gripper / --no_binarize_gripper**: Threshold last action dim to {−1,+1}.
- **--normalize_state_limits**: Normalize observations to [-1,1] using joint limits.
- **--rate_limit, --rate_limit_gripper**: Per-step change limits for actions.
- **--replan_every**: Replanning frequency (steps) for LeRobot policy.
- **--dump_model_input_step, --dump_model_input_dir**: Save the exact model input image at a given step.
- Extras: `--dump_rgb`, `--dump_rgb_dir`, `--dump_rgb_every`, `--dump_rgb_processed`, `--dump_raw_cam_step`, `--dump_raw_cam_dir`, `--squash`, `--min_cmd`, `--pos_tol`, `--curr_from_env`, `--state_lower`, `--state_upper`.

### Outputs
- Visual logs and diagnostics are written under `outputs/` (e.g., `outputs/model_input_rgb`, `outputs/eval_rgb`).

### Tips
- If the viewport image is black/empty, try `--camera_source sensor`.
- If the arm stalls near target, try: `--min_cmd 0.05 --pos_tol 0.02`.
- If actions saturate, try `--squash` and/or adjust `--action_gain`.

### Your preferred command (exact)
```bash
python3 /home/beable/IsaacLab-SO_100/scripts/skrl/test_ACT.py   --task Template-So-100-CubeLift-v0   --checkpoint /home/beable/lerobot/outputs/train/2025-08-18/02-38-09_act/checkpoints/080000/pretrained_model/   --dataset_root /home/beable/IsaacLab-SO_100/dataset   --enable_cameras    --action_map abs2default  --axis_signs 1,1,1,1,1 --binarize_gripper --replan_every 48 --rate_limit 0.02 --rate_limit_gripper 0.15   --camera_source viewport   --dump_model_input_step 0 --dump_model_input_dir outputs/model_input_rgb --binarize_gripper  --arm_scales 3.1,0.8,2.1,0.8,1.5 --normalize_state_limits --replan_every 16
``` 