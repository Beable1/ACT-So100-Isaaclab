# Gripper Action Fix

## Problem
The gripper was not moving because there was a mismatch between the action values sent by the policy and what the environment expected:

- **Policy output**: Raw continuous values (e.g., -0.5 to 0.5)
- **Binarization**: Was converting to {-1, +1} 
- **Environment expectation**: BinaryJointPositionActionCfg expects {0.0, 0.8} for open/close

## Solution
Fixed the gripper action mapping to properly convert policy outputs to environment-expected values:

1. **Updated binarization**: Now converts to {0.0, 0.8} instead of {-1, +1}
2. **Added configurable parameters**: Can adjust open/close values and thresholds
3. **Improved hysteresis**: Better state switching logic to prevent dithering
4. **Enhanced debugging**: Added verbose gripper action tracking

## New Command Line Options

### Gripper Action Configuration
```bash
--gripper_open_value 0.8      # Value sent to environment for open (default: 0.8)
--gripper_close_value 0.0     # Value sent to environment for close (default: 0.0)
--gripper_open_thresh 0.05    # Threshold above which gripper opens (default: 0.05)
--gripper_close_thresh -0.05  # Threshold below which gripper closes (default: -0.05)
```

### Debugging
```bash
--debug_gripper               # Enable verbose gripper action debugging
--no_binarize_gripper         # Disable gripper binarization (use raw values)
--invert_gripper              # Invert open/close mapping if needed
```

### Gripper Control
```bash
--gripper_close_after_step 100    # Force gripper to close after step 100
--gripper_open_after_step 200     # Force gripper to open after step 200
--gripper_force_close             # Force gripper to stay closed throughout episode
--gripper_force_open              # Force gripper to stay open throughout episode
```

### Troubleshooting Large Gripper Values
If you see very large gripper values (e.g., 2420.7285), the script now:
1. **Clips large values** to [-1, 1] range automatically
2. **Provides detailed debugging** to track the issue
3. **Shows warnings** when clipping occurs

## Usage Examples

### Basic Usage (with fix)
```bash
python3 scripts/skrl/test_ACT.py \
  --task Template-So-100-CubeLift-v0 \
  --checkpoint /path/to/checkpoint \
  --binarize_gripper \
  --debug_gripper
```

### Custom Gripper Values
```bash
python3 scripts/skrl/test_ACT.py \
  --task Template-So-100-CubeLift-v0 \
  --checkpoint /path/to/checkpoint \
  --binarize_gripper \
  --gripper_open_value 1.0 \
  --gripper_close_value 0.0 \
  --gripper_open_thresh 0.1 \
  --gripper_close_thresh -0.1
```

### Disable Binarization (use raw values)
```bash
python3 scripts/skrl/test_ACT.py \
  --task Template-So-100-CubeLift-v0 \
  --checkpoint /path/to/checkpoint \
  --no_binarize_gripper
```

### Force Gripper to Close After Step 100
```bash
python3 scripts/skrl/test_ACT.py \
  --task Template-So-100-CubeLift-v0 \
  --checkpoint /path/to/checkpoint \
  --binarize_gripper \
  --gripper_close_after_step 100 \
  --debug_gripper
```

### Force Gripper to Stay Closed
```bash
python3 scripts/skrl/test_ACT.py \
  --task Template-So-100-CubeLift-v0 \
  --checkpoint /path/to/checkpoint \
  --gripper_force_close \
  --debug_gripper
```

## Testing
Run the test script to verify the gripper logic:
```bash
python3 scripts/test_gripper_fix.py
```

Expected output:
```
Step 0: Raw= -0.50 -> Binarized=  0.00 (CLOSE)
Step 1: Raw= -0.10 -> Binarized=  0.00 (CLOSE)
Step 2: Raw=  0.00 -> Binarized=  0.80 (OPEN)
Step 3: Raw=  0.10 -> Binarized=  0.80 (OPEN)
Step 4: Raw=  0.50 -> Binarized=  0.80 (OPEN)
Step 5: Raw=  0.00 -> Binarized=  0.80 (OPEN)
Step 6: Raw= -0.10 -> Binarized=  0.00 (CLOSE)
```

## Environment Configuration
The environment is configured with:
```python
gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["Jaw"],
    open_command_expr={"Jaw": 0.8},   # Open position
    close_command_expr={"Jaw": 0.0}   # Close position
)
```

## Troubleshooting

### Gripper Still Not Moving
1. Check if `--debug_gripper` shows proper binarization
2. Verify environment configuration matches gripper values
3. Try `--no_binarize_gripper` to use raw policy outputs
4. Check if `--invert_gripper` is needed
5. Use `--gripper_force_close` or `--gripper_force_open` to test gripper functionality

### Gripper Dithering
1. Increase hysteresis thresholds: `--gripper_open_thresh 0.1 --gripper_close_thresh -0.1`
2. Check if policy outputs are stable

### Wrong Open/Close Direction
1. Use `--invert_gripper` to swap open/close
2. Or adjust `--gripper_open_value` and `--gripper_close_value` 