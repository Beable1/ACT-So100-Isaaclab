# CUDA Memory Optimization for Dual Camera Setup

## Problem
When using dual cameras (main + jaw), CUDA memory can fill up quickly, especially on 8GB GPUs, causing `torch.OutOfMemoryError`.

## Solutions Implemented ✅

### 1. Automatic GPU Memory Management
The code now automatically handles CUDA OOM errors:

#### **In test_ACT.py:**
- ✅ **Proactive cache clearing**: Every 10 steps in rollout loop
- ✅ **Periodic cache clearing**: Every 50 steps in policy inference
- ✅ **OOM recovery**: Automatic retry with cache clear on `env.step()` failure
- ✅ **Explicit tensor deletion** after use
- ✅ **Garbage collection** to free Python references
- ✅ **Batch dictionary cleanup** before creating new tensors

#### **In environment config:**
- ✅ **Interval event**: GPU cache cleared every ~3.33 seconds (200 steps)
- ✅ **Event-based cleanup**: Integrated with Isaac Lab's event manager
- ✅ **Per-observation cleanup**: GPU cache cleared **immediately after each camera image fetch**
- ✅ **Custom image wrapper**: `image_with_memory_cleanup()` wraps `mdp.image()`

### 2. Environment Variables (Before Running)
Add these to your shell or script:

```bash
# Enable PyTorch memory fragmentation avoidance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit PyTorch memory allocation (e.g., 90% of GPU)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Or combine both
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

### 3. Reduce Image Resolution (If Still Running Out)
In `test_ACT.py`, modify the LeRobotAdapter initialization:

```python
# Change from 512x512 to 256x256 or 384x384
resize_hw = (256, 256)  # Instead of (512, 512)
```

Or pass via CLI:
```bash
--rgb_height 256 --rgb_width 256
```

### 4. Use Mixed Precision (Future Enhancement)
Consider using PyTorch AMP (Automatic Mixed Precision):

```python
with torch.cuda.amp.autocast():
    act = self.policy.select_action(batch)
```

### 5. Monitor GPU Memory
Before running, check available memory:

```bash
nvidia-smi
```

Expected memory usage with dual 512x512 cameras:
- **Per step**: ~40-60 MB
- **Model weights**: ~500 MB - 2 GB (depending on architecture)
- **Isaac Sim overhead**: ~3-4 GB

**Total**: Expect 5-7 GB usage on 8GB GPU.

## Running the Script

```bash
# Set environment variable first
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Then run your script
python scripts/skrl/test_ACT.py \
    --task Template-So-100-fishrod-CubeLift-v0 \
    --checkpoint path/to/model \
    --horizon 500
```

## How It Works 🔄

### Automatic Recovery from OOM
When `torch.cuda.OutOfMemoryError` occurs:

1. **Catch the error** in `env.step()`
2. **Clear CUDA cache**: `torch.cuda.empty_cache()`
3. **Run garbage collector**: `gc.collect()`
4. **Retry the operation** once
5. **If still fails**: Stop gracefully with error message

### Proactive Prevention
- **Every 200 steps**: **FULL GPU MEMORY RESET** (aggressive cleanup)
  - `torch.cuda.empty_cache()`
  - `torch.cuda.synchronize()`
  - Triple `gc.collect()`
  - `torch.cuda.reset_peak_memory_stats()`
- **Every ~3.33 seconds**: Full GPU reset via environment event (~200 steps at 60Hz)

This focused 200-step reset approach ensures stable operation with dual cameras on limited GPU memory.

## Troubleshooting

### Still Getting OOM?
1. **Reduce batch size in model** (if applicable)
2. **Lower image resolution** to 256x256 or 384x384
3. **Use only one camera** temporarily to test
4. **Close other GPU applications** (browsers, etc.)
5. **Restart Isaac Sim** to clear any leaked memory
6. **Reduce cleanup interval** to every 5 steps (edit line 335 in `test_ACT.py`)

### Check Memory Usage During Run
In another terminal:
```bash
watch -n 1 nvidia-smi
```

This will show real-time GPU memory usage.

### Expected Behavior
- ✅ **Normal**: Periodic memory drops visible every 10 steps
- ⚠️ **Warning**: `[WARNING] CUDA OOM at step X, clearing cache and retrying...` (recoverable)
- ❌ **Error**: `[ERROR] CUDA OOM persists...` (need to reduce resolution or use single camera)

