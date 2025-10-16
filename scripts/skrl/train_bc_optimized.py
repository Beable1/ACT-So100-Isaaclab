#!/usr/bin/env python3
"""
train_bc_optimized.py  -- updated to use ResNet18 with ImageNet pretrained weights

This script now uses ResNet18 as the vision backbone instead of a custom CNN.
ResNet18 provides better feature extraction capabilities through transfer learning.

Recommended parameters for ResNet18:
- batch_size: 32-64 (smaller than custom CNN due to larger model)
- lr: 1e-4 to 1e-3 (lower learning rate for pretrained model)
- freeze_backbone: True for initial training, False for fine-tuning
- image_size: 224 (ResNet18's preferred input size)

Replace your existing script with this file (or copy the relevant functions).
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import pickle
import random
import cv2
import time
import torchvision.models as models

# Try import av for decoding if available
try:
    import av  # type: ignore
    _HAS_AV = True
except Exception:
    _HAS_AV = False

# -------------------------- Helpers: find files & derive paths --------------------------

def _find_parquet_files(base_path):
    patterns = [
        os.path.join(base_path, 'data/chunk-*/episode_*.parquet'),
        os.path.join(base_path, 'dataset/data/chunk-*/episode_*.parquet'),
        os.path.join(base_path, 'dataset*/data/chunk-*/episode_*.parquet'),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(list(set(files)))

def _derive_video_path(parquet_path: str) -> str:
    parquet_path = os.path.abspath(parquet_path)
    dir_path, parquet_file = os.path.split(parquet_path)
    chunk_dir = os.path.basename(dir_path)
    root = os.path.dirname(os.path.dirname(dir_path))
    video_path = os.path.join(root, 'videos', chunk_dir, 'observation.images.rgb', f"{os.path.splitext(parquet_file)[0]}.avi")
    return video_path

# -------------------------- Frame cache utilities --------------------------

def build_frame_cache(image_refs, image_size, cache_dir, target_video_width=640, target_video_height=360, fill_strategy: str = 'nearest'):
    os.makedirs(cache_dir, exist_ok=True)
    by_video = {}
    for video_path, fidx in image_refs:
        by_video.setdefault(video_path, set()).add(int(fidx))

    created = {}
    for vp, fidxs in tqdm(list(by_video.items()), desc='Building frame cache'):
        try:
            if not os.path.exists(vp):
                continue
            fname = os.path.splitext(os.path.basename(vp))[0]
            base = os.path.join(cache_dir, f"{fname}")
            frames_npy = f"{base}_frames.npy"
            idxs_npy = f"{base}_idxs.npy"
            if os.path.exists(frames_npy) and os.path.exists(idxs_npy):
                created[vp] = frames_npy
                continue

            frames_list = []
            idx_list = []
            total_frames = None
            try:
                cap_cnt = cv2.VideoCapture(vp)
                if cap_cnt.isOpened():
                    total_frames_val = int(cap_cnt.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_frames = total_frames_val if total_frames_val > 0 else None
                cap_cnt.release()
            except Exception:
                total_frames = None

            need_all = set(int(x) for x in fidxs)
            if total_frames is not None:
                need = {int(x) for x in need_all if 0 <= int(x) < total_frames}
            else:
                need = need_all
            if not need:
                continue

            # Try PyAV first if available
            if _HAS_AV:
                try:
                    container = av.open(vp)
                    stream = container.streams.video[0]
                    for i, frame in enumerate(container.decode(stream)):
                        if i in need:
                            try:
                                img = frame.to_ndarray(format='rgb24')
                            except Exception:
                                continue
                            img = cv2.resize(img, (target_video_width, target_video_height), interpolation=cv2.INTER_LINEAR)
                            frames_list.append(img.astype(np.uint8))
                            idx_list.append(i)
                            if len(idx_list) == len(need):
                                break
                except Exception:
                    pass

            # fallback OpenCV
            if not frames_list:
                cap = cv2.VideoCapture(vp)
                if cap.isOpened():
                    for idx in sorted(list(need)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ok, fr = cap.read()
                        if not ok or fr is None:
                            # try neighbors if requested
                            if fill_strategy == 'nearest' and total_frames is not None:
                                got = False
                                for off in (1, -1, 2, -2):
                                    ni = int(idx + off)
                                    if ni < 0 or (total_frames is not None and ni >= total_frames):
                                        continue
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, ni)
                                    oko, fro = cap.read()
                                    if oko and fro is not None:
                                        fr = fro
                                        ok = True
                                        got = True
                                        break
                                if not got and fill_strategy == 'zero':
                                    z = np.zeros((target_video_height, target_video_width, 3), dtype=np.uint8)
                                    frames_list.append(z); idx_list.append(int(idx))
                                    continue
                            elif fill_strategy == 'zero':
                                z = np.zeros((target_video_height, target_video_width, 3), dtype=np.uint8)
                                frames_list.append(z); idx_list.append(int(idx))
                            continue
                        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        fr = cv2.resize(fr, (target_video_width, target_video_height), interpolation=cv2.INTER_LINEAR)
                        frames_list.append(fr.astype(np.uint8))
                        idx_list.append(int(idx))
                    cap.release()

            if frames_list:
                frames_arr = np.ascontiguousarray(np.stack(frames_list, axis=0))
                idx_arr = np.array(idx_list, dtype=np.int32)
                np.save(frames_npy, frames_arr)
                np.save(idxs_npy, idx_arr)
                created[vp] = frames_npy
        except Exception as e:
            print(f"Warning: failed caching {vp}: {e}")
    return created

# -------------------------- Dataset --------------------------

class AdvancedRobotDataset(Dataset):
    def __init__(self, observations, actions, scaler=None, augment=True, image_refs=None,
                 image_size=128, cache_map=None, img_mean=None, img_std=None,
                 target_video_width=640, target_video_height=360, disable_fallback=False):
        self.observations = torch.from_numpy(np.asarray(observations)).float()
        self.actions = torch.from_numpy(np.asarray(actions)).float()
        self.scaler = scaler
        self.augment = augment
        self.image_refs = image_refs
        self.image_size = image_size
        self.cache_map = cache_map or {}
        self.img_mean = np.array(img_mean) if img_mean is not None else None
        self.img_std = np.array(img_std) if img_std is not None else None
        self.target_video_width = int(target_video_width)
        self.target_video_height = int(target_video_height)
        self.disable_fallback = bool(disable_fallback)

        self._npz_handles = {}
        self._index_maps = {}
        self._refs_by_video = {}
        if self.image_refs is not None:
            for vp, fidx in self.image_refs:
                self._refs_by_video.setdefault(vp, set()).add(int(fidx))

        # if user disabled fallback and cache_map provided, drop missing samples
        if self.image_refs is not None and self.cache_map and self.disable_fallback:
            available_by_video = {}
            for vp, cache_path in self.cache_map.items():
                try:
                    if cache_path.endswith('_frames.npy'):
                        idxs_path = cache_path.replace('_frames.npy', '_idxs.npy')
                        if os.path.exists(idxs_path):
                            frame_idxs = np.load(idxs_path)
                            available_by_video[vp] = set(int(x) for x in frame_idxs.tolist())
                except Exception:
                    pass
            keep_mask = []
            for (vp, fidx), _ in zip(self.image_refs, self.observations):
                if vp in available_by_video and int(fidx) in available_by_video[vp]:
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            keep_mask = np.array(keep_mask, dtype=bool)
            if keep_mask.any() and not keep_mask.all():
                self.observations = self.observations[keep_mask]
                self.actions = self.actions[keep_mask]
                self.image_refs = [self.image_refs[i] for i, k in enumerate(keep_mask.tolist()) if k]
                self._refs_by_video = {}
            for vp, fidx in self.image_refs:
                self._refs_by_video.setdefault(vp, set()).add(int(fidx))

    def _normalize_and_resize(self, img_hwc_uint8: np.ndarray) -> torch.Tensor:
        # ensure target video resolution first
        if (img_hwc_uint8.shape[0] != self.target_video_height) or (img_hwc_uint8.shape[1] != self.target_video_width):
            img_hwc_uint8 = cv2.resize(img_hwc_uint8, (self.target_video_width, self.target_video_height), interpolation=cv2.INTER_LINEAR)
        # resize to model image size
        img = cv2.resize(img_hwc_uint8, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        
        # Use ImageNet normalization for ResNet18
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        # per-channel normalize if custom mean/std provided, otherwise use ImageNet
        if self.img_mean is not None and self.img_std is not None:
            # broadcast: img (H,W,3)
            img = (img - self.img_mean[None, None, :]) / (self.img_std[None, None, :] + 1e-8)
        else:
            # Use ImageNet normalization
            img = (img - imagenet_mean[None, None, :]) / (imagenet_std[None, None, :] + 1e-8)
        
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img).float()

    def _ensure_cache_open(self, video_path: str) -> bool:
        if video_path in self._index_maps:
            return True
        npz_path = self.cache_map.get(video_path)
        if not npz_path or not os.path.exists(npz_path):
            return False
        try:
            if npz_path.endswith('.npz'):
                npz = np.load(npz_path)
                frame_idxs = npz['frame_idxs']
                idx_to_pos = {int(k): int(i) for i, k in enumerate(frame_idxs)}
                self._npz_handles[video_path] = npz
                self._index_maps[video_path] = idx_to_pos
                return True
            elif npz_path.endswith('_frames.npy'):
                frames = np.load(npz_path, mmap_mode='r')
                idxs_path = npz_path.replace('_frames.npy', '_idxs.npy')
                frame_idxs = np.load(idxs_path)
                idx_to_pos = {int(k): int(i) for i, k in enumerate(frame_idxs)}
                self._npz_handles[video_path] = frames
                self._index_maps[video_path] = idx_to_pos
                return True
        except Exception as e:
            print(f"Warning: failed to open cache {npz_path}: {e}")
        return False

    def __len__(self):
        return len(self.observations)

    def _load_image_from_cache(self, video_path: str, frame_idx: int):
        handle = self._npz_handles.get(video_path)
        idxmap = self._index_maps.get(video_path)
        if handle is None or idxmap is None:
            return None
        pos = idxmap.get(int(frame_idx))
        if pos is None:
            return None
        try:
            if isinstance(handle, np.lib.npyio.NpzFile):
                frames_arr = handle['frames']
            else:
                frames_arr = handle
            img_uint8 = frames_arr[pos]  # uint8 HWC (likely target_video res)
        except Exception as e:
            print(f"Warning: failed to read cache for {video_path} @ {pos}: {e}")
            self._npz_handles.pop(video_path, None)
            self._index_maps.pop(video_path, None)
            return None
        return self._normalize_and_resize(img_uint8)

    def _load_image_fallback(self, video_path: str, frame_idx: int):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, fr = cap.read()
        cap.release()
        if not ok or fr is None:
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        return self._normalize_and_resize(fr)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]
        if self.augment and random.random() < 0.3:
            obs = obs + torch.randn_like(obs) * 0.005

        if self.image_refs is not None:
            video_path, frame_idx = self.image_refs[idx]
            img = None
            if self._ensure_cache_open(video_path):
                img = self._load_image_from_cache(video_path, frame_idx)
            if img is None:
                if self.disable_fallback:
                    img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
                else:
                    img = self._load_image_fallback(video_path, frame_idx)
            return obs, img, action
        return obs, action

# -------------------------- Models --------------------------

class AdvancedBCNetwork(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, hidden_dims=[256,256,128], dropout=0.05):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))  # BatchNorm yerine LayerNorm kullan
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ResNet18VisionCNN(nn.Module):
    def __init__(self, img_channels=3, state_dim=6, action_dim=6, dropout: float = 0.05, head_hidden: int = 128, freeze_backbone: bool = False):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer if needed for different input channels
        if img_channels != 3:
            self.backbone.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ResNet18 feature dimension is 512
        fusion_dim = 512 + state_dim
        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(),
            nn.LayerNorm(head_hidden),  # BatchNorm yerine LayerNorm kullan
            nn.Dropout(dropout),
            nn.Linear(head_hidden, action_dim)
        )
        
        # Auxiliary state-only head to stabilize training
        self.state_head = nn.Sequential(
            nn.Linear(state_dim, head_hidden),
            nn.ReLU(),
            nn.LayerNorm(head_hidden),  # BatchNorm yerine LayerNorm kullan
            nn.Dropout(dropout),
            nn.Linear(head_hidden, action_dim)
        )
    
    def forward(self, state, image):
        # Extract features from ResNet18
        feat = self.backbone(image).flatten(1)  # (B, 512)
        
        # Concatenate with state features
        x = torch.cat([feat, state], dim=1)
        
        # Main prediction head
        fused = self.head(x)
        
        # Auxiliary state-only prediction
        state_only = self.state_head(state)
        
        # Return both for weighted training; inference can use fused
        return fused, state_only

# -------------------------- Training routine --------------------------

def _weighted_loss(pred: torch.Tensor, target: torch.Tensor, dim_weights: torch.Tensor, loss_type: str = 'mse', huber_delta: float = 1.0) -> torch.Tensor:
    # pred/target shape: (B, A)
    if loss_type == 'huber':
        diff = pred - target
        abs_diff = torch.abs(diff)
        quad = torch.minimum(abs_diff, torch.tensor(huber_delta, device=pred.device))
        lin = abs_diff - quad
        per_elem = 0.5 * quad * quad + huber_delta * lin
    else:
        per_elem = (pred - target) ** 2
    # mean over batch, weight dims, mean dims
    per_dim = per_elem.mean(dim=0)
    w = dim_weights.to(pred.device)
    return (per_dim * w).mean()

def train_model(model, train_loader, val_loader, num_epochs, device, lr=1e-3, amp=False, act_scaler=None,
                state_loss_w: float = 2.0, vision_loss_w: float = 1.0,
                state_boost_epochs: int = 0, state_boost_factor: float = 1.0,
                dim_loss_w: torch.Tensor = None, loss_type: str = 'mse', huber_delta: float = 1.0,
                lr_sched: str = 'plateau', warmup_epochs: int = 0, grad_accum_steps: int = 1,
                clip_grad: float = 1.0, patience: int = 12, auto_reweight: bool = False,
                reweight_alpha: float = 1.0, reweight_min: float = 0.5, reweight_max: float = 2.0, reweight_momentum: float = 0.9):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if lr_sched == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and device.type == 'cuda'))

    best_val = float('inf')
    history = {'train': [], 'val': []}
    wait = 0
    # per-dim weights tensor
    if dim_loss_w is None:
        # infer action dim after first batch
        dim_weights = None
    else:
        dim_weights = torch.tensor(dim_loss_w, dtype=torch.float32, device=device)
    # moving avg val per-dim mse for auto reweight
    val_mse_ma = None

    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        nb = 0
        # warmup
        if lr_sched == 'cosine' and warmup_epochs > 0 and epoch < warmup_epochs:
            warm_frac = float(epoch + 1) / float(max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = lr * warm_frac
        accum = 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                obs, imgs, acts = batch
                obs, imgs, acts = obs.to(device), imgs.to(device), acts.to(device)
                inputs = (obs, imgs)
            else:
                obs, acts = batch
                obs, acts = obs.to(device), acts.to(device)
                inputs = (obs,)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(amp and device.type == 'cuda')):
                if len(inputs) == 2:
                    preds = model(inputs[0], inputs[1])
                    if isinstance(preds, (tuple, list)) and len(preds) == 2:
                        fused_pred, state_pred = preds
                        # Curriculum: boost state loss early if requested
                        eff_state_w = state_loss_w * (state_boost_factor if epoch < state_boost_epochs else 1.0)
                        # initialize dim weights lazily
                        if dim_weights is None:
                            dim_weights = torch.ones(acts.shape[1], dtype=torch.float32, device=device)
                        loss_fused = _weighted_loss(fused_pred, acts, dim_weights, loss_type, huber_delta)
                        loss_state = _weighted_loss(state_pred, acts, dim_weights, loss_type, huber_delta)
                        loss = vision_loss_w * loss_fused + eff_state_w * loss_state
                    else:
                        if dim_weights is None:
                            dim_weights = torch.ones(acts.shape[1], dtype=torch.float32, device=device)
                        loss = _weighted_loss(preds, acts, dim_weights, loss_type, huber_delta)
                else:
                    preds = model(inputs[0])
                    if dim_weights is None:
                        dim_weights = torch.ones(acts.shape[1], dtype=torch.float32, device=device)
                    loss = _weighted_loss(preds, acts, dim_weights, loss_type, huber_delta)
            # Check if we have enough samples for BatchNorm (if any remain)
            if obs.shape[0] > 1:  # Only process if batch size > 1
                scaler.scale(loss / max(1, grad_accum_steps)).backward()
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                accum += 1
                if accum % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                # Skip this batch if batch size is 1
                print(f"Warning: Skipping batch with size 1 to avoid BatchNorm issues")
                continue

            train_loss += loss.item()
            nb += 1

        train_loss = train_loss / max(1, nb)
        history['train'].append(train_loss)

        # validation + diagnostics
        model.eval()
        val_loss = 0.0
        nb = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    obs, imgs, acts = batch
                    obs, imgs, acts = obs.to(device), imgs.to(device), acts.to(device)
                    inputs = (obs, imgs)
                else:
                    obs, acts = batch
                    obs, acts = obs.to(device), acts.to(device)
                    inputs = (obs,)
                with torch.amp.autocast(device_type='cuda', enabled=(amp and device.type == 'cuda')):
                    if len(inputs) == 2:
                        preds = model(inputs[0], inputs[1])
                        if isinstance(preds, (tuple, list)) and len(preds) == 2:
                            fused_pred, state_pred = preds
                            eff_state_w = state_loss_w * (state_boost_factor if epoch < state_boost_epochs else 1.0)
                            dw = torch.ones(acts.shape[1], dtype=torch.float32, device=device) if dim_weights is None else dim_weights
                            loss_fused = _weighted_loss(fused_pred, acts, dw, loss_type, huber_delta)
                            loss_state = _weighted_loss(state_pred, acts, dw, loss_type, huber_delta)
                            loss = vision_loss_w * loss_fused + eff_state_w * loss_state
                            main_out = fused_pred
                        else:
                            dw = torch.ones(acts.shape[1], dtype=torch.float32, device=device) if dim_weights is None else dim_weights
                            loss = _weighted_loss(preds, acts, dw, loss_type, huber_delta)
                            main_out = preds
                    else:
                        preds = model(inputs[0])
                        dw = torch.ones(acts.shape[1], dtype=torch.float32, device=device) if dim_weights is None else dim_weights
                        loss = _weighted_loss(preds, acts, dw, loss_type, huber_delta)
                        main_out = preds
                val_loss += loss.item()
                nb += 1
                all_preds.append(main_out.detach().cpu().numpy())
                all_targets.append(acts.cpu().numpy())
        val_loss = val_loss / max(1, nb)
        history['val'].append(val_loss)
        if lr_sched == 'cosine':
            if epoch >= warmup_epochs:
                scheduler.step()
        else:
            scheduler.step(val_loss)

        # diagnostics (per-dim scaled + orig if scaler)
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        mse_vec = np.mean((preds_arr - targets_arr)**2, axis=0)
        mse_per_dim = mse_vec.tolist()
        std_pred = preds_arr.std(axis=0).tolist()
        print(f"Epoch {epoch+1}/{num_epochs} — train: {train_loss:.6f} val: {val_loss:.6f} lr: {optimizer.param_groups[0]['lr']:.2e} time: {time.time()-t0:.1f}s")
        print("  diag (scaled) mse per-dim:", mse_per_dim)
        print("  diag (scaled) pred std per-dim:", std_pred)

        # Auto reweight per-dim loss to lift underperforming dims
        if auto_reweight:
            cur = torch.from_numpy(mse_vec).float().to(device)
            if val_mse_ma is None:
                val_mse_ma = cur
            else:
                val_mse_ma = reweight_momentum * val_mse_ma + (1 - reweight_momentum) * cur
            mean_mse = torch.clamp(val_mse_ma.mean(), min=1e-8)
            raw_w = (val_mse_ma / mean_mse) ** reweight_alpha
            new_w = torch.clamp(raw_w, min=reweight_min, max=reweight_max)
            if dim_weights is None:
                dim_weights = new_w
            else:
                dim_weights = 0.5 * dim_weights + 0.5 * new_w  # smooth update
            print("  auto-reweight dim_weights:", dim_weights.detach().cpu().numpy().round(3).tolist())

        if act_scaler is not None:
            try:
                preds_orig = act_scaler.inverse_transform(preds_arr)
                targets_orig = act_scaler.inverse_transform(targets_arr)
                mse_orig = np.mean((preds_orig - targets_orig)**2, axis=0).tolist()
                print("  diag (orig) mse per-dim:", mse_orig)
                print("  diag (orig) pred mean/std per-dim:", preds_orig.mean(axis=0).tolist(), preds_orig.std(axis=0).tolist())
                print("  diag (orig) target mean/std per-dim:", targets_orig.mean(axis=0).tolist(), targets_orig.std(axis=0).tolist())
            except Exception:
                pass

        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, 'outputs/bc_best.pt')
            print(f"Saved best model (val {best_val:.6f})")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    torch.save({'model_state_dict': model.state_dict(), 'history': history}, 'outputs/bc_final.pt')
    return history

# -------------------------- Data loader & main --------------------------

def load_dataset_advanced(data_paths, use_vision=False):
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    all_files = []
    for p in data_paths:
        all_files.extend(_find_parquet_files(p))
    all_files = sorted(list(set(all_files)))
    if not all_files:
        raise FileNotFoundError('No parquet episode files found under provided paths')

    all_obs = []
    all_act = []
    all_refs = []
    for f in tqdm(all_files, desc='Loading parquet'):
        df = pd.read_parquet(f)
        obs = np.array(df['observation.state'].tolist())
        act = np.array(df['action'].tolist())
        valid = ~(np.isnan(obs).any(axis=1) | np.isnan(act).any(axis=1))
        obs = obs[valid]
        act = act[valid]
        all_obs.append(obs)
        all_act.append(act)
        if use_vision:
            vid = _derive_video_path(f)
            fidxs = np.array(df['frame_index'].tolist())[valid]
            refs = [(vid, int(fi)) for fi in fidxs]
            all_refs.extend(refs)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_act, axis=0)

    if use_vision:
        return observations, actions, all_refs
    return observations, actions, None

def compute_image_stats(cache_map, n_samples=2000):
    means, sqmeans = [], []
    sampled = 0
    for vp, path in cache_map.items():
        if sampled >= n_samples: break
        if path.endswith('_frames.npy'):
            frames = np.load(path, mmap_mode='r')
            L = frames.shape[0]
            idxs = np.random.choice(L, min(L, n_samples - sampled), replace=False)
            for i in idxs:
                img = frames[int(i)].astype(np.float32) / 255.0
                means.append(img.mean(axis=(0,1)))
                sqmeans.append((img**2).mean(axis=(0,1)))
            sampled += len(idxs)
    if len(means) == 0:
        return None, None
    means = np.array(means)
    mean = means.mean(axis=0)
    sqmeans = np.array(sqmeans)
    var = sqmeans.mean(axis=0) - mean**2
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.tolist(), std.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='single dataset root')
    parser.add_argument('--data_dirs', type=str, default='', help='comma separated dataset roots')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--use_vision', action='store_true')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--preload_cache', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='outputs/frame_cache')
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--amp', action='store_true', help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--target_video_width', type=int, default=640)
    parser.add_argument('--target_video_height', type=int, default=360)
    parser.add_argument('--cache_fill_strategy', type=str, default='nearest', choices=['nearest','strict','zero'])
    parser.add_argument('--mlp_hidden', type=str, default='256,256,128')
    parser.add_argument('--cnn_width_mul', type=float, default=1.0, help='deprecated: kept for compatibility')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--head_hidden', type=int, default=128, help='hidden width for fused/state heads')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze ResNet18 backbone weights')
    # Weighted dual-head training (vision fused + state-only auxiliary)
    parser.add_argument('--state_loss_w', type=float, default=2.0, help='weight for auxiliary state-only head loss')
    parser.add_argument('--vision_loss_w', type=float, default=1.0, help='weight for fused vision+state head loss')
    parser.add_argument('--state_boost_epochs', type=int, default=0, help='epochs to boost state loss early (0 disables)')
    parser.add_argument('--state_boost_factor', type=float, default=1.0, help='multiplier for state loss during early boost epochs')
    # Per-dim loss controls
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse','huber'])
    parser.add_argument('--huber_delta', type=float, default=1.0)
    parser.add_argument('--dim_loss_w', type=str, default='', help='comma-separated per-dim weights (length = action_dim); empty = all ones')
    parser.add_argument('--auto_reweight', action='store_true', help='dynamically upweight underperforming dims based on val MSE')
    parser.add_argument('--reweight_alpha', type=float, default=1.0)
    parser.add_argument('--reweight_min', type=float, default=0.5)
    parser.add_argument('--reweight_max', type=float, default=2.0)
    parser.add_argument('--reweight_momentum', type=float, default=0.9)
    # Longer/better training utilities
    parser.add_argument('--lr_sched', type=str, default='plateau', choices=['plateau','cosine'])
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--no_video_fallback', action='store_true')
    parser.add_argument('--cache_decode_backend', type=str, default='auto', choices=['auto','cv2','av'])
    parser.add_argument('--use_per_channel_norm', action='store_true', default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs('outputs', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.data_dirs and args.data_dirs.strip():
        data_paths = [p.strip() for p in args.data_dirs.split(',') if p.strip()]
    else:
        data_paths = [args.data_dir]
    print(f"Dataset roots: {data_paths}")

    observations, actions, image_refs = load_dataset_advanced(data_paths, use_vision=args.use_vision)

    if args.use_vision and args.frame_stride > 1:
        idxs = np.arange(len(observations))[::args.frame_stride]
        observations = observations[idxs]
        actions = actions[idxs]
        image_refs = [image_refs[i] for i in idxs]

    if args.use_vision:
        X_train, X_test, y_train, y_test, refs_train, refs_test = train_test_split(
            observations, actions, image_refs, test_size=args.test_size, random_state=args.seed
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            observations, actions, test_size=args.test_size, random_state=args.seed
        )
        refs_train = refs_test = None

    obs_scaler = RobustScaler()
    X_train = obs_scaler.fit_transform(X_train)
    X_test = obs_scaler.transform(X_test)
    act_scaler = RobustScaler()
    y_train = act_scaler.fit_transform(y_train)
    y_test = act_scaler.transform(y_test)

    cache_map = {}
    if args.use_vision and args.preload_cache:
        print('Building frame cache (this may take a while the first time)')
        cache_map = build_frame_cache(refs_train + refs_test, args.image_size, args.cache_dir,
                                      target_video_width=args.target_video_width, target_video_height=args.target_video_height,
                                      fill_strategy=args.cache_fill_strategy)

    # compute per-channel stats if possible
    img_mean = None; img_std = None
    if args.use_vision and cache_map and args.use_per_channel_norm:
        mean, std = compute_image_stats(cache_map, n_samples=2000)
        if mean is not None:
            img_mean = np.array(mean, dtype=np.float32)
            img_std = np.array(std, dtype=np.float32)
            np.save('outputs/img_mean.npy', img_mean)
            np.save('outputs/img_std.npy', img_std)
            print("Saved img mean/std to outputs/img_mean.npy, outputs/img_std.npy")
        else:
            print("Warning: could not compute image stats from cache (no frames), falling back to center=0.5")

    train_dataset = AdvancedRobotDataset(X_train, y_train, scaler=obs_scaler, augment=not args.no_augment,
                                        image_refs=refs_train, image_size=args.image_size, cache_map=cache_map,
                                        img_mean=img_mean, img_std=img_std,
                                        target_video_width=args.target_video_width, target_video_height=args.target_video_height,
                                        disable_fallback=args.no_video_fallback)
    test_dataset = AdvancedRobotDataset(X_test, y_test, scaler=obs_scaler, augment=False,
                                       image_refs=refs_test, image_size=args.image_size, cache_map=cache_map,
                                       img_mean=img_mean, img_std=img_std,
                                       target_video_width=args.target_video_width, target_video_height=args.target_video_height,
                                       disable_fallback=args.no_video_fallback)

    if args.num_workers is None:
        if args.use_vision:
            num_workers = min(8, max(0, os.cpu_count() - 1 or 1))
        else:
            num_workers = max(2, min(8, os.cpu_count() // 2))
    else:
        num_workers = args.num_workers

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=(num_workers>0), pin_memory=(device.type=='cuda'),
                              prefetch_factor=2 if num_workers>0 else None, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=(num_workers>0), pin_memory=(device.type=='cuda'),
                             prefetch_factor=2 if num_workers>0 else None, drop_last=True)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    if args.use_vision:
        model = ResNet18VisionCNN(img_channels=3, state_dim=X_train.shape[1], action_dim=y_train.shape[1], dropout=args.dropout, head_hidden=args.head_hidden, freeze_backbone=args.freeze_backbone).to(device)
        print('Using ResNet18VisionCNN')
    else:
        if args.ensemble:
            hidden = [int(x) for x in args.mlp_hidden.split(',') if x.strip()]
            models = nn.ModuleList([AdvancedBCNetwork(input_dim=X_train.shape[1], output_dim=y_train.shape[1], hidden_dims=hidden, dropout=args.dropout) for _ in range(3)])
            class Ensemble(nn.Module):
                def __init__(self, models):
                    super().__init__()
                    self.models = models
                def forward(self, x):
                    outs = [m(x) for m in self.models]
                    return torch.mean(torch.stack(outs), dim=0)
            model = Ensemble(models).to(device)
            print('Using small ensemble of MLPs')
        else:
            hidden = [int(x) for x in args.mlp_hidden.split(',') if x.strip()]
            model = AdvancedBCNetwork(input_dim=X_train.shape[1], output_dim=y_train.shape[1], hidden_dims=hidden, dropout=args.dropout).to(device)
            print('Using Advanced MLP')

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # parse per-dim weights
    dim_w = None
    if args.dim_loss_w.strip():
        try:
            dim_w = [float(x) for x in args.dim_loss_w.split(',') if x.strip()]
        except Exception:
            dim_w = None

    history = train_model(
        model, train_loader, test_loader, args.num_epochs, device,
        lr=args.lr, amp=args.amp, act_scaler=act_scaler,
        state_loss_w=args.state_loss_w, vision_loss_w=args.vision_loss_w,
        state_boost_epochs=args.state_boost_epochs, state_boost_factor=args.state_boost_factor,
        dim_loss_w=dim_w, loss_type=args.loss_type, huber_delta=args.huber_delta,
        lr_sched=args.lr_sched, warmup_epochs=args.warmup_epochs,
        grad_accum_steps=max(1, args.grad_accum_steps), clip_grad=args.clip_grad,
        patience=args.patience, auto_reweight=args.auto_reweight,
        reweight_alpha=args.reweight_alpha, reweight_min=args.reweight_min,
        reweight_max=args.reweight_max, reweight_momentum=args.reweight_momentum,
    )

    with open('outputs/obs_scaler_optimized.pkl', 'wb') as f:
        pickle.dump(obs_scaler, f)
    with open('outputs/act_scaler_optimized.pkl', 'wb') as f:
        pickle.dump(act_scaler, f)

    try:
        with open('outputs/bc_training_meta.txt', 'w') as mf:
            arch = 'vision' if args.use_vision else 'mlp'
            mf.write(f'arch={arch}\n')
            mf.write(f'image_size={args.image_size}\n')
            mf.write(f'use_vision={args.use_vision}\n')
            if img_mean is not None:
                mf.write(f'img_mean={",".join([str(x) for x in img_mean.tolist()])}\n')
                mf.write(f'img_std={",".join([str(x) for x in img_std.tolist()])}\n')
            mf.write(f'cnn_width_mul={args.cnn_width_mul}\n')
            mf.write(f'freeze_backbone={args.freeze_backbone}\n')
            mf.write(f'head_hidden={args.head_hidden}\n')
            mf.write(f'mlp_hidden={args.mlp_hidden}\n')
            mf.write(f'dropout={args.dropout}\n')
            mf.write(f'state_loss_w={args.state_loss_w}\n')
            mf.write(f'vision_loss_w={args.vision_loss_w}\n')
            mf.write(f'state_boost_epochs={args.state_boost_epochs}\n')
            mf.write(f'state_boost_factor={args.state_boost_factor}\n')
            mf.write(f'loss_type={args.loss_type}\n')
            mf.write(f'huber_delta={args.huber_delta}\n')
            mf.write(f'dim_loss_w={args.dim_loss_w}\n')
            mf.write(f'lr_sched={args.lr_sched}\n')
            mf.write(f'warmup_epochs={args.warmup_epochs}\n')
            mf.write(f'grad_accum_steps={args.grad_accum_steps}\n')
            mf.write(f'clip_grad={args.clip_grad}\n')
            mf.write(f'patience={args.patience}\n')
            mf.write(f'auto_reweight={args.auto_reweight}\n')
            mf.write(f'reweight_alpha={args.reweight_alpha}\n')
            mf.write(f'reweight_min={args.reweight_min}\n')
            mf.write(f'reweight_max={args.reweight_max}\n')
            mf.write(f'reweight_momentum={args.reweight_momentum}\n')
    except Exception:
        pass

    print('Done')

if __name__ == '__main__':
    main()
