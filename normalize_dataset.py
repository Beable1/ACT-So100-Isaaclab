#!/usr/bin/env python3
"""
Dataset Normalization Script for LeRobot/Diffusion Policy

This script normalizes action and state values to [-1, 1] range
which is optimal for neural network training.

Usage:
    python normalize_dataset.py --input Jia-63ep --output Jia-63ep-normalized
"""

import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def compute_stats(data_dir: str):
    """Compute min/max statistics for all parquet files."""
    parquet_files = list(Path(data_dir).rglob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Collect all values
    all_actions = []
    all_states = []
    
    for pf in tqdm(parquet_files, desc="Reading files"):
        df = pd.read_parquet(pf)
        
        if 'action' in df.columns:
            actions = np.stack(df['action'].values)
            all_actions.append(actions)
        
        if 'observation.state' in df.columns:
            states = np.stack(df['observation.state'].values)
            all_states.append(states)
    
    # Concatenate
    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    
    # Compute stats
    stats = {
        'action': {
            'min': all_actions.min(axis=0).tolist(),
            'max': all_actions.max(axis=0).tolist(),
            'mean': all_actions.mean(axis=0).tolist(),
            'std': all_actions.std(axis=0).tolist(),
        },
        'observation.state': {
            'min': all_states.min(axis=0).tolist(),
            'max': all_states.max(axis=0).tolist(),
            'mean': all_states.mean(axis=0).tolist(),
            'std': all_states.std(axis=0).tolist(),
        }
    }
    
    print("\n📊 Original Statistics:")
    print("Action:")
    for i in range(len(stats['action']['min'])):
        print(f"  Joint {i}: min={stats['action']['min'][i]:.4f}, max={stats['action']['max'][i]:.4f}")
    print("State:")
    for i in range(len(stats['observation.state']['min'])):
        print(f"  Joint {i}: min={stats['observation.state']['min'][i]:.4f}, max={stats['observation.state']['max'][i]:.4f}")
    
    return stats


def normalize_value(value, min_val, max_val, target_min=-1.0, target_max=1.0):
    """Normalize value from [min_val, max_val] to [target_min, target_max]."""
    if max_val - min_val < 1e-8:
        return np.zeros_like(value)
    
    # Normalize to [0, 1]
    normalized = (value - min_val) / (max_val - min_val)
    # Scale to [target_min, target_max]
    scaled = normalized * (target_max - target_min) + target_min
    return scaled


def normalize_dataset(input_dir: str, output_dir: str, target_range=(-1, 1)):
    """Normalize dataset to target range."""
    
    # Compute stats
    stats = compute_stats(os.path.join(input_dir, 'data'))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy meta folder
    meta_src = os.path.join(input_dir, 'meta')
    meta_dst = os.path.join(output_dir, 'meta')
    if os.path.exists(meta_src):
        if os.path.exists(meta_dst):
            shutil.rmtree(meta_dst)
        shutil.copytree(meta_src, meta_dst)
        print(f"✅ Copied meta folder")
    
    # Copy videos folder (if exists)
    videos_src = os.path.join(input_dir, 'videos')
    videos_dst = os.path.join(output_dir, 'videos')
    if os.path.exists(videos_src):
        if not os.path.exists(videos_dst):
            # Create symlink instead of copying (saves space)
            os.symlink(os.path.abspath(videos_src), videos_dst)
            print(f"✅ Created symlink for videos folder")
    
    # Process parquet files
    data_src = os.path.join(input_dir, 'data')
    data_dst = os.path.join(output_dir, 'data')
    os.makedirs(data_dst, exist_ok=True)
    
    parquet_files = list(Path(data_src).rglob("*.parquet"))
    
    target_min, target_max = target_range
    
    for pf in tqdm(parquet_files, desc="Normalizing files"):
        df = pd.read_parquet(pf)
        
        # Normalize actions
        if 'action' in df.columns:
            actions = np.stack(df['action'].values)
            actions_norm = np.zeros_like(actions)
            
            for i in range(actions.shape[1]):
                actions_norm[:, i] = normalize_value(
                    actions[:, i],
                    stats['action']['min'][i],
                    stats['action']['max'][i],
                    target_min, target_max
                )
            
            df['action'] = list(actions_norm)
        
        # Normalize states
        if 'observation.state' in df.columns:
            states = np.stack(df['observation.state'].values)
            states_norm = np.zeros_like(states)
            
            for i in range(states.shape[1]):
                states_norm[:, i] = normalize_value(
                    states[:, i],
                    stats['observation.state']['min'][i],
                    stats['observation.state']['max'][i],
                    target_min, target_max
                )
            
            df['observation.state'] = list(states_norm)
        
        # Save to output
        rel_path = pf.relative_to(data_src)
        out_path = Path(data_dst) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
    
    # Save normalization stats for later use
    stats['target_range'] = [target_min, target_max]
    stats_path = os.path.join(output_dir, 'normalization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved normalization stats to {stats_path}")
    
    # Verify normalization
    print("\n📊 Verifying Normalized Data:")
    sample_pf = list(Path(data_dst).rglob("*.parquet"))[0]
    df_check = pd.read_parquet(sample_pf)
    
    if 'action' in df_check.columns:
        actions = np.stack(df_check['action'].values)
        print(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    
    if 'observation.state' in df_check.columns:
        states = np.stack(df_check['observation.state'].values)
        print(f"State range: [{states.min():.4f}, {states.max():.4f}]")
    
    print(f"\n✅ Normalized dataset saved to: {output_dir}")
    print(f"📝 Remember to use --action_map none when training with normalized data!")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize dataset for Diffusion Policy training")
    parser.add_argument("--input", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output dataset directory")
    parser.add_argument("--range", type=str, default="-1,1", help="Target range (e.g., '-1,1' or '0,1')")
    
    args = parser.parse_args()
    
    # Parse range
    target_range = tuple(map(float, args.range.split(',')))
    print(f"Normalizing to range: {target_range}")
    
    normalize_dataset(args.input, args.output, target_range)


if __name__ == "__main__":
    main()





