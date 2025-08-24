#!/usr/bin/env python3
"""
clean_lerobot_dataset.py

- LeRobot dataset (meta/ + data/chunk-*/episode_*.parquet) için:
  1) parquet dosyalarından next.reward / next.success / reward / success sütunlarını siler
  2) meta/info.json içindeki features tanımlarından ilgili anahtarları kaldırır
  3) isteğe bağlı olarak meta/episodes_stats.jsonl yeniden oluşturur (--regen_stats)
  4) Orijinali değiştirmez: tüm çıktı 'out_dir' içine yazılır

Usage:
  python clean_lerobot_dataset.py \
    --root /path/to/lerobot_root \
    --out /path/to/out_root \
    --workers 6 \
    --regen_stats

Requirements:
  pip install pandas pyarrow tqdm
"""

import os
import json
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np

DROP_COLS = ['next.reward', 'next.success', 'reward', 'success']

def find_parquets(root):
    pattern = os.path.join(root, "data", "chunk-*", "episode_*.parquet")
    return sorted(glob.glob(pattern, recursive=True))

def rel_path_under(root, path):
    return os.path.relpath(path, root)

def process_parquet(path, root, out_root):
    """Read parquet, drop columns if present, write to out_root preserving rel path."""
    rel = rel_path_under(root, path)
    out_path = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        df = pd.read_parquet(path, engine='pyarrow')
    except Exception as e:
        return (path, False, f"read_error:{e}")

    cols_before = set(df.columns)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    try:
        df.to_parquet(out_path, engine='pyarrow', index=False)
    except Exception as e:
        return (path, False, f"write_error:{e}")

    # basic per-file summary for later stats generation
    summary = {
        "path": rel,
        "n_rows": len(df),
    }
    # try to collect common columns if present
    for k in ("episode_index", "frame_index", "task_index"):
        if k in df.columns:
            arr = df[k].to_numpy()
            summary[k] = {
                "min": int(np.nanmin(arr).item()),
                "max": int(np.nanmax(arr).item()),
                "mean": float(np.nanmean(arr).item())
            }
    return (path, True, summary)

def update_info_json(orig_info_path, out_info_path, removed_feature_keys, recompute_counts=None):
    """
    Load info.json, remove feature keys (if exist), optionally update counts (total_episodes/frames).
    """
    with open(orig_info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    features = info.get("features", {})
    for k in removed_feature_keys:
        if k in features:
            del features[k]
    info["features"] = features

    if recompute_counts is not None:
        # recompute_counts should be dict e.g. {"total_episodes": N, "total_frames": M, "total_chunks": C}
        for kk, vv in recompute_counts.items():
            info[kk] = vv

    os.makedirs(os.path.dirname(out_info_path) or '.', exist_ok=True)
    with open(out_info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    return info

def regen_episode_stats(summaries, out_stats_path):
    """
    summaries: list of per-file summary dicts returned from process_parquet when successful.
    Write meta/episodes_stats.jsonl in LeRobot style (one JSON line per episode).
    """
    os.makedirs(os.path.dirname(out_stats_path) or '.', exist_ok=True)
    lines = []
    for s in summaries:
        rel = s["path"]
        try:
            # try to extract episode_index from filename: episode_000012.parquet
            base = os.path.basename(rel)
            idx = int(base.split("_")[1].split(".")[0])
        except Exception:
            idx = None
        stats = {}
        for k in ("episode_index", "frame_index", "task_index"):
            if k in s:
                stats[k] = {
                    "min": [s[k]["min"]],
                    "max": [s[k]["max"]],
                    "mean": [s[k]["mean"]]
                }
        # fallback: if frame_index not present, use n_rows to set min=0, max=n_rows-1, mean=(n_rows-1)/2
        if "frame_index" not in s:
            n = s["n_rows"]
            stats["frame_index"] = {"min":[0], "max":[max(0, n-1)], "mean":[(n-1)/2.0 if n>0 else 0.0]}
        entry = {"episode_index": idx if idx is not None else s["path"], "stats": stats}
        lines.append(entry)

    # write jsonl
    with open(out_stats_path, 'w', encoding='utf-8') as f:
        for e in lines:
            f.write(json.dumps(e) + "\n")
    return len(lines)

def regen_episodes_jsonl(summaries, out_episodes_path):
    """
    Create meta/episodes.jsonl listing episode_index, length (frames), tasks placeholder if missing.
    """
    os.makedirs(os.path.dirname(out_episodes_path) or '.', exist_ok=True)
    lines = []
    for s in summaries:
        rel = s["path"]
        try:
            base = os.path.basename(rel)
            idx = int(base.split("_")[1].split(".")[0])
        except Exception:
            idx = None
        length = s["n_rows"]
        entry = {"episode_index": idx if idx is not None else s["path"], "tasks": ["default_task"], "length": length}
        lines.append(entry)

    with open(out_episodes_path, 'w', encoding='utf-8') as f:
        for e in lines:
            f.write(json.dumps(e) + "\n")
    return len(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="root of LeRobot dataset (contains meta/ and data/)")
    p.add_argument("--out", required=True, help="output root (will be created); original is NOT modified")
    p.add_argument("--workers", type=int, default=4, help="parallel workers for parquet processing")
    p.add_argument("--regen_stats", action="store_true", help="re-generate meta/episodes_stats.jsonl and meta/episodes.jsonl")
    p.add_argument("--info", default=None, help="path to original info.json (default: <root>/meta/info.json)")
    args = p.parse_args()

    root = args.root
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    parquet_files = find_parquets(root)
    if not parquet_files:
        print("No parquet files found under data/chunk-*/. Check root path.")
        return

    print(f"Found {len(parquet_files)} parquet files. Processing with {args.workers} workers...")

    summaries = []
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_parquet, pth, root, out_root): pth for pth in parquet_files}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            pth = futures[fut]
            try:
                path, ok, info = fut.result()
                if ok:
                    summaries.append(info)
                else:
                    failed.append((path, info))
            except Exception as e:
                failed.append((pth, f"worker_exc:{e}"))

    print(f"Processed: success={len(summaries)}, failed={len(failed)}")
    if failed:
        for f in failed[:10]:
            print("FAILED:", f)

    # update meta/info.json
    orig_info = args.info or os.path.join(root, "meta", "info.json")
    out_info = os.path.join(out_root, "meta", "info.json")
    removed_feature_keys = ["next.reward", "next.success"]
    recompute = None
    # recompute some counts to keep info accurate
    total_episodes = len(summaries)
    total_frames = sum(s["n_rows"] for s in summaries)
    # count chunks from existing data or recompute from parquet paths
    chunk_dirs = set([os.path.dirname(rel_path_under(root, p)).split(os.sep)[1] for p in parquet_files])
    recompute = {"total_episodes": total_episodes, "total_frames": total_frames, "total_chunks": len(chunk_dirs)}
    if os.path.exists(orig_info):
        info = update_info_json(orig_info, out_info, removed_feature_keys, recompute_counts=recompute)
        print("info.json updated/written to:", out_info)
    else:
        print("No original info.json found at", orig_info, "; writing minimal info.json to out.")
        minimal = {
            "codebase_version": "unknown",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_chunks": len(chunk_dirs),
            "features": {}
        }
        os.makedirs(os.path.dirname(out_info) or '.', exist_ok=True)
        with open(out_info, 'w', encoding='utf-8') as f:
            json.dump(minimal, f, indent=2)

    if args.regen_stats:
        out_stats = os.path.join(out_root, "meta", "episodes_stats.jsonl")
        n = regen_episode_stats(summaries, out_stats)
        print("Re-generated episodes_stats.jsonl with", n, "entries at", out_stats)
        out_eps = os.path.join(out_root, "meta", "episodes.jsonl")
        m = regen_episodes_jsonl(summaries, out_eps)
        print("Re-generated episodes.jsonl with", m, "entries at", out_eps)
    else:
        # copy original meta files if present, but remove the info.json already written
        orig_meta_dir = os.path.join(root, "meta")
        if os.path.isdir(orig_meta_dir):
            for f in os.listdir(orig_meta_dir):
                src = os.path.join(orig_meta_dir, f)
                dst = os.path.join(out_root, "meta", f)
                if os.path.exists(dst):
                    continue
                try:
                    # skip episodes_stats.jsonl if you want to avoid copying stale stats
                    if f in ("info.json",):
                        continue
                    import shutil
                    shutil.copy2(src, dst)
                except Exception:
                    pass
        print("Meta files copied (except info.json which was regenerated).")

    print("Done. New dataset root:", out_root)
    print("NOTE: videos were not copied. If your loader expects videos at same relative path, either:")
    print("  - copy meta/videos from original root to out_root, or")
    print("  - update your loader to use original video's root (videos dir).")

if __name__ == "__main__":
    main()
