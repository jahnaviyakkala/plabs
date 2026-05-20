"""
data_loader.py
--------------
Loads and merges the 5 AUtool-Dataset CSVs per benchmark into a unified
feature matrix ready for the PLABS straggler detection pipeline.

CSVs used:
  map.csv        – per-map-task metrics (progress, executionTimeSec, …)
  workers.csv    – per-worker system metrics (cpu, mem, net)
  job.csv        – job-level progress / allocation
  jobDetails.csv – mapsCompleted, mapProgress, etc.
  cluster.csv    – cluster-level resource utilisation
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def _unzip_map(zip_path: str, dest_dir: str) -> str:
    """Extract map.csv from a zip if the plain CSV does not exist."""
    csv_path = os.path.join(dest_dir, "map.csv")
    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    return csv_path


def _find_file(directory: str, filename: str) -> str:
    """Find a file in a directory regardless of case."""
    for f in os.listdir(directory):
        if f.lower() == filename.lower():
            return os.path.join(directory, f)
    return os.path.join(directory, filename)

def load_map(bench_dir: str) -> pd.DataFrame:
    """Return the map-task DataFrame for a benchmark directory."""
    csv = _find_file(bench_dir, "map.csv")
    if not os.path.exists(csv):
        zip_f = _find_file(bench_dir, "map.zip")
        if os.path.exists(zip_f):
            csv = _unzip_map(zip_f, bench_dir)
        else:
            # Try plain Map.csv if map.csv failed
            csv = os.path.join(bench_dir, "Map.csv")
            
    df = _read_csv(csv, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def load_workers(bench_dir: str) -> pd.DataFrame:
    csv = _find_file(bench_dir, "workers.csv")
    df = _read_csv(csv, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def load_job(bench_dir: str) -> pd.DataFrame:
    csv = _find_file(bench_dir, "job.csv")
    df = _read_csv(csv, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def load_job_details(bench_dir: str) -> pd.DataFrame:
    # Try jobDetails.csv and Jobdetails.csv
    csv = _find_file(bench_dir, "jobDetails.csv")
    df = _read_csv(csv, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def load_cluster(bench_dir: str) -> pd.DataFrame:
    csv = _find_file(bench_dir, "cluster.csv")
    df = _read_csv(csv, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


# ── time-aligned merge helpers ────────────────────────────────────────────────

def _nearest_merge(left: pd.DataFrame, right: pd.DataFrame,
                   on_left: str, on_right: str,
                   by: str = "jobId") -> pd.DataFrame:
    """
    Merge right into left by nearest timestamp (within the same jobId).
    Both DataFrames must have a numeric 'timestamp' column.
    """
    left = left.dropna(subset=[on_left])
    right = right.dropna(subset=[on_right])
    result_parts = []
    for jid, grp_l in left.groupby(by):
        grp_r = right[right[by] == jid].copy()
        if grp_r.empty:
            result_parts.append(grp_l)
            continue
        grp_l = grp_l.sort_values(on_left)
        grp_r = grp_r.sort_values(on_right)
        merged = pd.merge_asof(
            grp_l, grp_r,
            left_on=on_left, right_on=on_right,
            by=by,
            direction="nearest",
            suffixes=("", "_r")
        )
        result_parts.append(merged)
    return pd.concat(result_parts, ignore_index=True)


# ── per-worker aggregation ────────────────────────────────────────────────────

def _agg_workers_by_host(workers: pd.DataFrame) -> pd.DataFrame:
    """
    workers.csv schema varies across benchmarks.
    Handles both metricsType (traditional) and node_name (Histogram) schemas.
    """
    w = workers.copy()
    
    # 1. Map host name
    if "metricsType" in w.columns:
        w["host_name"] = (w["metricsType"]
                          .str.replace(r'metricsType=["\']?', "", regex=True)
                          .str.strip('"').str.strip("'").str.strip())
    elif "node_name" in w.columns:
        w["host_name"] = w["node_name"].str.strip()
    else:
        w["host_name"] = "unknown"

    # 2. Map metric columns
    mapping = {
        "cpu_percent": ["cpu_percent", "cpu_usage"],
        "cpu_count":   ["cpu_count", "cpu_cores"],
        "mem_percent": ["mem_percent", "memory_usage"],
        "mem_total":   ["mem_total", "total_memory"],
        "net_upload":  ["net_upload"],
        "net_download":["net_download"]
    }
    
    for target, candidates in mapping.items():
        if target not in w.columns:
            for cand in candidates:
                if cand in w.columns:
                    w[target] = w[cand]
                    break
        if target not in w.columns:
            w[target] = 0  # Fallback
            
    # keep numeric columns
    num_cols = list(mapping.keys())
    existing_num_cols = [c for c in num_cols if c in w.columns]
    for c in existing_num_cols:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    
    if "timestamp" in w.columns:
        w["timestamp"] = pd.to_numeric(w["timestamp"], errors="coerce")
    else:
        w["timestamp"] = 0
    
    # 3. Final Selection (Robust)
    final_cols = ["host_name", "timestamp", "jobId"] + existing_num_cols
    available_final = [c for c in final_cols if c in w.columns]
    return w[available_final]


# ── main loader ───────────────────────────────────────────────────────────────

def build_feature_matrix(bench_dir: str, benchmark_name: str) -> pd.DataFrame:
    """
    Load all 5 CSVs for *bench_dir*, merge them, and return a feature matrix
    with one row per map-task observation.

    Parameters
    ----------
    bench_dir       : path to e.g. AUtool-Dataset/Heterogeneity/Grep
    benchmark_name  : label string, e.g. 'Grep'
    """
    # ── 1. raw loads ──────────────────────────────────────────────────────────
    map_df = load_map(bench_dir)
    workers_df = load_workers(bench_dir)
    job_df = load_job(bench_dir)
    jd_df = load_job_details(bench_dir)
    cl_df = load_cluster(bench_dir)

    # ── 2. numeric coercions ─────────────────────────────────────────────────
    for col in ["progress", "executionTimeSec", "elapsedTimeSec",
                "startTime", "finishTime", "timestamp"]:
        if col in map_df.columns:
            map_df[col] = pd.to_numeric(map_df[col], errors="coerce")

    map_df = map_df.dropna(subset=["jobId"]).copy()
    if "timestamp" in map_df.columns:
        map_df["timestamp"] = pd.to_numeric(map_df["timestamp"], errors="coerce")
    else:
        map_df["timestamp"] = 0

    # ── 3. extract hostname from map host column (format: worker1:8042) ───────
    if "host" in map_df.columns:
        map_df["host_name"] = map_df["host"].str.split(":").str[0]
    else:
        map_df["host_name"] = "unknown"

    # ── 4. worker system metrics (vectorized) ────────────────────────────────
    w_agg = _agg_workers_by_host(workers_df)
    w_num = ["cpu_percent", "cpu_count", "mem_percent",
             "mem_total", "net_upload", "net_download"]
    
    # Ensure all w_num columns exist for grouping
    for c in w_num:
        if c not in w_agg.columns:
            w_agg[c] = 0.0

    # Aggregate worker metrics per (jobId, timestamp) – mean across all workers
    group_cols = ["jobId", "timestamp"]
    if not all(col in w_agg.columns for col in group_cols):
        print(f"  [WARNING] Worker metrics missing group columns. Skipping join.")
        w_job = pd.DataFrame(columns=["jobId", "w_timestamp"] + [f"w_{c}" for c in w_num])
    else:
        w_job = (w_agg.groupby(group_cols)[w_num]
                 .mean().reset_index()
                 .rename(columns={c: f"w_{c}" for c in w_num})
                 .rename(columns={"timestamp": "w_timestamp"}))

    parts = []
    for jid, grp_m in map_df.groupby("jobId"):
        grp_w = w_job[w_job["jobId"] == jid].sort_values("w_timestamp")
        grp_m = grp_m.sort_values("timestamp")
        if grp_w.empty:
            parts.append(grp_m)
            continue
        merged = pd.merge_asof(
            grp_m, grp_w.drop(columns="jobId"),
            left_on="timestamp", right_on="w_timestamp",
            direction="nearest"
        )
        parts.append(merged)

    df = pd.concat(parts, ignore_index=True)

    # ── 5. job details (mapProgress, mapsRunning, …) ─────────────────────────
    jd_num_cols = ["mapProgress", "reduceProgress", "mapsCompleted",
                   "reducesCompleted", "mapsTotal", "reducesTotal",
                   "mapsRunning", "mapsPending"]
    for c in jd_num_cols:
        if c in jd_df.columns:
            jd_df[c] = pd.to_numeric(jd_df[c], errors="coerce")
    jd_df["timestamp"] = pd.to_numeric(jd_df["timestamp"], errors="coerce")
    jd_sel_cols = ["jobId", "timestamp"] + [c for c in jd_num_cols if c in jd_df.columns]
    jd_sel = jd_df[jd_sel_cols].rename(columns={"timestamp": "jd_timestamp"})
    if "mapProgress" in jd_sel.columns:
        jd_sel = jd_sel.dropna(subset=["mapProgress"])

    df = _nearest_merge(df, jd_sel, "timestamp", "jd_timestamp")

    # ── 6. cluster metrics ────────────────────────────────────────────────────
    cl_num = ["cpuUsage", "memoryUsage", "availableMB", "allocatedMB",
              "availableVirtualCores", "allocatedVirtualCores"]
    for c in cl_num:
        if c in cl_df.columns:
            cl_df[c] = pd.to_numeric(cl_df[c], errors="coerce")
    cl_df["timestamp"] = pd.to_numeric(cl_df["timestamp"], errors="coerce")
    cl_sel_cols = ["jobId", "timestamp"] + [c for c in cl_num if c in cl_df.columns]
    cl_sel = cl_df[cl_sel_cols].rename(columns={"timestamp": "cl_timestamp"})

    df = _nearest_merge(df, cl_sel, "timestamp", "cl_timestamp")

    # ── 7. job-level progress ─────────────────────────────────────────────────
    job_num = ["progress", "elapsedTime", "allocatedMB",
               "allocatedVCores", "runningContainers", "clusterUsagePercentage"]
    for c in job_num:
        if c in job_df.columns:
            job_df[c] = pd.to_numeric(job_df[c], errors="coerce")
    job_df["timestamp"] = pd.to_numeric(job_df["timestamp"], errors="coerce")
    job_sel_cols = ["jobId", "timestamp"] + [c for c in job_num if c in job_df.columns]
    job_sel = job_df[job_sel_cols].rename(
        columns={c: f"job_{c}" for c in job_num}
    ).rename(columns={"timestamp": "job_timestamp"})

    df = _nearest_merge(df, job_sel, "timestamp", "job_timestamp")

    # ── 8. add benchmark label ─────────────────────────────────────────────────
    df["benchmark"] = benchmark_name

    return df


def load_all_benchmarks(dataset_root: str) -> pd.DataFrame:
    """
    Load all 5 benchmarks from AUtool+ and Heterogeneity.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_dirs = [
        os.path.join(project_root, "AUtool+"),
        os.path.join(project_root, "Heterogeneity")
    ]
    
    benchmark_paths = {
        "Grep": "Grep",
        "Histogram-Ratings": "Histogram-Ratings",
        "HistogramMovie": "HistogramMovie",
        "PageRank": "PageRank",
        "WordCount": "WordCount"
    }
    
    parts = []
    for name, subpath in benchmark_paths.items():
        found = False
        for s_dir in search_dirs:
            path = os.path.join(s_dir, subpath)
            if os.path.isdir(path):
                print(f"  Loading {name} from {s_dir}...", flush=True)
                try:
                    parts.append(build_feature_matrix(path, name))
                    found = True
                    break
                except Exception as e:
                    print(f"  [WARNING] {name} failed: {e}")
        if not found:
            print(f"  [WARNING] Benchmark {name} not found in search paths.")
            
    if not parts:
        raise ValueError("No benchmarks loaded. Check dataset paths.")
        
    df = pd.concat(parts, ignore_index=True)
    print(f"  Total rows loaded: {len(df):,}")
    return df
