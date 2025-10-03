"""
Beiwe accelerometer data processing utilities.
"""

import pyarrow.parquet as pq
import pyarrow as pa
from collections import defaultdict
from pathlib import Path
import re
import random
import math
import warnings
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pytz
from datetime import timedelta

# Constants
DTYPE_MAP = {
    "timestamp": "int64",
    "UTC time": "string", 
    "accuracy": "string",
    "x": "float32", "y": "float32", "z": "float32",
}

USECOLS = ["timestamp", "UTC time", "accuracy", "x", "y", "z"]
HOURLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}_00_00\+00_00\.csv$")
US_EASTERN = pytz.timezone("America/New_York")

# File I/O functions
def read_hour_file(path: Path) -> pd.DataFrame:
    """Read hourly accelerometer CSV file with standard preprocessing."""
    df = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=DTYPE_MAP,
        na_values=["", "NA", "NaN", "null", "None"]
    )
    df["_orig_order"] = np.arange(len(df), dtype=np.int64)
    return df

def list_subjects(input_root: Path) -> List[str]:
    """List all subject directories in input root."""
    if not input_root.exists():
        raise FileNotFoundError(f"{input_root} not found")
    subs = sorted([p.name for p in input_root.iterdir() if p.is_dir()])
    return subs

def accel_dir(subject_id: str, input_root: Path) -> Path:
    """Get accelerometer directory path for a subject."""
    return input_root / subject_id / "accelerometer"

def list_hourly_files(subject_id: str, input_root: Path) -> List[Path]:
    """List hourly accelerometer files for a subject."""
    acc_dir = accel_dir(subject_id, input_root)
    if not acc_dir.exists():
        logging.warning(f"No accelerometer folder for subject {subject_id} at {acc_dir}")
        return []
    
    files = []
    skipped = 0
    for p in acc_dir.iterdir():
        if p.is_file():
            if HOURLY_RE.match(p.name):
                files.append(p)
            else:
                skipped += 1
    
    files = sorted(files, key=lambda p: p.name)
    if skipped:
        logging.warning(f"{subject_id}: skipped {skipped} non-matching files.")
    return files


def get_first_N_files(subject_id: str, input_root: Path, N: int = 7) -> List[Path]:
    """Get first N hourly files for a subject."""
    files = list_hourly_files(subject_id, input_root)
    if not files:
        raise FileNotFoundError(f"No files found for subject '{subject_id}'")
    if len(files) < N:
        raise ValueError(
            f"Subject '{subject_id}' has only {len(files)} files, but {N} were requested")
    return files[:N]