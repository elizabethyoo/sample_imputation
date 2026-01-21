"""
Data loading utilities for Beiwe accelerometer, GPS, and gyroscope data.

File structure expected:
    project_root_dir/
        data/
            subject_id/
                accelerometer/
                    YYYY-MM-DD HH_00_00+00_00.csv
                gps/
                    YYYY-MM-DD HH_00_00+00_00.csv
                gyro/
                    YYYY-MM-DD HH_00_00+00_00.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import pytz
import warnings


def parse_hourly_filename(filename: str) -> Optional[datetime]:
    """
    Parse the datetime from a Beiwe hourly data filename.
    
    Expected format: 'YYYY-MM-DD HH_00_00+00_00.csv'
    
    Parameters
    ----------
    filename : str
        The filename to parse
        
    Returns
    -------
    datetime or None
        The parsed datetime in UTC, or None if parsing fails
    """
    try:
        # Remove .csv extension
        name = filename.replace('.csv', '')
        # Parse: 'YYYY-MM-DD HH_00_00+00_00' -> datetime
        # The format has spaces and underscores
        # Example: '2022-03-21 13_00_00+00_00'
        date_part, time_part = name.split(' ')
        hour = int(time_part.split('_')[0])
        dt = datetime.strptime(date_part, '%Y-%m-%d')
        dt = dt.replace(hour=hour, tzinfo=pytz.UTC)
        return dt
    except (ValueError, IndexError):
        return None


def load_accelerometer_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single accelerometer CSV file.
    
    Parameters
    ----------
    filepath : Path
        Path to the accelerometer CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, x, y, z
        timestamp is parsed as datetime with UTC timezone
    """
    df = pd.read_csv(filepath)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'x', 'y', 'z']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Select only required columns (drop accuracy if present)
    df = df[required_cols].copy()
    
    return df


def load_gps_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single GPS CSV file.
    
    Parameters
    ----------
    filepath : Path
        Path to the GPS CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, latitude, longitude, altitude, accuracy
        timestamp is converted to datetime with UTC timezone
    """
    df = pd.read_csv(filepath)
    
    # GPS has epoch timestamp in milliseconds
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    elif 'UTC time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['UTC time'], utc=True)
    else:
        raise ValueError("No valid timestamp column found in GPS data")
    
    # Select relevant columns
    cols_to_keep = ['timestamp']
    for col in ['latitude', 'longitude', 'altitude', 'accuracy']:
        if col in df.columns:
            cols_to_keep.append(col)
    
    df = df[cols_to_keep].copy()
    
    return df


def load_gyroscope_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single gyroscope CSV file.
    
    Parameters
    ----------
    filepath : Path
        Path to the gyroscope CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, x, y, z
        timestamp is converted to datetime with UTC timezone
    """
    df = pd.read_csv(filepath)
    
    # Gyro has epoch timestamp in milliseconds
    if 'timestamp' in df.columns and df['timestamp'].dtype in ['int64', 'float64']:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    elif 'UTC time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['UTC time'], utc=True)
    else:
        # Try parsing as datetime string
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Select only required columns
    required_cols = ['timestamp', 'x', 'y', 'z']
    df = df[required_cols].copy()
    
    return df


def load_subject_data(
    data_dir: Path,
    subject_id: str,
    modality: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load all data for a subject and modality within a date range.
    
    Parameters
    ----------
    data_dir : Path
        Root data directory containing subject folders
    subject_id : str
        Subject identifier
    modality : str
        One of 'accelerometer', 'gps', 'gyro'
    start_date : datetime, optional
        Start of date range (inclusive)
    end_date : datetime, optional
        End of date range (inclusive)
        
    Returns
    -------
    pd.DataFrame
        Concatenated data from all matching files, sorted by timestamp
    """
    modality_dir = data_dir / subject_id / modality
    
    if not modality_dir.exists():
        raise FileNotFoundError(f"Modality directory not found: {modality_dir}")
    
    # Get all CSV files
    csv_files = list(modality_dir.glob("*.csv"))
    
    if not csv_files:
        warnings.warn(f"No CSV files found in {modality_dir}")
        return pd.DataFrame()
    
    # Filter by date range if specified
    files_to_load = []
    for f in csv_files:
        file_dt = parse_hourly_filename(f.name)
        if file_dt is None:
            continue
        
        if start_date is not None and file_dt < start_date:
            continue
        if end_date is not None and file_dt > end_date:
            continue
        
        files_to_load.append(f)
    
    if not files_to_load:
        warnings.warn(f"No files match the date range in {modality_dir}")
        return pd.DataFrame()
    
    # Load appropriate loader based on modality
    loaders = {
        'accelerometer': load_accelerometer_file,
        'gps': load_gps_file,
        'gyro': load_gyroscope_file
    }
    
    if modality not in loaders:
        raise ValueError(f"Unknown modality: {modality}")
    
    loader = loaders[modality]
    
    # Load and concatenate
    dfs = []
    for f in sorted(files_to_load):
        try:
            df = loader(f)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Error loading {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicates
    result = result.drop_duplicates(subset=['timestamp'], keep='first')
    
    return result


def get_subject_ids(data_dir: Path) -> List[str]:
    """
    Get list of subject IDs from the data directory.
    
    Parameters
    ----------
    data_dir : Path
        Root data directory
        
    Returns
    -------
    List[str]
        List of subject IDs (directory names)
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    subjects = [
        d.name for d in data_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    return sorted(subjects)


def compute_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute acceleration magnitude from x, y, z components.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns x, y, z
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'magnitude' column
    """
    df = df.copy()
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    return df


def convert_timezone(
    df: pd.DataFrame, 
    target_tz: str,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Convert timestamps to a target timezone.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column
    target_tz : str
        Target timezone (e.g., 'US/Eastern')
    timestamp_col : str
        Name of timestamp column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with converted timestamps
    """
    df = df.copy()
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(target_tz)
    return df
