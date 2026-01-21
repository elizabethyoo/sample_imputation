"""
Cross-sensor concordance analysis.

Compare accelerometer-based wear classification with GPS and gyroscope data
to validate wear detection and potentially inform non-wear imputation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from scipy.spatial.distance import cdist
import warnings


def compute_gps_displacement(
    gps_df: pd.DataFrame,
    time_window_minutes: int = 1
) -> pd.DataFrame:
    """
    Compute GPS-based displacement over time windows.
    
    Parameters
    ----------
    gps_df : pd.DataFrame
        GPS data with timestamp, latitude, longitude columns
    time_window_minutes : int
        Time window for aggregation
        
    Returns
    -------
    pd.DataFrame
        One row per time window with columns:
        - minute_start: datetime
        - has_gps: bool
        - n_gps_points: int
        - displacement_m: float (total displacement within window)
        - mean_accuracy: float
    """
    if gps_df.empty:
        return pd.DataFrame(columns=[
            'minute_start', 'has_gps', 'n_gps_points', 
            'displacement_m', 'mean_accuracy'
        ])
    
    df = gps_df.copy()
    df['minute'] = df['timestamp'].dt.floor(f'{time_window_minutes}min')
    
    results = []
    
    for minute_start, group in df.groupby('minute'):
        result = {
            'minute_start': minute_start,
            'has_gps': True,
            'n_gps_points': len(group)
        }
        
        if len(group) >= 2:
            # Compute displacement using Haversine formula
            displacement = compute_total_displacement(
                group['latitude'].values,
                group['longitude'].values
            )
            result['displacement_m'] = displacement
        else:
            result['displacement_m'] = 0.0
        
        if 'accuracy' in group.columns:
            result['mean_accuracy'] = group['accuracy'].mean()
        else:
            result['mean_accuracy'] = np.nan
        
        results.append(result)
    
    return pd.DataFrame(results)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute Haversine distance between two points in meters.
    
    Parameters
    ----------
    lat1, lon1 : float
        Coordinates of first point (degrees)
    lat2, lon2 : float
        Coordinates of second point (degrees)
        
    Returns
    -------
    float
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def compute_total_displacement(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Compute total path displacement from sequence of coordinates.
    
    Parameters
    ----------
    lats : np.ndarray
        Latitude values
    lons : np.ndarray
        Longitude values (may be negative or >180 for western hemisphere)
        
    Returns
    -------
    float
        Total displacement in meters
    """
    if len(lats) < 2:
        return 0.0
    
    # Handle longitude wraparound (e.g., -252 should be 108 W = -108)
    lons = np.where(lons < -180, lons + 360, lons)
    lons = np.where(lons > 180, lons - 360, lons)
    
    total_dist = 0.0
    for i in range(len(lats) - 1):
        total_dist += haversine_distance(
            lats[i], lons[i], lats[i+1], lons[i+1]
        )
    
    return total_dist


def compute_gyro_activity(
    gyro_df: pd.DataFrame,
    time_window_minutes: int = 1
) -> pd.DataFrame:
    """
    Compute gyroscope-based activity metrics over time windows.
    
    Parameters
    ----------
    gyro_df : pd.DataFrame
        Gyroscope data with timestamp, x, y, z columns
    time_window_minutes : int
        Time window for aggregation
        
    Returns
    -------
    pd.DataFrame
        One row per time window with columns:
        - minute_start: datetime
        - has_gyro: bool
        - n_gyro_points: int
        - mean_angular_rate: float (mean of magnitude)
        - std_angular_rate: float (std of magnitude)
    """
    if gyro_df.empty:
        return pd.DataFrame(columns=[
            'minute_start', 'has_gyro', 'n_gyro_points',
            'mean_angular_rate', 'std_angular_rate'
        ])
    
    df = gyro_df.copy()
    
    # Compute angular rate magnitude
    df['angular_rate'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    df['minute'] = df['timestamp'].dt.floor(f'{time_window_minutes}min')
    
    results = []
    
    for minute_start, group in df.groupby('minute'):
        result = {
            'minute_start': minute_start,
            'has_gyro': True,
            'n_gyro_points': len(group),
            'mean_angular_rate': group['angular_rate'].mean(),
            'std_angular_rate': group['angular_rate'].std()
        }
        results.append(result)
    
    return pd.DataFrame(results)


def merge_sensor_data(
    wear_df: pd.DataFrame,
    gps_activity: pd.DataFrame,
    gyro_activity: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge accelerometer wear classification with GPS and gyro activity.
    
    Parameters
    ----------
    wear_df : pd.DataFrame
        Output from detect_wear_minutes
    gps_activity : pd.DataFrame
        Output from compute_gps_displacement
    gyro_activity : pd.DataFrame
        Output from compute_gyro_activity
        
    Returns
    -------
    pd.DataFrame
        Merged data with all sensor information aligned by minute
    """
    # Start with wear data
    result = wear_df.copy()
    
    # Merge GPS
    if not gps_activity.empty:
        result = result.merge(
            gps_activity,
            on='minute_start',
            how='left'
        )
        result['has_gps'] = result['has_gps'].fillna(False)
    else:
        result['has_gps'] = False
        result['n_gps_points'] = 0
        result['displacement_m'] = np.nan
        result['mean_accuracy'] = np.nan
    
    # Merge gyro
    if not gyro_activity.empty:
        result = result.merge(
            gyro_activity,
            on='minute_start',
            how='left'
        )
        result['has_gyro'] = result['has_gyro'].fillna(False)
    else:
        result['has_gyro'] = False
        result['n_gyro_points'] = 0
        result['mean_angular_rate'] = np.nan
        result['std_angular_rate'] = np.nan
    
    return result


def compute_concordance_metrics(
    merged_df: pd.DataFrame,
    displacement_threshold_m: float = 10.0,
    angular_rate_threshold: float = 1.0
) -> Dict:
    """
    Compute concordance metrics between accelerometer wear and other sensors.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Output from merge_sensor_data
    displacement_threshold_m : float
        Threshold for considering GPS as showing "movement"
    angular_rate_threshold : float
        Threshold for considering gyro as showing "movement"
        
    Returns
    -------
    Dict with concordance statistics:
        - acc_wear_gps_move: proportion of acc-wear minutes with GPS movement
        - acc_nonwear_gps_stationary: proportion of acc-nonwear minutes with no GPS movement
        - acc_wear_gyro_active: proportion of acc-wear minutes with gyro activity
        - acc_nonwear_gyro_inactive: proportion of acc-nonwear minutes with low gyro
        - gps_coverage: proportion of valid minutes with GPS data
        - gyro_coverage: proportion of valid minutes with gyro data
    """
    # Filter to valid accelerometer minutes
    valid_df = merged_df[merged_df['valid']].copy()
    
    if valid_df.empty:
        return {
            'acc_wear_gps_move': np.nan,
            'acc_nonwear_gps_stationary': np.nan,
            'acc_wear_gyro_active': np.nan,
            'acc_nonwear_gyro_inactive': np.nan,
            'gps_coverage': np.nan,
            'gyro_coverage': np.nan,
            'n_valid_minutes': 0
        }
    
    n_valid = len(valid_df)
    
    # GPS concordance
    gps_valid = valid_df[valid_df['has_gps']].copy()
    gps_coverage = len(gps_valid) / n_valid if n_valid > 0 else np.nan
    
    if len(gps_valid) > 0:
        gps_valid['gps_moving'] = gps_valid['displacement_m'] > displacement_threshold_m
        
        wear_with_gps = gps_valid[gps_valid['wear'] == True]
        nonwear_with_gps = gps_valid[gps_valid['wear'] == False]
        
        acc_wear_gps_move = (
            wear_with_gps['gps_moving'].mean() 
            if len(wear_with_gps) > 0 else np.nan
        )
        acc_nonwear_gps_stationary = (
            (~nonwear_with_gps['gps_moving']).mean() 
            if len(nonwear_with_gps) > 0 else np.nan
        )
    else:
        acc_wear_gps_move = np.nan
        acc_nonwear_gps_stationary = np.nan
    
    # Gyro concordance
    gyro_valid = valid_df[valid_df['has_gyro']].copy()
    gyro_coverage = len(gyro_valid) / n_valid if n_valid > 0 else np.nan
    
    if len(gyro_valid) > 0:
        gyro_valid['gyro_active'] = gyro_valid['mean_angular_rate'] > angular_rate_threshold
        
        wear_with_gyro = gyro_valid[gyro_valid['wear'] == True]
        nonwear_with_gyro = gyro_valid[gyro_valid['wear'] == False]
        
        acc_wear_gyro_active = (
            wear_with_gyro['gyro_active'].mean() 
            if len(wear_with_gyro) > 0 else np.nan
        )
        acc_nonwear_gyro_inactive = (
            (~nonwear_with_gyro['gyro_active']).mean() 
            if len(nonwear_with_gyro) > 0 else np.nan
        )
    else:
        acc_wear_gyro_active = np.nan
        acc_nonwear_gyro_inactive = np.nan
    
    return {
        'acc_wear_gps_move': acc_wear_gps_move,
        'acc_nonwear_gps_stationary': acc_nonwear_gps_stationary,
        'acc_wear_gyro_active': acc_wear_gyro_active,
        'acc_nonwear_gyro_inactive': acc_nonwear_gyro_inactive,
        'gps_coverage': gps_coverage,
        'gyro_coverage': gyro_coverage,
        'n_valid_minutes': n_valid
    }


def identify_informative_nonwear(
    merged_df: pd.DataFrame,
    displacement_threshold_m: float = 10.0
) -> pd.DataFrame:
    """
    Identify non-wear periods where GPS suggests the person was actually moving.
    
    This can help identify potential "informative missingness" where the device
    was not worn during active periods.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Output from merge_sensor_data
    displacement_threshold_m : float
        Threshold for considering GPS as showing "movement"
        
    Returns
    -------
    pd.DataFrame
        Subset of merged_df where accelerometer shows non-wear but GPS shows movement
    """
    if merged_df.empty:
        return merged_df
    
    # Valid accelerometer, classified as non-wear, but GPS shows movement
    mask = (
        (merged_df['valid'] == True) &
        (merged_df['wear'] == False) &
        (merged_df['has_gps'] == True) &
        (merged_df['displacement_m'] > displacement_threshold_m)
    )
    
    return merged_df[mask].copy()
