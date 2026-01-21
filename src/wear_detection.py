"""
Wear detection using hypothesis testing on accelerometer magnitude.

Statistical Framework:
    Under H_0 (device stationary): 
        X(t), Y(t), Z(t) ~ N(0, σ²) independently
        => M(t)² / σ² ~ χ²(3)
        => M(t) / σ ~ χ(3) (chi distribution with 3 df)
    
    Under H_1 (device moving):
        At least one of X_acc, Y_acc, Z_acc ≠ 0
        => M(t) is stochastically larger
    
    Test: One-sided upper-tail test on M(t)² / σ²
    Reject H_0 (conclude movement/wear) if T > χ²(3, 1-α)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import warnings


@dataclass
class WearDetectionParams:
    """Parameters for wear detection hypothesis test."""
    noise_sigma: float  # Standard deviation of noise per axis (in g units)
    alpha: float = 0.05  # Significance level
    aggregation_method: str = 'mean'  # How to aggregate within a minute: 'mean', 'median', 'max'
    min_samples_per_minute: int = 100  # Minimum samples needed to classify a minute
    
    def __post_init__(self):
        if self.noise_sigma <= 0:
            raise ValueError("noise_sigma must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.aggregation_method not in ['mean', 'median', 'max']:
            raise ValueError("aggregation_method must be 'mean', 'median', or 'max'")


def estimate_noise_sigma_from_stationary(
    df: pd.DataFrame,
    stationary_hours: Tuple[int, int] = (2, 5),
    timestamp_col: str = 'timestamp',
    method: str = 'pooled_std'
) -> float:
    """
    Estimate noise standard deviation from assumed stationary periods.
    
    Assumes device is stationary during certain hours (e.g., 2-5 AM).
    During stationary periods, the signal should be approximately:
        X ≈ ε_X, Y ≈ ε_Y, Z ≈ g + ε_Z (if lying flat)
    
    We estimate σ from the variation around the mean during these periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Accelerometer data with timestamp and x, y, z columns
    stationary_hours : Tuple[int, int]
        Start and end hour (24h format) of assumed stationary period
    timestamp_col : str
        Name of timestamp column
    method : str
        Estimation method: 'pooled_std', 'mad', or 'per_axis_mean'
        
    Returns
    -------
    float
        Estimated noise standard deviation (per axis)
    """
    if df.empty:
        raise ValueError("Cannot estimate noise from empty DataFrame")
    
    start_hour, end_hour = stationary_hours
    
    # Filter to stationary hours
    hours = df[timestamp_col].dt.hour
    mask = (hours >= start_hour) & (hours < end_hour)
    stationary_df = df[mask]
    
    if len(stationary_df) < 100:
        warnings.warn(
            f"Only {len(stationary_df)} samples in stationary period. "
            "Estimate may be unreliable."
        )
    
    if len(stationary_df) == 0:
        raise ValueError("No data in specified stationary period")
    
    if method == 'pooled_std':
        # Remove the mean (bias) from each axis, then compute pooled std
        x_centered = stationary_df['x'] - stationary_df['x'].mean()
        y_centered = stationary_df['y'] - stationary_df['y'].mean()
        z_centered = stationary_df['z'] - stationary_df['z'].mean()
        
        # Pooled variance estimate
        n = len(stationary_df)
        pooled_var = (x_centered.var() + y_centered.var() + z_centered.var()) / 3
        sigma = np.sqrt(pooled_var)
        
    elif method == 'mad':
        # Use median absolute deviation (more robust to outliers)
        # MAD ≈ 0.6745 * σ for normal distribution
        x_mad = np.median(np.abs(stationary_df['x'] - stationary_df['x'].median()))
        y_mad = np.median(np.abs(stationary_df['y'] - stationary_df['y'].median()))
        z_mad = np.median(np.abs(stationary_df['z'] - stationary_df['z'].median()))
        
        sigma = np.mean([x_mad, y_mad, z_mad]) / 0.6745
        
    elif method == 'per_axis_mean':
        # Simple mean of per-axis standard deviations
        sigma = np.mean([
            stationary_df['x'].std(),
            stationary_df['y'].std(),
            stationary_df['z'].std()
        ])
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return sigma


def compute_test_statistic(
    magnitude_squared: float,
    sigma: float
) -> float:
    """
    Compute the chi-squared test statistic.
    
    T = M² / σ² ~ χ²(3) under H_0
    
    Parameters
    ----------
    magnitude_squared : float
        Squared magnitude M² = X² + Y² + Z²
    sigma : float
        Noise standard deviation per axis
        
    Returns
    -------
    float
        Test statistic T
    """
    return magnitude_squared / (sigma ** 2)


def get_critical_value(alpha: float, df: int = 3) -> float:
    """
    Get the critical value for the chi-squared test.
    
    Parameters
    ----------
    alpha : float
        Significance level
    df : int
        Degrees of freedom
        
    Returns
    -------
    float
        Critical value χ²(df, 1-α)
    """
    return stats.chi2.ppf(1 - alpha, df)


def classify_minute_wear(
    minute_data: pd.DataFrame,
    params: WearDetectionParams
) -> Dict:
    """
    Classify a single minute as wear or non-wear.
    
    Parameters
    ----------
    minute_data : pd.DataFrame
        Accelerometer data for one minute with x, y, z columns
    params : WearDetectionParams
        Detection parameters
        
    Returns
    -------
    Dict with keys:
        - 'wear': bool, True if classified as wear
        - 'test_statistic': float, the computed test statistic
        - 'critical_value': float, the critical value used
        - 'p_value': float, p-value of the test
        - 'n_samples': int, number of samples in the minute
        - 'mean_magnitude': float, mean magnitude for the minute
        - 'valid': bool, True if enough samples to make classification
    """
    n_samples = len(minute_data)
    
    result = {
        'n_samples': n_samples,
        'valid': n_samples >= params.min_samples_per_minute
    }
    
    if not result['valid']:
        result.update({
            'wear': None,
            'test_statistic': np.nan,
            'critical_value': np.nan,
            'p_value': np.nan,
            'mean_magnitude': np.nan
        })
        return result
    
    # Compute magnitude for each sample
    magnitudes = np.sqrt(
        minute_data['x']**2 + 
        minute_data['y']**2 + 
        minute_data['z']**2
    )
    
    # Aggregate magnitude for the minute
    if params.aggregation_method == 'mean':
        agg_magnitude = magnitudes.mean()
    elif params.aggregation_method == 'median':
        agg_magnitude = magnitudes.median()
    else:  # max
        agg_magnitude = magnitudes.max()
    
    result['mean_magnitude'] = magnitudes.mean()
    
    # Compute test statistic
    # For aggregated data, we use M² / σ²
    # Note: For mean of n samples, the variance is σ²/n, but the mean magnitude
    # doesn't follow chi distribution simply. We use a simplified approach:
    # Test if the observed magnitude is significantly above noise level.
    
    # Under H_0 with stationary device:
    # E[M] ≈ σ * sqrt(2) * Γ(2)/Γ(1.5) ≈ 1.60 * σ for χ(3)
    # We test if M² > threshold
    
    test_stat = compute_test_statistic(agg_magnitude ** 2, params.noise_sigma)
    critical_val = get_critical_value(params.alpha)
    p_value = 1 - stats.chi2.cdf(test_stat, df=3)
    
    result.update({
        'wear': test_stat > critical_val,  # Reject H_0 => wear
        'test_statistic': test_stat,
        'critical_value': critical_val,
        'p_value': p_value
    })
    
    return result


def detect_wear_minutes(
    df: pd.DataFrame,
    params: WearDetectionParams,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Classify each minute of data as wear or non-wear.
    
    Parameters
    ----------
    df : pd.DataFrame
        Accelerometer data with timestamp, x, y, z columns
    params : WearDetectionParams
        Detection parameters
    timestamp_col : str
        Name of timestamp column
        
    Returns
    -------
    pd.DataFrame
        One row per minute with columns:
        - minute_start: datetime
        - wear: bool or None
        - test_statistic: float
        - critical_value: float
        - p_value: float
        - n_samples: int
        - mean_magnitude: float
        - valid: bool
    """
    if df.empty:
        return pd.DataFrame(columns=[
            'minute_start', 'wear', 'test_statistic', 'critical_value',
            'p_value', 'n_samples', 'mean_magnitude', 'valid'
        ])
    
    # Floor timestamps to minute
    df = df.copy()
    df['minute'] = df[timestamp_col].dt.floor('min')
    
    results = []
    
    for minute_start, minute_group in df.groupby('minute'):
        classification = classify_minute_wear(minute_group, params)
        classification['minute_start'] = minute_start
        results.append(classification)
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('minute_start').reset_index(drop=True)
    
    # Reorder columns
    col_order = [
        'minute_start', 'wear', 'test_statistic', 'critical_value',
        'p_value', 'n_samples', 'mean_magnitude', 'valid'
    ]
    result_df = result_df[col_order]
    
    return result_df


def compute_daily_wear_summary(
    wear_df: pd.DataFrame,
    timezone: str = 'UTC'
) -> pd.DataFrame:
    """
    Compute daily summary of wear time.
    
    Parameters
    ----------
    wear_df : pd.DataFrame
        Output from detect_wear_minutes
    timezone : str
        Timezone for defining "day"
        
    Returns
    -------
    pd.DataFrame
        One row per day with columns:
        - date: date
        - total_valid_minutes: int
        - wear_minutes: int
        - nonwear_minutes: int
        - wear_fraction: float
        - invalid_minutes: int
    """
    if wear_df.empty:
        return pd.DataFrame(columns=[
            'date', 'total_valid_minutes', 'wear_minutes', 
            'nonwear_minutes', 'wear_fraction', 'invalid_minutes'
        ])
    
    df = wear_df.copy()
    
    # Convert to target timezone and extract date
    df['local_time'] = df['minute_start'].dt.tz_convert(timezone)
    df['date'] = df['local_time'].dt.date
    
    # Aggregate by date
    daily = df.groupby('date').agg(
        total_valid_minutes=('valid', lambda x: x.sum()),
        wear_minutes=('wear', lambda x: x.fillna(False).sum()),
        invalid_minutes=('valid', lambda x: (~x).sum())
    ).reset_index()
    
    daily['nonwear_minutes'] = daily['total_valid_minutes'] - daily['wear_minutes']
    daily['wear_fraction'] = daily['wear_minutes'] / daily['total_valid_minutes'].replace(0, np.nan)
    
    return daily


def compute_hourly_wear_pattern(
    wear_df: pd.DataFrame,
    timezone: str = 'UTC'
) -> pd.DataFrame:
    """
    Compute hourly wear patterns (for visualization of temporal distribution).
    
    Parameters
    ----------
    wear_df : pd.DataFrame
        Output from detect_wear_minutes
    timezone : str
        Timezone for defining "hour of day"
        
    Returns
    -------
    pd.DataFrame
        One row per hour (0-23) with columns:
        - hour: int
        - total_minutes: int (across all days)
        - wear_minutes: int
        - wear_rate: float (proportion of time worn at this hour)
    """
    if wear_df.empty:
        # Return empty pattern with all 24 hours
        return pd.DataFrame({
            'hour': range(24),
            'total_minutes': [0] * 24,
            'wear_minutes': [0] * 24,
            'wear_rate': [np.nan] * 24
        })
    
    df = wear_df.copy()
    
    # Convert to target timezone and extract hour
    df['local_time'] = df['minute_start'].dt.tz_convert(timezone)
    df['hour'] = df['local_time'].dt.hour
    
    # Only consider valid minutes
    valid_df = df[df['valid']]
    
    if valid_df.empty:
        return pd.DataFrame({
            'hour': range(24),
            'total_minutes': [0] * 24,
            'wear_minutes': [0] * 24,
            'wear_rate': [np.nan] * 24
        })
    
    # Aggregate by hour
    hourly = valid_df.groupby('hour').agg(
        total_minutes=('valid', 'count'),
        wear_minutes=('wear', 'sum')
    ).reset_index()
    
    # Ensure all 24 hours are represented
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly = all_hours.merge(hourly, on='hour', how='left')
    hourly = hourly.fillna({'total_minutes': 0, 'wear_minutes': 0})
    
    hourly['wear_rate'] = hourly['wear_minutes'] / hourly['total_minutes'].replace(0, np.nan)
    
    return hourly
