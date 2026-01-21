"""
Main analysis pipeline for wear detection and characterization.

This module orchestrates the full pipeline:
1. Load data for all subjects
2. Estimate noise parameters
3. Detect wear/non-wear for each minute
4. Compute daily and hourly summaries
5. Compute cross-sensor concordance
6. Generate summary reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import warnings
from datetime import datetime

from .data_loader import (
    load_subject_data, 
    get_subject_ids, 
    convert_timezone
)
from .wear_detection import (
    WearDetectionParams,
    estimate_noise_sigma_from_stationary,
    detect_wear_minutes,
    compute_daily_wear_summary,
    compute_hourly_wear_pattern
)
from .concordance import (
    compute_gps_displacement,
    compute_gyro_activity,
    merge_sensor_data,
    compute_concordance_metrics
)


@dataclass
class SubjectConfig:
    """Configuration for a single subject."""
    subject_id: str
    timezone: str = 'US/Eastern'
    stationary_hours: Tuple[int, int] = (2, 5)


@dataclass 
class PipelineConfig:
    """Configuration for the full analysis pipeline."""
    data_dir: Path
    output_dir: Path
    subjects: List[SubjectConfig]
    alpha: float = 0.05
    aggregation_method: str = 'mean'
    min_samples_per_minute: int = 100
    noise_estimation_method: str = 'pooled_std'
    # If None, estimate from data; otherwise use this value
    fixed_noise_sigma: Optional[float] = None
    

@dataclass
class SubjectResults:
    """Results for a single subject."""
    subject_id: str
    noise_sigma: float
    wear_minutes: pd.DataFrame
    daily_summary: pd.DataFrame
    hourly_pattern: pd.DataFrame
    merged_sensor_data: pd.DataFrame
    concordance_metrics: Dict
    

def run_subject_pipeline(
    config: PipelineConfig,
    subject_config: SubjectConfig
) -> SubjectResults:
    """
    Run the full pipeline for a single subject.
    
    Parameters
    ----------
    config : PipelineConfig
        Global pipeline configuration
    subject_config : SubjectConfig
        Subject-specific configuration
        
    Returns
    -------
    SubjectResults
        All results for the subject
    """
    subject_id = subject_config.subject_id
    print(f"Processing subject: {subject_id}")
    
    # Load accelerometer data
    print(f"  Loading accelerometer data...")
    acc_df = load_subject_data(
        config.data_dir, 
        subject_id, 
        'accelerometer'
    )
    
    if acc_df.empty:
        raise ValueError(f"No accelerometer data found for subject {subject_id}")
    
    print(f"  Loaded {len(acc_df)} accelerometer samples")
    
    # Estimate or use fixed noise parameter
    if config.fixed_noise_sigma is not None:
        noise_sigma = config.fixed_noise_sigma
        print(f"  Using fixed noise sigma: {noise_sigma:.6f} g")
    else:
        print(f"  Estimating noise from stationary periods...")
        # Convert to local timezone for hour filtering
        acc_local = convert_timezone(acc_df, subject_config.timezone)
        try:
            noise_sigma = estimate_noise_sigma_from_stationary(
                acc_local,
                stationary_hours=subject_config.stationary_hours,
                method=config.noise_estimation_method
            )
            print(f"  Estimated noise sigma: {noise_sigma:.6f} g")
        except ValueError as e:
            warnings.warn(f"Could not estimate noise for {subject_id}: {e}")
            # Fall back to a typical value from literature
            noise_sigma = 0.01  # ~10 mg typical for smartphone accelerometers
            print(f"  Using fallback noise sigma: {noise_sigma:.6f} g")
    
    # Create wear detection parameters
    params = WearDetectionParams(
        noise_sigma=noise_sigma,
        alpha=config.alpha,
        aggregation_method=config.aggregation_method,
        min_samples_per_minute=config.min_samples_per_minute
    )
    
    # Detect wear/non-wear
    print(f"  Detecting wear/non-wear...")
    wear_df = detect_wear_minutes(acc_df, params)
    print(f"  Classified {len(wear_df)} minutes")
    
    # Compute daily summary
    print(f"  Computing daily summary...")
    daily_summary = compute_daily_wear_summary(wear_df, subject_config.timezone)
    
    # Compute hourly pattern
    print(f"  Computing hourly pattern...")
    hourly_pattern = compute_hourly_wear_pattern(wear_df, subject_config.timezone)
    
    # Load GPS and gyroscope data for concordance
    print(f"  Loading GPS data...")
    try:
        gps_df = load_subject_data(config.data_dir, subject_id, 'gps')
        print(f"  Loaded {len(gps_df)} GPS samples")
        gps_activity = compute_gps_displacement(gps_df)
    except FileNotFoundError:
        print(f"  No GPS data found")
        gps_activity = pd.DataFrame()
    
    print(f"  Loading gyroscope data...")
    try:
        gyro_df = load_subject_data(config.data_dir, subject_id, 'gyro')
        print(f"  Loaded {len(gyro_df)} gyroscope samples")
        gyro_activity = compute_gyro_activity(gyro_df)
    except FileNotFoundError:
        print(f"  No gyroscope data found")
        gyro_activity = pd.DataFrame()
    
    # Merge sensor data
    print(f"  Computing cross-sensor concordance...")
    merged_df = merge_sensor_data(wear_df, gps_activity, gyro_activity)
    concordance = compute_concordance_metrics(merged_df)
    
    return SubjectResults(
        subject_id=subject_id,
        noise_sigma=noise_sigma,
        wear_minutes=wear_df,
        daily_summary=daily_summary,
        hourly_pattern=hourly_pattern,
        merged_sensor_data=merged_df,
        concordance_metrics=concordance
    )


def run_full_pipeline(config: PipelineConfig) -> Dict[str, SubjectResults]:
    """
    Run the pipeline for all subjects.
    
    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
        
    Returns
    -------
    Dict[str, SubjectResults]
        Results keyed by subject ID
    """
    results = {}
    
    for subject_config in config.subjects:
        try:
            result = run_subject_pipeline(config, subject_config)
            results[subject_config.subject_id] = result
        except Exception as e:
            warnings.warn(f"Error processing {subject_config.subject_id}: {e}")
            continue
    
    return results


def aggregate_results(results: Dict[str, SubjectResults]) -> Dict:
    """
    Aggregate results across all subjects.
    
    Parameters
    ----------
    results : Dict[str, SubjectResults]
        Results from run_full_pipeline
        
    Returns
    -------
    Dict with aggregated statistics
    """
    if not results:
        return {}
    
    # Collect daily summaries
    all_daily = []
    for subject_id, result in results.items():
        df = result.daily_summary.copy()
        df['subject_id'] = subject_id
        all_daily.append(df)
    
    combined_daily = pd.concat(all_daily, ignore_index=True)
    
    # Collect hourly patterns
    all_hourly = []
    for subject_id, result in results.items():
        df = result.hourly_pattern.copy()
        df['subject_id'] = subject_id
        all_hourly.append(df)
    
    combined_hourly = pd.concat(all_hourly, ignore_index=True)
    
    # Collect concordance metrics
    concordance_list = []
    for subject_id, result in results.items():
        metrics = result.concordance_metrics.copy()
        metrics['subject_id'] = subject_id
        metrics['noise_sigma'] = result.noise_sigma
        concordance_list.append(metrics)
    
    concordance_df = pd.DataFrame(concordance_list)
    
    # Compute aggregate statistics
    aggregate = {
        'n_subjects': len(results),
        'daily_wear_summary': {
            'mean_valid_minutes': combined_daily['total_valid_minutes'].mean(),
            'std_valid_minutes': combined_daily['total_valid_minutes'].std(),
            'mean_wear_fraction': combined_daily['wear_fraction'].mean(),
            'std_wear_fraction': combined_daily['wear_fraction'].std(),
            'min_wear_fraction': combined_daily['wear_fraction'].min(),
            'max_wear_fraction': combined_daily['wear_fraction'].max()
        },
        'noise_sigma_summary': {
            'mean': concordance_df['noise_sigma'].mean(),
            'std': concordance_df['noise_sigma'].std(),
            'min': concordance_df['noise_sigma'].min(),
            'max': concordance_df['noise_sigma'].max()
        },
        'concordance_summary': {
            'mean_gps_coverage': concordance_df['gps_coverage'].mean(),
            'mean_gyro_coverage': concordance_df['gyro_coverage'].mean()
        },
        'combined_daily': combined_daily,
        'combined_hourly': combined_hourly,
        'concordance_df': concordance_df
    }
    
    return aggregate


def save_results(
    results: Dict[str, SubjectResults],
    aggregate: Dict,
    output_dir: Path
):
    """
    Save all results to files.
    
    Parameters
    ----------
    results : Dict[str, SubjectResults]
        Per-subject results
    aggregate : Dict
        Aggregated results
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-subject results
    subject_dir = output_dir / 'subjects'
    subject_dir.mkdir(exist_ok=True)
    
    for subject_id, result in results.items():
        subj_dir = subject_dir / subject_id
        subj_dir.mkdir(exist_ok=True)
        
        result.wear_minutes.to_csv(subj_dir / 'wear_minutes.csv', index=False)
        result.daily_summary.to_csv(subj_dir / 'daily_summary.csv', index=False)
        result.hourly_pattern.to_csv(subj_dir / 'hourly_pattern.csv', index=False)
        result.merged_sensor_data.to_csv(subj_dir / 'merged_sensor_data.csv', index=False)
        
        with open(subj_dir / 'concordance_metrics.json', 'w') as f:
            # Convert any nan to None for JSON serialization
            metrics = {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in result.concordance_metrics.items()
            }
            json.dump(metrics, f, indent=2)
    
    # Save aggregate results
    if aggregate:
        aggregate['combined_daily'].to_csv(
            output_dir / 'combined_daily_summary.csv', index=False
        )
        aggregate['combined_hourly'].to_csv(
            output_dir / 'combined_hourly_pattern.csv', index=False
        )
        aggregate['concordance_df'].to_csv(
            output_dir / 'concordance_summary.csv', index=False
        )
        
        # Save summary statistics
        summary = {
            k: v for k, v in aggregate.items() 
            if not isinstance(v, pd.DataFrame)
        }
        with open(output_dir / 'aggregate_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}")
