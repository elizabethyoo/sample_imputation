import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

# Timezone support: prefer Python 3.9+'s zoneinfo, fallback to pytz if unavailable
try:
    from zoneinfo import ZoneInfo  # type: ignore
    _HAS_ZONEINFO = True
except Exception:  # pragma: no cover - runtime guard
    ZoneInfo = None  # type: ignore
    _HAS_ZONEINFO = False
    try:
        import pytz  # type: ignore
    except Exception:  # pragma: no cover - runtime guard
        pytz = None  # type: ignore

# Helper functions used to read files, compute summaries,
# concatenate summaries, and plot 

def get_color(std):
    """Determine color based on standard deviation threshold."""
    if std < 0.01:
        return 'green'  # Very still
    elif std < 0.1:
        return 'yellow'  # Light movement
    elif std < 0.3:
        return 'orange'  # Moderate movement
    else:
        return 'red'    # High movement


def _resolve_timezone_string(tz_str: Optional[str]) -> Optional[str]:
    """Map simple aliases to IANA timezone names; pass through IANA names.

    Examples: 'us_east' -> 'America/New_York'
    """
    if tz_str is None:
        return None
    normalized = tz_str.strip().lower().replace('-', '_').replace(' ', '_')
    alias_map = {
        'us_east': 'America/New_York',
        'us_eastern': 'America/New_York',
        'eastern': 'America/New_York',
        'us_central': 'America/Chicago',
        'central': 'America/Chicago',
        'us_mountain': 'America/Denver',
        'mountain': 'America/Denver',
        'us_pacific': 'America/Los_Angeles',
        'pacific': 'America/Los_Angeles',
        'uk': 'Europe/London',
        'british': 'Europe/London',
        'utc': 'UTC',
    }
    return alias_map.get(normalized, tz_str)


def summarize_hourly_file(file_path, verbose=True):
    """Analyze a single hourly accelerometer CSV file."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            if verbose:
                print(f"Missing: {os.path.basename(file_path)}")
            return None
            
        # If it exists read CSV
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            if verbose:
                print(f"Empty CSV file: {os.path.basename(file_path)}")
            return None
        except pd.errors.ParserError as e:
            if verbose:
                print(f"CSV parsing error in {os.path.basename(file_path)}: {e}")
            return None
            
        # Handle empty files
        if len(df) == 0:
            if verbose:
                print(f"Empty: {os.path.basename(file_path)}")
            return None
            
        # Validate required columns
        required_cols = ['timestamp', 'x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if verbose:
                print(f"Missing columns in {os.path.basename(file_path)}: {missing_cols}")
            return None

    except Exception as e:
        if verbose:
            print(f"An unexpected error occurred with {os.path.basename(file_path)}: {str(e)}")
        return None

    # Convert timestamps
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    # Compute magnitude
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2) - 1.0 # subtract 1.0 to account for gravity

    # Calculate metrics
    n_rows = len(df)
    start_time = df['datetime_utc'].iloc[0]
    end_time = df['datetime_utc'].iloc[-1]

    # Duration in minutes
    duration_sec = (end_time - start_time).total_seconds()
    duration_min = duration_sec / 60

    # Sampling time in minutes (at 10 Hz)
    sampling_min = n_rows / 10 / 60

    # Duty cycle
    duty_cycle = sampling_min / duration_min

    # Count bursts
    df['time_diff_ms'] = df['timestamp'].diff()
    gaps = df['time_diff_ms'] > 1000
    gaps_count = gaps.sum()
    n_bursts = gaps_count + 1

    # Mean magnitude
    mean_magnitude = df['magnitude'].mean()

    return {
        'n_rows': n_rows,
        'start_time': start_time,
        'end_time': end_time,
        'duration_min': duration_min,
        'sampling_min': sampling_min,
        'duty_cycle': duty_cycle,
        'n_bursts': n_bursts,
        'mean_magnitude': mean_magnitude
    }


def summarize_period(subject_id, start_date, end_date, base_path, timezone: Optional[str] = None):
    """
    Summarize accelerometer data for a subject across a date range.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., '3si9xdvl')
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime
        End date in 'YYYY-MM-DD' format (inclusive)
    base_path : str
        Base directory path (e.g., 'sample_imputation/data/raw/')
    timezone : Optional[str]
        Desired local timezone for output column 'local_time'. Accepts IANA names
        (e.g., 'America/New_York') or simple aliases (e.g., 'us_east'). If None,
        'local_time' equals 'datetime_utc'.
    
    Returns:
    --------
    pd.DataFrame
        Summary with columns: subject_id, date, hour, datetime_utc,
        n_rows, duration_min, sampling_min, duty_cycle, n_bursts,
        mean_magnitude, std_magnitude
    """
    
    # Convert to datetime if strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Storage for results
    all_summaries = []

    # Path to accelerometer files
    accel_path = os.path.join(base_path, subject_id, 'accelerometer')

    print(f"Summarizing data for {subject_id} from "
          f"{start_date.date()} to {end_date.date()}...")
    print("=" * 70)

    # Loop through each date
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        print(f"\n{date_str}:")

        # Loop through 24 hours
        for hour in range(24):
            file_name = f"{date_str} {hour:02d}_00_00+00_00.csv"
            file_path = os.path.join(accel_path, file_name)

            result = summarize_hourly_file(file_path, verbose=False)

            if result is not None:
                # Calculate std (need to reload file)
                df_temp = pd.read_csv(file_path)
                df_temp['magnitude'] = (
                    np.sqrt(df_temp['x']**2 + df_temp['y']**2 +
                            df_temp['z']**2)
                ) - 1.0 # subtract 1.0 to account for gravity
                
                # Add metadata
                result['subject_id'] = subject_id
                result['date'] = date_str
                result['hour'] = hour
                result['std_magnitude'] = df_temp['magnitude'].std()

                # Create full datetime for the hour
                result['datetime_utc'] = pd.Timestamp(
                    f"{date_str} {hour:02d}:00:00", tz='UTC'
                )
                
                all_summaries.append(result)
                print(f"OK: Hour {hour:02d}: {result['n_rows']:,} rows")
            else:
                print(f"MISSING: Hour {hour:02d}: missing")
    
    print("\n" + "=" * 70)
    print(f"Processed {len(all_summaries)} hours across "
          f"{len(date_range)} days")

    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)

    # Sort by datetime
    summary_df = summary_df.sort_values('datetime_utc').reset_index(drop=True)

    # Add local_time column based on timezone argument
    resolved_tz = _resolve_timezone_string(timezone)
    if resolved_tz is None or resolved_tz == 'UTC':
        summary_df['local_time'] = summary_df['datetime_utc']
    else:
        if _HAS_ZONEINFO and ZoneInfo is not None:
            tzinfo = ZoneInfo(resolved_tz)
            summary_df['local_time'] = summary_df['datetime_utc'].dt.tz_convert(tzinfo)
        else:
            if 'pytz' in globals() and pytz is not None:
                tzinfo = pytz.timezone(resolved_tz)
                summary_df['local_time'] = summary_df['datetime_utc'].dt.tz_convert(tzinfo)
            else:
                # Fallback: if no timezone library available, default to UTC
                summary_df['local_time'] = summary_df['datetime_utc']

    # Derive day_of_wk (mon..sun) and weekend (bool) based on local_time
    # Use pandas dt accessors which respect timezone-aware timestamps
    weekday_num = summary_df['local_time'].dt.weekday  # Monday=0, Sunday=6
    dow_map = {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'}
    summary_df['day_of_wk'] = weekday_num.map(dow_map)
    summary_df['weekend'] = weekday_num.isin([5, 6])

    # Get mode of all duty_cycle values, round to whole number
    duty_cycle_mode = summary_df['duty_cycle'].round(2).mode()[0]
    print(f"Duty cycle mode: {duty_cycle_mode}")
    summary_df['duty_cycle_mode'] = duty_cycle_mode
    # Compute proportion of each observed duty cycle to mode, rounded to 2 decimal places
    duty_cycle_prop = (summary_df['duty_cycle'] / duty_cycle_mode).round(2)
    duty_cycle_prop = np.minimum(duty_cycle_prop, 1.0) 
    summary_df['duty_cycle_prop'] = duty_cycle_prop

    return summary_df


def plot_data_availability(summary_df, time_unit='hour', figsize=(15, 4)):
    """
    Plot data availability timeline.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Output from summarize_period()
    time_unit : str
        'hour', 'day', or 'week'
    figsize : tuple
        Figure size
        
    Returns:
    --------
    tuple
        (figure, axis) matplotlib objects
        
    Raises:
    -------
    ValueError
        If time_unit is invalid or required columns are missing
    """
    # Validate input parameters
    valid_time_units = ['hour', 'day', 'week']
    if time_unit not in valid_time_units:
        raise ValueError(f"time_unit must be one of {valid_time_units}")
        
    # Check if DataFrame is empty
    if summary_df is None or summary_df.empty:
        raise ValueError("Input DataFrame is empty")
        
    # Validate required columns
    required_cols = ['datetime_utc', 'duration_min', 'sampling_min', 'subject_id', 'local_time', 'day_of_wk']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if time_unit == 'hour':
        # Build present segments by merging overlapping intervals
        start_ref = summary_df['local_time'].min()
        intervals = []
        for _, row in summary_df.iterrows():
            start_hr = (row['local_time'] - start_ref).total_seconds() / 3600
            end_hr = start_hr + (row['duration_min'] / 60)
            intervals.append((start_hr, end_hr))

        intervals.sort(key=lambda x: x[0])

        present_segments = []
        for seg in intervals:
            if not present_segments:
                present_segments.append([seg[0], seg[1]])
            else:
                last = present_segments[-1]
                if seg[0] <= last[1]:
                    last[1] = max(last[1], seg[1])
                else:
                    present_segments.append([seg[0], seg[1]])

        # Determine full span and compute missing segments
        full_start = 0.0
        full_end = max((e for _, e in present_segments), default=0.0)
        missing_segments = []
        if present_segments:
            if present_segments[0][0] > full_start:
                missing_segments.append([full_start, present_segments[0][0]])
            for i in range(len(present_segments) - 1):
                missing_segments.append([present_segments[i][1], present_segments[i+1][0]])
        # No trailing missing segment because we only plot the observed span

        # Plot horizontal lines: present at y=1, missing at y=0
        for s, e in present_segments:
            ax.hlines(y=1, xmin=s, xmax=e, color='steelblue', linewidth=1.5)
        for s, e in missing_segments:
            ax.hlines(y=0, xmin=s, xmax=e, color='steelblue', linewidth=1.5)

        ax.set_xlim(full_start - 0.5, full_end + 0.5)
        ax.set_ylim(-0.1, 1.1)
        # Build x tick labels from local_time + day_of_wk
        tick_positions = sorted(set(int((lt - summary_df['local_time'].min()).total_seconds() / 3600)
                                    for lt in summary_df['local_time']))
        tick_labels = []
        for pos in tick_positions:
            # Find a representative row at this hour
            ref_time = summary_df['local_time'].min() + pd.Timedelta(hours=pos)
            # Choose closest actual time
            idx = (summary_df['local_time'] - ref_time).abs().idxmin()
            lt = summary_df.loc[idx, 'local_time']
            dow = summary_df.loc[idx, 'day_of_wk']
            tick_labels.append(f"{lt.strftime('%m-%d %H:%M')} ({dow})")

        ax.set_xlabel('Local Time', fontsize=11)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_ylabel('Data Present', fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.grid(axis='x', alpha=0.3)
        
    elif time_unit == 'day':
        # For daily view, show % of day with data
        # Compute daily coverage keyed by local date
        daily_summary = summary_df.copy()
        daily_summary['local_date'] = daily_summary['local_time'].dt.date
        daily_summary = daily_summary.groupby('local_date').agg({
            'sampling_min': 'sum',
            'local_time': 'min',
            'day_of_wk': 'first'
        }).reset_index()
        daily_summary['coverage_pct'] = daily_summary['sampling_min'] / (24 * 60) * 100
        days_since_start = [(lt - daily_summary['local_time'].min()).days 
                           for lt in daily_summary['local_time']]
        ax.bar(days_since_start, daily_summary['coverage_pct'], 
               color='steelblue', alpha=0.7, edgecolor='black', width=0.8)
        # Tick labels as local date + day_of_wk
        ax.set_xlabel('Local Date (with day of week)', fontsize=11)
        print(days_since_start)
        ax.set_xticks(days_since_start)
        ax.set_xticklabels([f"{lt.strftime('%m-%d')} ({dow})" for lt, dow in zip(daily_summary['local_time'], daily_summary['day_of_wk'])],
                           rotation=0)
        ax.set_ylabel('Data Coverage (%)', fontsize=11)
        ax.set_ylim(0, 100)
        
    ax.set_title(f'Data Availability - {summary_df["subject_id"].iloc[0]}', 
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_sampling_time(summary_df, time_unit='hour', figsize=(15, 4)):
    """
    Plot actual sampling time as absolute minutes sampled.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Output from summarize_period()
    time_unit : str
        'hour', 'day', or 'week'
    figsize : tuple
        Figure size
        
    Returns:
    --------
    tuple
        (figure, axis) matplotlib objects
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get expected minutes per hour from duty_cycle_mode
    if 'duty_cycle_mode' in summary_df.columns:
        mode_duty_cycle = summary_df['duty_cycle_mode'].iloc[0]
        # duty_cycle_mode is a proportion, convert to minutes per hour
        expected_minutes_per_hour = int(round(mode_duty_cycle * 60))
    else:
        # Fallback: use mode of duty_cycle column
        mode_duty_cycle = summary_df['duty_cycle'].mode()[0]
        expected_minutes_per_hour = int(round(mode_duty_cycle * 60))
    
    if time_unit == 'hour':
        # For hour view: plot each hour's sampling_min
        # Use local hour of day (0-23) for x-axis positioning
        summary_df['local_hour'] = summary_df['local_time'].dt.hour
        
        # Create bins for all 24 hours, even if data is missing
        all_hours = range(24)
        hour_data = summary_df.groupby('local_hour')['sampling_min'].sum().reindex(all_hours, fill_value=0)
        
        ax.bar(all_hours, hour_data.values, 
               color='coral', alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)
        ax.set_xlim(-0.5, 23.5)
        
        # Build x-axis labels with local_time and day_of_wk
        # For each hour, find a representative local_time and day_of_wk
        tick_labels = []
        for hour in all_hours:
            # Find rows with this hour
            hour_rows = summary_df[summary_df['local_hour'] == hour]
            if len(hour_rows) > 0:
                # Use first occurrence for label
                lt = hour_rows['local_time'].iloc[0]
                dow = hour_rows['day_of_wk'].iloc[0]
                tick_labels.append(f"{lt.strftime('%H:%M')} ({dow})")
            else:
                tick_labels.append(f"{hour:02d}:00")
        
        ax.set_xlabel('Local Time (with day of week)', fontsize=11)
        ax.set_ylabel('Sampling Time (min)', fontsize=11)
        ax.set_xticks(all_hours)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # Expected line based on duty cycle mode
        ax.axhline(y=expected_minutes_per_hour, color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label=f'Expected ({expected_minutes_per_hour} min/hr)')
        
    elif time_unit == 'day':
        # Aggregate by local date
        daily_summary = summary_df.copy()
        daily_summary['local_date'] = daily_summary['local_time'].dt.date
        daily_summary = daily_summary.groupby('local_date').agg({
            'sampling_min': 'sum',
            'local_time': 'min',
            'day_of_wk': 'first'
        }).reset_index()
        
        # X-axis: days since start
        days_since_start = [(lt.date() - daily_summary['local_time'].iloc[0].date()).days 
                           for lt in daily_summary['local_time']]
        
        ax.bar(days_since_start, daily_summary['sampling_min'],
               color='coral', alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)
        
        # Expected line: expected minutes per hour * 24 hours per day
        expected_minutes_per_day = expected_minutes_per_hour * 24
        ax.axhline(y=expected_minutes_per_day, color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label=f'Expected ({expected_minutes_per_day} min/day)')
        
        # X-axis labels
        ax.set_xlabel('Local Date (with day of week)', fontsize=11)
        ax.set_xticks(days_since_start)
        ax.set_xticklabels([f"{lt.strftime('%m-%d')} ({dow})" for lt, dow in zip(daily_summary['local_time'], daily_summary['day_of_wk'])],
                           rotation=45, ha='right')
        ax.set_ylabel('Sampling Time (min)', fontsize=11)
        
    elif time_unit == 'week':
        # Aggregate by week (7-day periods from start)
        summary_df['days_since_start'] = (summary_df['local_time'].dt.date - 
                                         summary_df['local_time'].dt.date.min()).apply(lambda x: x.days)
        summary_df['week'] = summary_df['days_since_start'] // 7
        
        weekly_summary = summary_df.groupby('week').agg({
            'sampling_min': 'sum',
            'local_time': 'min',
            'day_of_wk': 'first'
        }).reset_index()
        
        # X-axis: week numbers
        ax.bar(weekly_summary['week'], weekly_summary['sampling_min'],
               color='coral', alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)
        
        # Expected line: expected minutes per hour * 24 hours * 7 days
        expected_minutes_per_week = expected_minutes_per_hour * 24 * 7
        ax.axhline(y=expected_minutes_per_week, color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label=f'Expected ({expected_minutes_per_week} min/week)')
        
        # X-axis labels
        ax.set_xlabel('Week', fontsize=11)
        ax.set_ylabel('Sampling Time (min)', fontsize=11)
        
    else:
        raise ValueError(f"time_unit must be 'hour', 'day', or 'week', got '{time_unit}'")
    
    ax.set_title(f'Actual Sampling Time - {summary_df["subject_id"].iloc[0]}', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_mean_magnitude(summary_df, time_unit='hour', figsize=(15, 4)):
    """
    Plot mean magnitude with activity classification.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Output from summarize_period()
    time_unit : str
        'hour', 'day', or 'week'
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by std (using your empirical thresholds for now)
    def get_color(std):
        if std < 0.01:
            return 'green'
        elif std < 0.1:
            return 'yellow'
        elif std < 0.3:
            return 'orange'
        else:
            return 'red'
    
    if time_unit == 'hour':
        hours_since_start = [(lt - summary_df['local_time'].min()).total_seconds() / 3600 
                            for lt in summary_df['local_time']]
        
        colors = [get_color(std) for std in summary_df['std_magnitude']]
        
        ax.scatter(hours_since_start, summary_df['mean_magnitude'], 
                  s=50, c=colors, alpha=0.8, edgecolor='black', linewidth=0.5, zorder=3)
        
        # Connect only consecutive hours
        for i in range(len(summary_df) - 1):
            time_gap = (summary_df['datetime_utc'].iloc[i+1] - 
                       summary_df['datetime_utc'].iloc[i]).total_seconds() / 3600
            
            if time_gap <= 1.5:  # Allow small gaps (within 1.5 hours)
                ax.plot([hours_since_start[i], hours_since_start[i+1]], 
                       [summary_df['mean_magnitude'].iloc[i], 
                        summary_df['mean_magnitude'].iloc[i+1]], 
                       color='gray', alpha=0.5, linewidth=1, zorder=2)
        
        # Build tick labels from local_time + day_of_wk
        tick_positions = [int(x) for x in sorted(set(hours_since_start))]
        tick_labels = []
        for pos in tick_positions:
            ref_time = summary_df['local_time'].min() + pd.Timedelta(hours=pos)
            idx = (summary_df['local_time'] - ref_time).abs().idxmin()
            lt = summary_df.loc[idx, 'local_time']
            dow = summary_df.loc[idx, 'day_of_wk']
            tick_labels.append(f"{lt.strftime('%m-%d %H:%M')} ({dow})")
        ax.set_xlabel('Local Time', fontsize=11)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
    elif time_unit == 'day':
        daily_summary = summary_df.copy()
        daily_summary['local_date'] = daily_summary['local_time'].dt.date
        daily_summary = daily_summary.groupby('local_date').agg({
            'mean_magnitude': 'mean',
            'std_magnitude': 'mean',
            'local_time': 'min',
            'day_of_wk': 'first'
        }).reset_index()
        
        days_since_start = [(lt - daily_summary['local_time'].min()).days 
                           for lt in daily_summary['local_time']]
        
        colors = [get_color(std) for std in daily_summary['std_magnitude']]
        
        ax.plot(days_since_start, daily_summary['mean_magnitude'], 
               marker='o', markersize=8, linewidth=2, color='gray', alpha=0.5)
        ax.scatter(days_since_start, daily_summary['mean_magnitude'], 
                  s=100, c=colors, alpha=0.8, edgecolor='black', linewidth=1, zorder=3)
        
        ax.set_xlabel('Local Date (with day of week)', fontsize=11)
        ax.set_xticks(days_since_start)
        ax.set_xticklabels([f"{lt.strftime('%m-%d')} ({dow})" for lt, dow in zip(daily_summary['local_time'], daily_summary['day_of_wk'])])
    
    ax.axhline(y=0.0, color='blue', linestyle='--', linewidth=1, alpha=0.5,
              label='0g baseline')
    ax.set_ylabel('Mean Magnitude (g)', fontsize=11)
    ax.set_title(f'Mean Accelerometer Magnitude - {summary_df["subject_id"].iloc[0]}', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_daily_summary(summary_df, time_unit='hour', figsize=(15, 10)):
    """
    Create all 3 plots together in a single figure.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Output from summarize_period()
    time_unit : str
        'hour', 'day', or 'week'
    figsize : tuple
        Figure size
        
    Returns:
    --------
    tuple
        (figure, list of axes) matplotlib objects
    """
    # Input validation
    valid_time_units = ['hour', 'day', 'week']
    if time_unit not in valid_time_units:
        raise ValueError(f"time_unit must be one of {valid_time_units}")
    
    if summary_df is None or summary_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)
    
    # Plot 1: Data Availability (reuse plotting logic)
    if time_unit == 'hour':
        for _, row in summary_df.iterrows():
            time_diff = row['datetime_utc'] - summary_df['datetime_utc'].min()
            hours_since_start = time_diff.total_seconds() / 3600
            duration_hours = row['duration_min'] / 60
            ax1.fill_between([hours_since_start, hours_since_start + duration_hours],
                           0, 1, alpha=0.7, color='steelblue',
                           edgecolor='black', linewidth=0.3)
        ax1.set_xlabel('Hours Since Start')
        ax1.set_ylabel('Data Present')
    
    # Plot 2: Sampling Time (use same logic as plot_sampling_time)
    if time_unit == 'hour':
        # Get expected minutes per hour from duty_cycle_mode
        if 'duty_cycle_mode' in summary_df.columns:
            mode_duty_cycle = summary_df['duty_cycle_mode'].iloc[0]
            expected_minutes_per_hour = int(round(mode_duty_cycle * 60))
        else:
            mode_duty_cycle = summary_df['duty_cycle'].mode()[0]
            expected_minutes_per_hour = int(round(mode_duty_cycle * 60))
        
        # Use local hour of day (0-23) for x-axis positioning
        summary_df_copy = summary_df.copy()
        summary_df_copy['local_hour'] = summary_df_copy['local_time'].dt.hour
        
        # Create bins for all 24 hours, even if data is missing
        all_hours = range(24)
        hour_data = summary_df_copy.groupby('local_hour')['sampling_min'].sum().reindex(all_hours, fill_value=0)
        
        ax2.bar(all_hours, hour_data.values,
                color='coral', alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)
        ax2.set_xlim(-0.5, 23.5)
        
        # Build x-axis labels
        tick_labels = []
        for hour in all_hours:
            hour_rows = summary_df_copy[summary_df_copy['local_hour'] == hour]
            if len(hour_rows) > 0:
                lt = hour_rows['local_time'].iloc[0]
                dow = hour_rows['day_of_wk'].iloc[0]
                tick_labels.append(f"{lt.strftime('%H:%M')} ({dow})")
            else:
                tick_labels.append(f"{hour:02d}:00")
        
        ax2.set_xlabel('Local Time (with day of week)')
        ax2.set_xticks(all_hours)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax2.set_ylabel('Sampling Time (min)')
        ax2.axhline(y=expected_minutes_per_hour, color='red', linestyle='--', 
                   linewidth=1, alpha=0.5, label=f'Expected ({expected_minutes_per_hour} min/hr)')
        ax2.legend()
        
        # Plot 3: Mean Magnitude (for hour view, reuse same x-axis setup)
        colors = [get_color(std) for std in summary_df['std_magnitude']]
        # Use local hour positions for scatter plot
        summary_df_copy = summary_df.copy()
        summary_df_copy['local_hour'] = summary_df_copy['local_time'].dt.hour
        ax3.scatter(summary_df_copy['local_hour'], summary_df['mean_magnitude'],
                    s=50, c=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.axhline(y=0.0, color='blue', linestyle='--', alpha=0.5,
                    label='0g baseline')
        ax3.set_xlabel('Local Time (with day of week)')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(all_hours)
        ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax3.set_ylabel('Mean Magnitude (g)')
        ax3.legend()
    
    # Set titles
    ax1.set_title('Data Availability Timeline')
    ax2.set_title('Sampling Time')
    ax3.set_title('Mean Accelerometer Magnitude')
    
    # Add grid to all plots
    for ax in [ax1, ax2, ax3]:
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, [ax1, ax2, ax3]

