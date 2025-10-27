import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Helper functions used to read files, compute summaries,
# concatenate summaries, and plot 


def summarize_hourly_file(file_path, verbose=True):
    """Analyze a single hourly accelerometer CSV file."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            if verbose:
                print(f"Missing: {os.path.basename(file_path)}")
            return None
        # If it exists read CSV
        df = pd.read_csv(file_path)

        # Handle empty files
        if len(df) == 0:
            if verbose:
                print(f"Empty: {os.path.basename(file_path)}")
            return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Convert timestamps
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    # Compute magnitude
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

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


def summarize_period(subject_id, start_date, end_date, base_path):
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
                )
                
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
                print(f"  ✓ Hour {hour:02d}: {result['n_rows']:,} rows")
            else:
                print(f"  ⊘ Hour {hour:02d}: missing")
    
    print("\n" + "=" * 70)
    print(f"Processed {len(all_summaries)} hours across "
          f"{len(date_range)} days")

    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)

    # Sort by datetime
    summary_df = summary_df.sort_values('datetime_utc').reset_index(drop=True)

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
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if time_unit == 'hour':
        # For hourly view, show fractional coverage within each hour
        for _, row in summary_df.iterrows():
            # Calculate x position (hours since start)
            time_diff = row['datetime_utc'] - summary_df['datetime_utc'].min()
            hours_since_start = time_diff.total_seconds() / 3600
            duration_hours = row['duration_min'] / 60

            ax.fill_between([hours_since_start, hours_since_start + duration_hours],
                           0, 1, alpha=0.7, color='steelblue',
                           edgecolor='black', linewidth=0.3)

        ax.set_xlabel('Hours Since Start', fontsize=11)
        ax.set_ylabel('Data Present', fontsize=11)
        
    elif time_unit == 'day':
        # For daily view, show % of day with data
        daily_summary = summary_df.groupby('date').agg({
            'sampling_min': 'sum',
            'datetime_utc': 'min'
        }).reset_index()
        
        daily_summary['coverage_pct'] = daily_summary['sampling_min'] / (24 * 60) * 100
        
        days_since_start = [(dt - daily_summary['datetime_utc'].min()).days 
                           for dt in daily_summary['datetime_utc']]
        
        ax.bar(days_since_start, daily_summary['coverage_pct'], 
               color='steelblue', alpha=0.7, edgecolor='black', width=0.8)
        ax.set_xlabel('Days Since Start', fontsize=11)
        ax.set_ylabel('Data Coverage (%)', fontsize=11)
        ax.set_ylim(0, 100)
        
    ax.set_title(f'Data Availability - {summary_df["subject_id"].iloc[0]}', 
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_sampling_time(summary_df, time_unit='hour', figsize=(15, 4)):
    """
    Plot actual sampling time.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Output from summarize_period()
    time_unit : str
        'hour', 'day', or 'week'
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if time_unit == 'hour':
        hours_since_start = [(dt - summary_df['datetime_utc'].min()).total_seconds() / 3600 
                            for dt in summary_df['datetime_utc']]
        
        ax.bar(hours_since_start, summary_df['sampling_min'], 
               color='coral', alpha=0.7, edgecolor='black', width=0.8)
        ax.axhline(y=30, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label='Expected (50% duty cycle)')
        ax.set_xlabel('Hours Since Start', fontsize=11)
        ax.set_ylabel('Sampling Time (min)', fontsize=11)
        
    elif time_unit == 'day':
        daily_summary = summary_df.groupby('date').agg({
            'sampling_min': 'sum',
            'datetime_utc': 'min'
        }).reset_index()
        
        days_since_start = [(dt - daily_summary['datetime_utc'].min()).days 
                           for dt in daily_summary['datetime_utc']]
        
        ax.bar(days_since_start, daily_summary['sampling_min'] / 60,  # Convert to hours
               color='coral', alapha=0.7, edgecolor='black', width=0.8)
        ax.axhline(y=12, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label='Expected (50% duty cycle)')
        ax.set_xlabel('Days Since Start', fontsize=11)
        ax.set_ylabel('Sampling Time (hours)', fontsize=11)
    
    ax.set_title(f'Actual Sampling Time - {summary_df["subject_id"].iloc[0]}', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
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
        hours_since_start = [(dt - summary_df['datetime_utc'].min()).total_seconds() / 3600 
                            for dt in summary_df['datetime_utc']]
        
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
        
        ax.set_xlabel('Hours Since Start', fontsize=11)
        
    elif time_unit == 'day':
        daily_summary = summary_df.groupby('date').agg({
            'mean_magnitude': 'mean',
            'std_magnitude': 'mean',
            'datetime_utc': 'min'
        }).reset_index()
        
        days_since_start = [(dt - daily_summary['datetime_utc'].min()).days 
                           for dt in daily_summary['datetime_utc']]
        
        colors = [get_color(std) for std in daily_summary['std_magnitude']]
        
        ax.plot(days_since_start, daily_summary['mean_magnitude'], 
               marker='o', markersize=8, linewidth=2, color='gray', alpha=0.5)
        ax.scatter(days_since_start, daily_summary['mean_magnitude'], 
                  s=100, c=colors, alpha=0.8, edgecolor='black', linewidth=1, zorder=3)
        
        ax.set_xlabel('Days Since Start', fontsize=11)
    
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=1, alpha=0.5,
              label='1g baseline')
    ax.set_ylabel('Mean Magnitude (g)', fontsize=11)
    ax.set_title(f'Mean Accelerometer Magnitude - {summary_df["subject_id"].iloc[0]}', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_daily_summary(summary_df, time_unit='hour', figsize=(15, 10)):
    """
    Create all 3 plots together.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Output from summarize_period()
    time_unit : str
        'hour', 'day', or 'week'
    """
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot 1: Data Availability
    plot_data_availability(summary_df, time_unit=time_unit, figsize=figsize)
    plt.close()  # Close individual plot
    
    # Plot 2: Sampling Time
    plot_sampling_time(summary_df, time_unit=time_unit, figsize=figsize)
    plt.close()
    
    # Plot 3: Mean Magnitude
    plot_mean_magnitude(summary_df, time_unit=time_unit, figsize=figsize)
    plt.close()
    
    # Recreate all in one figure (you'll refine this)
    # For now, just show message
    print("Use individual plot functions or combine manually")
    
    return None

