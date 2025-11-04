import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Import helper directly from src to avoid package side-effects
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))
from helper import summarize_period  # noqa: E402


def _write_hour_csv(base_dir: Path, subject: str, date_str: str, hour: int):
    accel_dir = base_dir / subject / 'accelerometer'
    accel_dir.mkdir(parents=True, exist_ok=True)

    # Generate 10 samples at 10 Hz starting exactly at the hour
    start_ts = pd.Timestamp(f"{date_str} {hour:02d}:00:00", tz='UTC')
    timestamps_ms = (start_ts.value // 10**6) + np.arange(0, 1000, 100)  # 0..900 ms
    x = np.ones_like(timestamps_ms) * 0.1
    y = np.ones_like(timestamps_ms) * 0.2
    z = np.ones_like(timestamps_ms) * 0.97

    df = pd.DataFrame({
        'timestamp': timestamps_ms,
        'x': x,
        'y': y,
        'z': z,
    })

    file_name = f"{date_str} {hour:02d}_00_00+00_00.csv"
    df.to_csv(accel_dir / file_name, index=False)


def test_local_time_us_east_alias():
    subject = 'testsubj'
    date_str = '2021-01-01'
    hour = 13  # 13:00 UTC -> 08:00 America/New_York (EST)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        _write_hour_csv(base_path, subject, date_str, hour)

        df = summarize_period(subject, date_str, date_str, str(base_path), timezone='us_east')

        assert 'local_time' in df.columns
        assert len(df) == 1

        utc_time = pd.Timestamp(f"{date_str} {hour:02d}:00:00", tz='UTC')
        expected_local = utc_time.tz_convert('America/New_York')
        assert df['datetime_utc'].iloc[0] == utc_time
        assert df['local_time'].iloc[0] == expected_local


def test_local_time_iana_name():
    subject = 'testsubj2'
    date_str = '2021-07-01'
    hour = 15  # 15:00 UTC -> 11:00 America/New_York (EDT)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        _write_hour_csv(base_path, subject, date_str, hour)

        df = summarize_period(subject, date_str, date_str, str(base_path), timezone='America/New_York')

        utc_time = pd.Timestamp(f"{date_str} {hour:02d}:00:00", tz='UTC')
        expected_local = utc_time.tz_convert('America/New_York')
        assert df['local_time'].iloc[0] == expected_local


def test_local_time_none_defaults_to_utc():
    subject = 'testsubj3'
    date_str = '2022-03-10'
    hour = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        _write_hour_csv(base_path, subject, date_str, hour)

        df = summarize_period(subject, date_str, date_str, str(base_path))

        assert 'local_time' in df.columns
        assert (df['local_time'] == df['datetime_utc']).all()


def test_day_of_wk_and_weekend_from_local_time_alias():
    subject = 'testsubj4'
    # 2021-01-02 is a Saturday; pick 03:00 UTC -> 22:00 previous day US/Eastern (Friday)
    date_str = '2021-01-02'
    hour = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        _write_hour_csv(base_path, subject, date_str, hour)

        df = summarize_period(subject, date_str, date_str, str(base_path), timezone='us_east')

        # local time should be previous day Friday 22:00, so weekday = 4 (fri)
        assert df['day_of_wk'].iloc[0] == 'fri'
        assert bool(df['weekend'].iloc[0]) is False


def test_day_of_wk_and_weekend_true():
    subject = 'testsubj5'
    # 2021-01-03 is Sunday; 12:00 UTC -> 07:00 US/Eastern (still Sunday)
    date_str = '2021-01-03'
    hour = 12

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        _write_hour_csv(base_path, subject, date_str, hour)

        df = summarize_period(subject, date_str, date_str, str(base_path), timezone='America/New_York')

        assert df['day_of_wk'].iloc[0] == 'sun'
        assert bool(df['weekend'].iloc[0]) is True


