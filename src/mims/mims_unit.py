"""
Monitor Independent Movement Summary (MIMS) implementation in Python.

This module ports the core logic of the MIMS-unit algorithm described in
Zhang et al. (2012) and implemented in the R package `MIMSunit`
[`mims_unit.R`](https://github.com/mHealthGroup/MIMSunit/blob/master/R/mims_unit.R).

The public entry point is :func:`mims_unit`, which mirrors the high-level R API.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas import Timedelta
from scipy import signal
from scipy.interpolate import CubicSpline


DEFAULT_DYNAMIC_RANGE: Tuple[float, float] = (-8.0, 8.0)
DEFAULT_EPOCH: str = "5 sec"
TARGET_SAMPLING_RATE_HZ: int = 100
RECTIFY_THRESHOLD: float = -150.0
TRUNCATION_FACTOR: float = 1e-4
AXIS_COLUMNS: Tuple[str, str, str] = ("x", "y", "z")


class MIMSComputationError(RuntimeError):
    """Raised when MIMS-unit computation encounters irrecoverable issues."""


def mims_unit(
    df: pd.DataFrame,
    epoch: str = DEFAULT_EPOCH,
    dynamic_range: Tuple[float, float] = DEFAULT_DYNAMIC_RANGE,
    output_mims_per_axis: bool = False,
    *,
    before_df: Optional[pd.DataFrame] = None,
    after_df: Optional[pd.DataFrame] = None,
    use_filtering: bool = True,
) -> pd.DataFrame:
    """
    Compute Monitor Independent Movement Summary (MIMS) units.

    Parameters
    ----------
    df:
        Input accelerometer samples. The first column must be a timestamp
        (timezone-aware or naive) and the remaining columns must be the
        acceleration axes expressed in units of ``g``.
    epoch:
        Epoch length accepted by :func:`pandas.to_timedelta`, e.g. ``"1 sec"``,
        ``"5 sec"``, or ``"1 min"``.
    dynamic_range:
        Tuple ``(low, high)`` describing the sensor dynamic range in ``g``.
    output_mims_per_axis:
        Whether to append axial MIMS values to the output.
    before_df / after_df:
        Optional samples to prepend/append to ``df`` to reduce filter edge
        effects. They must conform to the same schema as ``df``.
    use_filtering:
        Apply the 0.2–5 Hz band-pass Butterworth cascade used by the reference
        implementation before temporal aggregation.

    Returns
    -------
    pandas.DataFrame
        A dataframe with epoch start timestamps and the aggregated MIMS-unit
        value. Axial values are included when ``output_mims_per_axis`` is true.

    Notes
    -----
    The reference implementation performs sophisticated extrapolation to
    reconstruct saturated samples. The current Python port performs spline
    interpolation to 100 Hz and clips to the declared dynamic range. This
    covers the non-saturated regime which is the common case in free-living
    accelerometer recordings. The scaffolding has been designed to extend the
    extrapolation stage when parity with the R implementation is required.
    """

    if len(dynamic_range) != 2:
        raise MIMSComputationError("Dynamic range must be a two-element tuple.")
    if dynamic_range[0] >= dynamic_range[1]:
        raise MIMSComputationError("Dynamic range lower bound must be less than the upper bound.")

    df = _prepare_input(df)
    frames: list[pd.DataFrame] = [df]
    if before_df is not None:
        frames.insert(0, _prepare_input(before_df))
    if after_df is not None:
        frames.append(_prepare_input(after_df))

    work_df = pd.concat(frames, ignore_index=True)
    work_df.sort_values("timestamp", inplace=True)
    _assert_unique_timestamps(work_df)

    start_time = work_df["timestamp"].iloc[0].floor("s")
    stop_time = work_df["timestamp"].iloc[-1].floor("s")

    resampled = _resample_to_target_rate(work_df, dynamic_range)
    sr = _sampling_rate(resampled)

    abnormal_mask = (resampled[list(AXIS_COLUMNS)] < RECTIFY_THRESHOLD).any(axis=1)
    normal = resampled.loc[~abnormal_mask].copy()

    if use_filtering:
        filtered = _bandpass_filter(normal, sr=sr, low=0.2, high=5.0, order=4)
    else:
        filtered = normal

    if abnormal_mask.any():
        abnormal = resampled.loc[abnormal_mask, ["timestamp", *AXIS_COLUMNS]].copy()
        rename_map = {
            axis: f"IIR_{axis.upper()}" for axis in AXIS_COLUMNS if f"IIR_{axis.upper()}" in filtered.columns
        }
        abnormal.rename(columns=rename_map, inplace=True)
        filtered = pd.concat([filtered, abnormal], ignore_index=True)

    filtered.sort_values("timestamp", inplace=True)
    integrated = _aggregate_for_mims(filtered, epoch=epoch, sr=sr)

    aggregated_axes = [f"AGGREGATED_{axis}" for axis in AXIS_COLUMNS]

    summed = _sum_axes(integrated, aggregated_axes)
    if output_mims_per_axis:
        per_axis = integrated[aggregated_axes].copy()
        per_axis.columns = [f"MIMS_UNIT_{axis.upper()}" for axis in AXIS_COLUMNS]
        mims_df = pd.concat([summed, per_axis], axis=1)
    else:
        mims_df = summed

    mims_df.rename(columns={"SUM_MIMS": "MIMS_UNIT"}, inplace=True)

    abnormal_epochs = (integrated[aggregated_axes] < 0).any(axis=1)
    target_cols = ["MIMS_UNIT"]
    if output_mims_per_axis:
        target_cols.extend([f"MIMS_UNIT_{axis.upper()}" for axis in AXIS_COLUMNS])
    mims_df.loc[abnormal_epochs.values, target_cols] = -0.01

    keep = (mims_df["timestamp"] >= start_time) & (mims_df["timestamp"] < stop_time)
    mims_df = mims_df.loc[keep].reset_index(drop=True)
    return mims_df


def _prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 4:
        raise MIMSComputationError(
            "Input data must contain a timestamp column followed by X/Y/Z axes."
        )

    ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    if ts.isna().any():
        raise MIMSComputationError("Timestamp column contains invalid entries.")

    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    axis_cols = df.columns[1:4]
    renamed = df.copy()
    renamed.columns = ["timestamp", *axis_cols, *renamed.columns[4:]]
    renamed = renamed[["timestamp", *axis_cols]].copy()
    renamed["timestamp"] = ts
    numeric = renamed[axis_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise MIMSComputationError("Axis values must be numeric.")
    numeric.columns = AXIS_COLUMNS
    prepared = pd.concat([renamed[["timestamp"]], numeric], axis=1)
    prepared.sort_values("timestamp", inplace=True)
    return prepared.reset_index(drop=True)


def _assert_unique_timestamps(df: pd.DataFrame) -> None:
    if df["timestamp"].duplicated().any():
        raise MIMSComputationError("Input data contains duplicated timestamps.")


def _resample_to_target_rate(
    df: pd.DataFrame,
    dynamic_range: Tuple[float, float],
    target_sr: int = TARGET_SAMPLING_RATE_HZ,
) -> pd.DataFrame:
    """
    Resample the signal to the target sampling rate (100 Hz by default).

    The reference implementation performs extrapolation; the current version
    uses cubic spline interpolation and clips to the declared dynamic range.
    """

    ts_seconds = df["timestamp"].astype("int64") / 1e9
    t_start = ts_seconds.iloc[0]
    t_end = ts_seconds.iloc[-1]
    target_times = np.arange(t_start, t_end + 1e-9, 1.0 / target_sr)

    resampled = {"timestamp": pd.to_datetime(target_times, unit="s", utc=True)}
    low, high = dynamic_range

    for axis in AXIS_COLUMNS:
        spline = CubicSpline(ts_seconds.to_numpy(), df[axis].to_numpy(), bc_type="natural", extrapolate=True)
        values = spline(target_times)
        values = np.clip(values, low, high)
        resampled[axis] = values

    return pd.DataFrame(resampled)


def _sampling_rate(df: pd.DataFrame) -> float:
    ts = df["timestamp"]
    duration = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    if duration == 0:
        return TARGET_SAMPLING_RATE_HZ
    sr = len(df) / duration
    return float(np.round(sr / 10.0) * 10.0)


def _bandpass_filter(
    df: pd.DataFrame,
    *,
    sr: float,
    low: float,
    high: float,
    order: int = 4,
) -> pd.DataFrame:
    nyquist = 0.5 * sr
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass")
    filtered = pd.DataFrame({"timestamp": df["timestamp"]})
    for axis in AXIS_COLUMNS:
        filtered[f"IIR_{axis.upper()}"] = signal.lfilter(b, a, df[axis].to_numpy())
    return filtered


def _aggregate_for_mims(df: pd.DataFrame, epoch: str, sr: float) -> pd.DataFrame:
    epoch_td = _parse_epoch(epoch)
    expected_samples = int(np.round(epoch_td.total_seconds() * sr))
    if expected_samples <= 0:
        raise MIMSComputationError(f"Epoch '{epoch}' resolves to zero samples at {sr} Hz.")

    segment_index = _segment_indices(df["timestamp"], epoch_td)
    df = df.copy()
    df["_segment"] = segment_index

    rows = []
    epoch_origin = _floor_to_epoch(df["timestamp"].iloc[0], epoch)
    for seg_id, segment in df.groupby("_segment", sort=True):
        timestamp = epoch_origin + seg_id * epoch_td

        segment = segment.sort_values("timestamp")
        if len(segment) < 0.9 * expected_samples:
            rows.append([timestamp, *([-1.0] * len(AXIS_COLUMNS))])
            continue

        time_offset = (segment["timestamp"] - segment["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
        values = segment.iloc[:, 1:4].to_numpy()
        values = np.where(values > RECTIFY_THRESHOLD, np.abs(values), values)

        if (values < 0).any():
            rows.append([timestamp, *([-1.0] * len(AXIS_COLUMNS))])
            continue

        auc = np.trapezoid(values, time_offset, axis=0)
        max_values = 16.0 * expected_samples
        auc = np.where(auc >= max_values, -1.0, auc)
        rows.append([timestamp, *auc])

    aggregated = pd.DataFrame(
        rows,
        columns=["timestamp", *(f"AGGREGATED_{axis}" for axis in AXIS_COLUMNS)],
    )
    aggregated["timestamp"] = pd.to_datetime(aggregated["timestamp"], utc=True)

    axis_columns = [f"AGGREGATED_{axis}" for axis in AXIS_COLUMNS]
    mask = (aggregated[axis_columns] > 0) & (
        aggregated[axis_columns] <= (TRUNCATION_FACTOR * expected_samples)
    )
    aggregated.loc[:, axis_columns] = aggregated.loc[:, axis_columns].mask(mask, 0.0)
    return aggregated


def _sum_axes(df: pd.DataFrame, axis_columns: Sequence[str]) -> pd.DataFrame:
    summed = df[["timestamp"] + list(axis_columns)].copy()
    summed["SUM_MIMS"] = summed[list(axis_columns)].sum(axis=1)
    return summed[["timestamp", "SUM_MIMS"]]


def _parse_epoch(epoch: str) -> Timedelta:
    try:
        td = pd.to_timedelta(epoch)
    except ValueError as exc:
        raise MIMSComputationError(f"Unsupported epoch value '{epoch}'.") from exc
    if td <= Timedelta(0):
        raise MIMSComputationError("Epoch length must be positive.")
    return td


def _segment_indices(timestamps: pd.Series, epoch_td: Timedelta) -> np.ndarray:
    start = timestamps.iloc[0]
    offsets = (timestamps - start).dt.total_seconds().to_numpy()
    return np.floor_divide(offsets, epoch_td.total_seconds()).astype(int)


def _floor_to_epoch(ts: pd.Timestamp, epoch: str) -> pd.Timestamp:
    epoch_lower = epoch.lower()
    if "sec" in epoch_lower:
        return ts.floor("s")
    if "min" in epoch_lower:
        return ts.floor("min")
    if "hour" in epoch_lower:
        return ts.floor("H")
    if "day" in epoch_lower:
        return ts.floor("D")
    return ts

