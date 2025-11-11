import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mims import MIMSComputationError, mims_unit  # noqa: E402


def _constant_signal(seconds: int = 2, sr: int = 20) -> pd.DataFrame:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    timestamps = start + pd.to_timedelta(np.arange(seconds * sr) / sr, unit="s")
    data = {
        "timestamp": timestamps,
        "x": np.full(len(timestamps), 0.1, dtype=float),
        "y": np.full(len(timestamps), 0.2, dtype=float),
        "z": np.full(len(timestamps), 0.3, dtype=float),
    }
    return pd.DataFrame(data)


def test_mims_unit_per_axis_sum_matches_components():
    df = _constant_signal()
    result = mims_unit(
        df,
        epoch="1 sec",
        dynamic_range=(-2.0, 2.0),
        output_mims_per_axis=True,
        use_filtering=False,
    )

    assert list(result.columns) == [
        "timestamp",
        "MIMS_UNIT",
        "MIMS_UNIT_X",
        "MIMS_UNIT_Y",
        "MIMS_UNIT_Z",
    ]
    assert len(result) == 1

    total_expected = 0.1 + 0.2 + 0.3
    np.testing.assert_allclose(result["MIMS_UNIT_X"], 0.1, rtol=1e-3)
    np.testing.assert_allclose(result["MIMS_UNIT_Y"], 0.2, rtol=1e-3)
    np.testing.assert_allclose(result["MIMS_UNIT_Z"], 0.3, rtol=1e-3)
    np.testing.assert_allclose(result["MIMS_UNIT"], total_expected, rtol=1e-3)


def test_mims_unit_handles_prefiltering_flag():
    df = _constant_signal()
    result = mims_unit(df, epoch="1 sec", dynamic_range=(-2.0, 2.0))
    assert "MIMS_UNIT" in result.columns
    assert len(result) == 1


def test_mims_unit_rejects_duplicate_timestamps():
    df = _constant_signal().copy()
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    with pytest.raises(MIMSComputationError, match="duplicated timestamps"):
        mims_unit(df, dynamic_range=(-2.0, 2.0))


def test_mims_unit_requires_valid_dynamic_range():
    df = _constant_signal()
    with pytest.raises(MIMSComputationError, match="Dynamic range lower bound"):
        mims_unit(df, dynamic_range=(1.0, -1.0))


def test_mims_unit_handles_fixture_csv():
    csv_path = PROJECT_ROOT / "tests" / "data" / "synthetic_acc.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    result = mims_unit(df, epoch="1 sec", dynamic_range=(-8.0, 8.0), use_filtering=False)
    assert not result.empty
    assert (result["timestamp"].diff().dropna() >= pd.Timedelta(seconds=1)).all()

