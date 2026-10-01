"""Tests for clean_downloaded_data module."""

import pandas as pd

from ifunnel.financial_data_preprocessing.clean_downloaded_data import clean_data


def test_clean_data_basic():
    """Test basic cleaning functionality."""
    # Create test data with complete data for all periods
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": [100.0] * 30,
            "Asset2": [200.0] * 30,
        },
        index=dates,
    )

    result = clean_data(data)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Asset1", "Asset2"]
    # Wednesdays Jan 4, 11, 18, 25 give three weekly returns, all zero for constant prices
    assert len(result) == 3
    assert (result == 0).all().all()


def test_clean_data_writes_nothing_by_default(tmp_path, monkeypatch):
    """Test that clean_data writes no file unless an output path is given."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    data = pd.DataFrame({"Asset1": [100.0] * 30}, index=dates)

    clean_data(data)

    assert list(tmp_path.rglob("*")) == [workdir]


def test_clean_data_writes_to_output_path(tmp_path):
    """Test that clean_data writes the returned frame to the given output path."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": [100.0 + i for i in range(30)],
            "Asset2": [200.0] * 30,
        },
        index=dates,
    )
    output_path = tmp_path / "rets.parquet.gzip"

    result = clean_data(data, output_path=output_path)

    pd.testing.assert_frame_equal(pd.read_parquet(output_path), result, check_freq=False)


def test_clean_data_with_missing_values():
    """Test that an asset missing its last values is dropped."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": [100.0] * 20,
            "Asset2": [200.0] * 10 + [""] * 10,  # Missing values
        },
        index=dates,
    )

    result = clean_data(data)

    assert list(result.columns) == ["Asset1"]


def test_clean_data_with_outliers():
    """Test that an asset with a >20% daily return is dropped."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": [100.0] * 10 + [130.0] * 10,  # 30% jump
            "Asset2": [200.0] * 20,
        },
        index=dates,
    )

    result = clean_data(data)

    assert list(result.columns) == ["Asset2"]


def test_clean_data_wednesday_selection():
    """Test that clean_data selects Wednesday prices."""
    # Create data spanning multiple weeks with Wednesdays
    dates = pd.date_range("2023-01-01", periods=21, freq="D")  # 3 weeks
    data = pd.DataFrame(
        {
            "Asset1": list(range(100, 121)),
            "Asset2": list(range(200, 221)),
        },
        index=dates,
    )

    result = clean_data(data)

    assert (pd.DatetimeIndex(result.index).weekday == 2).all()
    # Wednesday Jan 4 -> Jan 11: Asset1 moves from 103 to 110
    assert result.loc[pd.Timestamp("2023-01-11"), "Asset1"] == 110 / 103 - 1


def test_clean_data_incomplete_at_start():
    """Test that an asset missing its first values is dropped."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": ["", "", ""] + [100.0] * 17,  # Missing first 3 values
            "Asset2": [200.0] * 20,
        },
        index=dates,
    )

    result = clean_data(data)

    assert list(result.columns) == ["Asset2"]


def test_clean_data_incomplete_at_end():
    """Test that an asset missing its last 3 values is dropped."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": [100.0] * 17 + ["", "", ""],  # Missing last 3 values
            "Asset2": [200.0] * 20,
        },
        index=dates,
    )

    result = clean_data(data)

    assert list(result.columns) == ["Asset2"]


def test_clean_data_with_missing_price_in_middle():
    """Test cleaning data with missing price in the middle that gets filled from future."""
    # Create a date range starting on a Monday to ensure we have Wednesdays
    dates = pd.date_range("2023-01-02", periods=28, freq="D")  # 4 weeks starting Monday
    # Create data with a missing value on a Wednesday (Jan 11)
    asset1_values = [100.0] * 9 + [110.0] * 19
    asset1_values[9] = ""  # Missing value in middle
    data = pd.DataFrame(
        {
            "Asset1": asset1_values,
            "Asset2": [200.0] * 28,
        },
        index=dates,
    )

    result = clean_data(data)

    # The asset is kept, and its missing Wednesday price is filled from the next day (110)
    assert "Asset1" in result.columns
    assert result.loc[pd.Timestamp("2023-01-11"), "Asset1"] == 110 / 100 - 1


def test_clean_data_with_missing_wednesday():
    """Test cleaning data where a Wednesday is missing and gets filled."""
    # Create data where we skip a Wednesday to trigger the gap-filling logic
    # Start on a Monday
    dates = pd.date_range("2023-01-02", periods=21, freq="D")  # 3 weeks

    # Create the dataframe
    data = pd.DataFrame(
        {
            "Asset1": [100.0 + i for i in range(21)],
            "Asset2": [200.0 + i for i in range(21)],
        },
        index=dates,
    )

    # Remove a Wednesday (index 2 is Wednesday Jan 4, index 9 is Wednesday Jan 11)
    # We'll drop the middle Wednesday
    data = data.drop(dates[9])  # Remove Wednesday Jan 11

    result = clean_data(data)

    # The missing Wednesday is filled with the price from 5 days earlier (Jan 6: 104)
    assert pd.Timestamp("2023-01-11") in result.index
    assert result.loc[pd.Timestamp("2023-01-11"), "Asset1"] == 104 / 102 - 1
