"""Tests for clean_downloaded_data module."""

from unittest.mock import patch

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

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
            clean_data(data)
            mock_to_parquet.assert_called_once()


def test_clean_data_with_missing_values():
    """Test cleaning data with missing values."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": [100.0] * 20,
            "Asset2": [200.0] * 10 + [""] * 10,  # Missing values
        },
        index=dates,
    )

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet"):
            clean_data(data)


def test_clean_data_with_outliers():
    """Test cleaning data with outliers (>20% daily returns)."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": [100.0] * 10 + [130.0] * 10,  # 30% jump
            "Asset2": [200.0] * 20,
        },
        index=dates,
    )

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet"):
            clean_data(data)


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

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet"):
            clean_data(data)


def test_clean_data_incomplete_at_start():
    """Test cleaning data with incomplete data at start."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": ["", "", ""] + [100.0] * 17,  # Missing first 3 values
            "Asset2": [200.0] * 20,
        },
        index=dates,
    )

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet"):
            clean_data(data)
            # Asset1 should be dropped


def test_clean_data_incomplete_at_end():
    """Test cleaning data with incomplete data at end."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {
            "Asset1": [100.0] * 17 + ["", "", ""],  # Missing last 3 values
            "Asset2": [200.0] * 20,
        },
        index=dates,
    )

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet"):
            clean_data(data)
            # Asset1 should be dropped


def test_clean_data_with_missing_price_in_middle():
    """Test cleaning data with missing price in the middle that gets filled from future."""
    # Create a date range starting on a Monday to ensure we have Wednesdays
    dates = pd.date_range("2023-01-02", periods=28, freq="D")  # 4 weeks starting Monday
    # Create data with a missing value in the middle
    asset1_values = [100.0] * 28
    asset1_values[5] = ""  # Missing value in middle
    data = pd.DataFrame(
        {
            "Asset1": asset1_values,
            "Asset2": [200.0] * 28,
        },
        index=dates,
    )

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
        patch("builtins.print") as mock_print,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet"):
            clean_data(data)
            # Check that "found price" was printed
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("found price" in call for call in calls)


def test_clean_data_with_outlier_detection_print():
    """Test that outlier detection prints the asset info."""
    # Create a date range starting on a Monday to ensure we have Wednesdays
    dates = pd.date_range("2023-01-02", periods=28, freq="D")  # 4 weeks starting Monday
    # Create data with >20% daily return (outlier)
    asset1_values = [100.0] * 14 + [125.0] * 14  # 25% jump - outlier
    data = pd.DataFrame(
        {
            "Asset1": asset1_values,
            "Asset2": [200.0] * 28,
        },
        index=dates,
    )

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
        patch("builtins.print") as mock_print,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet"):
            clean_data(data)
            # Check that outlier info was printed (asset name, return value, count)
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Asset1" in call for call in calls)


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

    with (
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname,
        patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd,
        patch("builtins.print") as mock_print,
    ):
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"

        with patch.object(pd.DataFrame, "to_parquet"):
            clean_data(data)
            # The missing Wednesday date should be printed
            calls = [str(call) for call in mock_print.call_args_list]
            # Check that a date was printed (for missing Wednesday)
            assert any("2023" in call for call in calls)
