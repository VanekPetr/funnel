"""Module for testing data analyser functionality."""

import pandas as pd
import pytest

from ifunnel.models.data_analyser import final_stats, get_weekly_returns, mean_an_returns


def test_final_stats_with_valid_input() -> None:
    """Test final_stats with valid input data."""
    data = pd.DataFrame(
        {
            "Asset1": [100, 102, 104, 108],
            "Asset2": [200, 205, 210, 215],
        }
    )
    result = final_stats(data)
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    assert list(result.columns) == ["Avg An Ret", "Std Dev of Ret", "Sharpe R"], "Result should have proper columns"
    assert len(result) == 2, "Result should contain statistics for all assets"
    assert all(result.notnull().all()), "All statistics values should be non-null"


# def test_final_stats_empty_dataframe() -> None:
#     """Test final_stats with an empty DataFrame."""
#     data = pd.DataFrame()
#     with pytest.raises(IndexError):
#         final_stats(data)


def test_final_stats_single_column() -> None:
    """Test final_stats with single-column DataFrame."""
    data = pd.DataFrame({"Asset1": [100, 102, 105, 108]})
    result = final_stats(data)
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(result) == 1, "Result should contain statistics for the single asset"
    assert all(result.notnull().all()), "All values in the result should be non-null"
    assert list(result.columns) == ["Avg An Ret", "Std Dev of Ret", "Sharpe R"], "Result should have proper columns"


def test_final_stats_with_non_numeric_data() -> None:
    """Test final_stats with non-numeric data."""
    data = pd.DataFrame(
        {
            "Asset1": [100, 102, "a", 108],
            "Asset2": [200, "b", 210, 215],
        }
    )
    with pytest.raises(TypeError):
        final_stats(data)


def test_final_stats_with_identical_prices() -> None:
    """Test final_stats when all prices are identical."""
    data = pd.DataFrame(
        {
            "Asset1": [100, 100, 100, 100],
            "Asset2": [200, 200, 200, 200],
        }
    )
    result = final_stats(data)
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(result) == 2, "Result should contain statistics for both assets"
    assert all(result["Std Dev of Ret"] == 0), "Standard deviation should be zero for identical prices"
    assert all(result["Sharpe R"].isna()), "Sharpe ratio should be NaN when standard deviation is zero"


# def test_mean_an_returns_single_column() -> None:
#     """Test mean_an_returns with a single column DataFrame."""
#     data = pd.DataFrame({'Asset1': [0.02, 0.03, -0.01, 0.04]})
#     result = mean_an_returns(data)
#     assert isinstance(result, float), "Result should be a float"
#     assert -1 <= result <= 1, "Result should be within a valid returns range"


# def test_mean_an_returns_multi_column() -> None:
#     """Test mean_an_returns with a multi-column DataFrame."""
#     data = pd.DataFrame({
#         'Asset1': [0.02, 0.03, -0.01, 0.04],
#         'Asset2': [0.01, -0.02, 0.05, 0.03],
#     })
#     result = mean_an_returns(data)
#     assert isinstance(result, pd.Series), "Result should be a pandas Series"
#     assert all(-1 <= val <= 1 for val in result), "Each value in the result should be within a valid returns range"


# def test_mean_an_returns_empty_dataframe() -> None:
#     """Test mean_an_returns with an empty DataFrame."""
#     data = pd.DataFrame()
#     with pytest.raises(IndexError):
#         mean_an_returns(data)


def test_mean_an_returns_non_numeric_data() -> None:
    """Test mean_an_returns with non-numeric values in the DataFrame."""
    data = pd.DataFrame(
        {
            "Asset1": [0.02, 0.03, -0.01, 0.04],
            "Asset2": ["a", "b", "c", "d"],
        }
    )
    with pytest.raises(TypeError):
        mean_an_returns(data)


# def test_mean_an_returns_single_value_per_column() -> None:
#     """Test mean_an_returns with a single value per column."""
#     data = pd.DataFrame({
#         'Asset1': [0.05],
#         'Asset2': [0.02],
#     })
#     result = mean_an_returns(data)
#     assert isinstance(result, pd.Series), "Result should be a pandas Series"
#     assert all(-1 <= val <= 1 for val in result), "Each value in the result should be within a valid returns range"


def test_get_weekly_returns_basic() -> None:
    """Test get_weekly_returns with basic weekly data."""
    # Create a date range with some Wednesdays
    dates = pd.date_range(start="2023-01-04", periods=10, freq="W-WED")  # Wednesdays
    data = pd.DataFrame(
        {
            "Asset1": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            "Asset2": [200, 202, 204, 206, 208, 210, 212, 214, 216, 218],
        },
        index=dates,
    )
    result = get_weekly_returns(data)
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(result) == 9, "Result should have one less row (first row dropped)"
    assert all(result.notnull().all()), "All values should be non-null"


def test_get_weekly_returns_mixed_weekdays() -> None:
    """Test get_weekly_returns with mixed weekdays."""
    # Create data with different days of the week
    dates = pd.date_range(start="2023-01-02", periods=15, freq="D")  # Daily data
    data = pd.DataFrame(
        {"Asset1": range(100, 115), "Asset2": range(200, 215)},
        index=dates,
    )
    result = get_weekly_returns(data)
    # Should only select Wednesdays
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    assert all(date.weekday() == 2 for date in result.index), "All dates should be Wednesdays"
