"""Tests for Yahoo Finance data download module."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from ifunnel.financial_data_preprocessing.get_yahoo_data import download_data


def test_download_data_success():
    """Test successful data download."""
    with patch("ifunnel.financial_data_preprocessing.get_yahoo_data.yf.download") as mock_download:
        # Mock the response
        mock_df = pd.DataFrame(
            {"AAPL": [100, 101, 102], "MSFT": [200, 201, 202]},
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_download.return_value = {"Adj Close": mock_df}
        
        result = download_data("2023-01-01", "2023-01-03", ["AAPL", "MSFT"])
        
        assert result is not None
        mock_download.assert_called_once()


def test_download_data_exception():
    """Test download_data handles exceptions gracefully."""
    with patch("ifunnel.financial_data_preprocessing.get_yahoo_data.yf.download") as mock_download:
        mock_download.side_effect = Exception("Network error")
        
        result = download_data("2023-01-01", "2023-01-03", ["AAPL"])
        
        assert result is None


def test_download_data_with_empty_ticker_list():
    """Test download_data with empty ticker list."""
    with patch("ifunnel.financial_data_preprocessing.get_yahoo_data.yf.download") as mock_download:
        result = download_data("2023-01-01", "2023-01-03", [])
        mock_download.assert_called_once_with([], start="2023-01-01", end="2023-01-03")
