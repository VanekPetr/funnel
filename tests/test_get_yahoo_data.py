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


def test_download_data_main_block():
    """Test the main block functionality (indirectly through imports)."""
    # The __main__ block is only executed when run as a script
    # We can't directly test it, but we can verify the function works
    with patch("ifunnel.financial_data_preprocessing.get_yahoo_data.yf.download") as mock_download, \
         patch("ifunnel.financial_data_preprocessing.get_yahoo_data.pd.read_excel") as mock_read_excel, \
         patch("ifunnel.financial_data_preprocessing.get_yahoo_data.os.path.join") as mock_join, \
         patch("ifunnel.financial_data_preprocessing.get_yahoo_data.os.path.dirname") as mock_dirname, \
         patch("ifunnel.financial_data_preprocessing.get_yahoo_data.os.getcwd") as mock_getcwd:
        
        # Mock the required components
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.xlsx"
        
        # This tests that the function can be called successfully
        result = download_data("2023-01-01", "2023-01-03", ["AAPL"])
        mock_download.assert_called_once()
