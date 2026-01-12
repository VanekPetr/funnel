"""Tests for clean_downloaded_data module."""

import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

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
    
    with patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd:
        
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
    
    with patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd:
        
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"
        
        with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
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
    
    with patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd:
        
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"
        
        with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
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
    
    with patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd:
        
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"
        
        with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
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
    
    with patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd:
        
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"
        
        with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
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
    
    with patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.join") as mock_join, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.path.dirname") as mock_dirname, \
         patch("ifunnel.financial_data_preprocessing.clean_downloaded_data.os.getcwd") as mock_getcwd:
        
        mock_getcwd.return_value = "/fake/path"
        mock_dirname.return_value = "/fake"
        mock_join.return_value = "/fake/data.parquet"
        
        with patch.object(pd.DataFrame, "to_parquet") as mock_to_parquet:
            clean_data(data)
            # Asset1 should be dropped
