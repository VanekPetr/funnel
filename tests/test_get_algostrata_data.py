"""Tests for AlgoStrata data fetching module."""

from unittest.mock import patch

import pandas as pd
import pytest

from ifunnel.financial_data_preprocessing.get_algostrata_data import batch, get_algostrata_data


def test_batch_basic():
    """Test basic batch functionality."""
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = list(batch(items, 3))
    assert len(result) == 4
    assert result[0] == [1, 2, 3]
    assert result[1] == [4, 5, 6]
    assert result[2] == [7, 8, 9]
    assert result[3] == [10]


def test_batch_exact_division():
    """Test batch with exact division."""
    items = [1, 2, 3, 4, 5, 6]
    result = list(batch(items, 2))
    assert len(result) == 3
    assert all(len(b) == 2 for b in result)


def test_batch_single_element():
    """Test batch with single element batches."""
    items = [1, 2, 3]
    result = list(batch(items, 1))
    assert len(result) == 3
    assert all(len(b) == 1 for b in result)


def test_get_algostrata_data_success():
    """Test successful AlgoStrata data fetching."""
    with (
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.get") as mock_get,
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.post") as mock_post,
    ):
        # Mock the names API response
        mock_get.return_value.json.return_value = [
            {"id": "1", "isin": "US123", "name": "Asset1"},
            {"id": "2", "isin": "US456", "name": "Asset2"},
        ]

        # Mock the prices API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "result": [
                {
                    "priceData": {
                        "reInvestedPrices": [
                            {"date": "2023-01-01T00:00:00Z", "unit_DKK": 100},
                            {"date": "2023-01-02T00:00:00Z", "unit_DKK": 101},
                        ]
                    }
                }
            ]
        }

        result = get_algostrata_data()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


def test_get_algostrata_data_with_null_price_data():
    """Test AlgoStrata data fetching with null price data."""
    with (
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.get") as mock_get,
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.post") as mock_post,
    ):
        # Mock the names API response
        mock_get.return_value.json.return_value = [
            {"id": "1", "isin": "US123", "name": "Asset1"},
        ]

        # Mock the prices API response with null price data
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"result": [{"priceData": None}]}

        # This will fail with UnboundLocalError because daily_prices is never created
        with pytest.raises(UnboundLocalError):
            get_algostrata_data()


def test_get_algostrata_data_error_response():
    """Test AlgoStrata data fetching with error response."""
    with (
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.get") as mock_get,
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.post") as mock_post,
    ):
        # Mock the names API response
        mock_get.return_value.json.return_value = [
            {"id": "1", "isin": "US123", "name": "Asset1"},
        ]

        # Mock an error response from prices API
        mock_post.return_value.status_code = 500
        mock_post.return_value.reason = "Internal Server Error"
        mock_post.return_value.text = "Error message"

        # This will fail with UnboundLocalError because daily_prices is never created
        with pytest.raises(UnboundLocalError):
            get_algostrata_data()
