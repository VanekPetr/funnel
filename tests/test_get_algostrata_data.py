"""Tests for AlgoStrata data fetching module."""

from unittest.mock import patch

import pandas as pd
import pytest

from ifunnel.financial_data_preprocessing.get_algostrata_data import batch, get_algostrata_data
from ifunnel.settings import settings

_ALGOSTRATA_SETTINGS = ("ALGOSTRATA_NAMES_URL", "ALGOSTRATA_PRICES_URL", "ALGOSTRATA_KEY")


@pytest.fixture(autouse=True)
def algostrata_configured(monkeypatch):
    """Configure every AlgoStrata setting, so the mocked requests are reached."""
    for name in _ALGOSTRATA_SETTINGS:
        monkeypatch.setattr(settings, name, f"test-{name.lower()}")


@pytest.mark.parametrize("unset", _ALGOSTRATA_SETTINGS)
def test_get_algostrata_data_requires_configuration(monkeypatch, unset):
    """An unset AlgoStrata setting is named in the error, and no request is sent."""
    monkeypatch.setattr(settings, unset, None)
    with (
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.get") as mock_get,
        pytest.raises(ValueError, match=f"AlgoStrata is not configured: set {unset} "),
    ):
        get_algostrata_data()
    mock_get.assert_not_called()


def test_get_algostrata_data_names_every_missing_setting(monkeypatch):
    """With nothing configured, all three settings are named."""
    for name in _ALGOSTRATA_SETTINGS:
        monkeypatch.setattr(settings, name, None)
    with pytest.raises(ValueError, match="set ALGOSTRATA_NAMES_URL, ALGOSTRATA_PRICES_URL, ALGOSTRATA_KEY "):
        get_algostrata_data()


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

        with pytest.raises(ValueError, match="No price data"):
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

        with pytest.raises(ValueError, match="No price data"):
            get_algostrata_data()


def test_get_algostrata_data_with_nan_values():
    """Test AlgoStrata data fetching with NaN values in price data."""
    with (
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.get") as mock_get,
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.post") as mock_post,
    ):
        # Mock the names API response with multiple assets
        mock_get.return_value.json.return_value = [
            {"id": "1", "isin": "US123", "name": "Asset1"},
            {"id": "2", "isin": "US456", "name": "Asset2"},
            {"id": "3", "isin": "US789", "name": "Asset3"},
        ]

        # Mock the prices API response - first has valid data, second has NaN data
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
                },
                {
                    "priceData": {
                        "reInvestedPrices": [
                            {"date": "2023-01-01T00:00:00Z", "unit_DKK": None},
                            {"date": "2023-01-02T00:00:00Z", "unit_DKK": None},
                        ]
                    }
                },
                {
                    "priceData": {
                        "reInvestedPrices": [
                            {"date": "2023-01-01T00:00:00Z", "unit_DKK": 200},
                            {"date": "2023-01-02T00:00:00Z", "unit_DKK": 201},
                        ]
                    }
                },
            ]
        }

        result = get_algostrata_data()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


def _price_result(*prices):
    """Return one asset's entry in a prices API response, dated from 2023-01-01."""
    return {
        "priceData": {
            "reInvestedPrices": [
                {"date": f"2023-01-0{day}T00:00:00Z", "unit_DKK": price} for day, price in enumerate(prices, start=1)
            ]
        }
    }


def test_get_algostrata_data_mixed_responses():
    """A failed middle batch drops its assets without shifting the labels of later ones."""
    with (
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.get") as mock_get,
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.post") as mock_post,
    ):
        # Nine assets, so three batches of three
        mock_get.return_value.json.return_value = [
            {"id": str(i), "isin": f"US{i:03d}", "name": f"Asset{i}"} for i in range(9)
        ]

        def mock_post_side_effect(*args, **kwargs):
            """Fail the second POST call and price every requested asset otherwise."""
            mock_response = type("Response", (), {})()
            if mock_post.call_count == 2:
                mock_response.status_code = 500
                mock_response.reason = "Internal Server Error"
                mock_response.text = "Error"
            else:
                ids = kwargs["json"]["idList"]
                mock_response.status_code = 200
                mock_response.json = lambda: {"result": [_price_result(int(i), int(i) + 1) for i in ids]}
            return mock_response

        mock_post.side_effect = mock_post_side_effect

        result = get_algostrata_data()

        assert list(result.columns) == [
            ("US000", "Asset0"),
            ("US001", "Asset1"),
            ("US002", "Asset2"),
            ("US006", "Asset6"),
            ("US007", "Asset7"),
            ("US008", "Asset8"),
        ]
        # Each column holds the prices of the asset it is labelled with
        assert result[("US007", "Asset7")].tolist() == [7, 8]


def test_get_algostrata_data_first_asset_without_prices():
    """When the first asset has no price data, the next one keeps its own label."""
    with (
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.get") as mock_get,
        patch("ifunnel.financial_data_preprocessing.get_algostrata_data.requests.post") as mock_post,
    ):
        mock_get.return_value.json.return_value = [
            {"id": "1", "isin": "US123", "name": "Asset1"},
            {"id": "2", "isin": "US456", "name": "Asset2"},
            {"id": "3", "isin": "US789", "name": "Asset3"},
        ]
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "result": [{"priceData": None}, _price_result(200, 201), _price_result(300, 301)]
        }

        result = get_algostrata_data()

        assert list(result.columns) == [("US456", "Asset2"), ("US789", "Asset3")]
        assert result[("US456", "Asset2")].tolist() == [200, 201]
