"""Module for downloading financial data from Yahoo Finance.

This module provides functions to download historical price data for ETFs and
other financial instruments from Yahoo Finance using the yfinance library.
It handles data retrieval, error handling, and basic preprocessing.
"""

import os

import pandas as pd
import yfinance as yf
from loguru import logger


def download_data(start_date: str, end_date: str, tickers: list[str]) -> pd.DataFrame | None:
    """Download historical adjusted close price data for specified tickers.

    This function retrieves historical price data from Yahoo Finance for a list
    of ticker symbols between the specified date range. It handles potential
    download errors and returns the adjusted close prices.

    Args:
        start_date: Start date for data retrieval in 'YYYY-MM-DD' format
        end_date: End date for data retrieval in 'YYYY-MM-DD' format
        tickers: List of ticker symbols to download data for

    Returns:
        Optional[pd.DataFrame]: DataFrame containing adjusted close prices with
                               dates as index and tickers as columns, or None
                               if the download fails

    Raises:
        No exceptions are raised as errors are caught and logged
    """
    # Download price data from Yahoo! finance based on list of ETF tickers and start/end dates.
    # `auto_adjust=True` folds splits and dividends into "Close" and drops the separate
    # "Adj Close" column, so the adjusted series is read from "Close". Passed explicitly
    # rather than relying on the default, which yfinance flipped in 0.2.51.
    try:
        daily_prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
    except Exception as e:  # noqa: BLE001  # yfinance surfaces many error types; treat any as a failed download
        logger.warning(f"⚠️ Problem when downloading our data with an error: {e}")
        daily_prices = None

    return daily_prices


if __name__ == "__main__":  # pragma: no cover
    # Load tickers' names
    path_to_tickers = os.path.join(os.path.dirname(os.getcwd()), "financial_data/top_2000_etfs.xlsx")
    data_excel = pd.read_excel(path_to_tickers)
    tickers = data_excel["List of Top 100 ETFs"].to_list()[1:]
    mapping = dict(
        zip(
            data_excel["List of Top 100 ETFs"].to_list()[1:],
            data_excel["Unnamed: 1"].to_list()[1:],
            strict=False,
        )
    )

    # Download raw data
    data_yahoo = download_data(start_date="2022-12-31", end_date="2023-07-30", tickers=tickers)
    if data_yahoo is None:
        raise ValueError("Failed to download Yahoo data")  # noqa: TRY003
    data_yahoo.columns = [
        data_yahoo.columns,
        [mapping[col] for col in data_yahoo.columns],
    ]
    data_yahoo.to_parquet(os.path.join(os.path.dirname(os.getcwd()), "financial_data/daily_price.parquet"))
