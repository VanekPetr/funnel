"""Module for financial data analysis and performance metrics calculation.

This module provides functions for analyzing financial data, calculating
performance metrics such as returns, volatility, and Sharpe ratio, and
preprocessing data for further analysis.
"""

import numpy as np
import pandas as pd


def mean_an_returns(data: pd.DataFrame) -> float | pd.Series:
    """Compute the annualized geometric mean of returns.

    This function calculates the annualized geometric mean return from a
    series of periodic returns, assuming 52 periods per year (weekly data).

    Args:
        data: DataFrame containing periodic returns with time as index
             and assets as columns

    Returns:
        Union[float, pd.Series]: Annualized geometric mean return for each asset.
                                If data has one column, returns a float;
                                otherwise, returns a pandas Series.
    """
    result = 1
    for i in range(len(data.index)):
        result *= 1 + data.iloc[i, :]

    result = result ** (1 / float(len(data.index) / 52)) - 1

    return result


def final_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Compute performance statistics for financial data.

    This function calculates key performance metrics including:
    - Annualized geometric mean returns
    - Annualized standard deviation of returns
    - Sharpe ratio

    The function first converts price data to returns if needed.

    Args:
        data: DataFrame containing price or return data with time as index
             and assets as columns

    Returns:
        pd.DataFrame: DataFrame containing performance statistics for each asset,
                     with assets as index and metrics as columns
    """
    # TABLE WITH AVG RET AND STD OF RET
    data = data.pct_change()
    data = data.drop(data.index[:1])

    mu_ga = round(mean_an_returns(data), 2)  # annual geometric mean
    std_dev_a = round(data.std(axis=0) * np.sqrt(52), 2)  # standard deviation of Annual Returns

    stat_df = pd.concat([mu_ga, std_dev_a], axis=1)  # table  # ty: ignore[no-matching-overload]
    stat_names = ["Average Annual Returns", "Standard Deviation of Returns"]
    stat_df.columns = stat_names  # add names

    # COMPUTE SHARPE RATIO AND ADD IT INTO THE TABLE
    sharpe = round(
        stat_df.loc[:, "Average Annual Returns"] / stat_df.loc[:, "Standard Deviation of Returns"],
        2,
    )
    stat_df = pd.concat([stat_df, sharpe], axis=1)  # add sharpe ratio into the table
    stat_names = ["Avg An Ret", "Std Dev of Ret", "Sharpe R"]
    stat_df.columns = stat_names

    return stat_df


def get_weekly_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Extract and calculate weekly returns from price data.

    This function processes price data to extract only Wednesday prices
    (to avoid weekend effects) and calculates weekly percentage returns.
    The first row of returns (which contains NaN values) is dropped.

    Args:
        data: DataFrame containing daily price data with datetime index
             and assets as columns

    Returns:
        pd.DataFrame: DataFrame containing weekly returns with Wednesday dates
                     as index and assets as columns
    """
    # DEFINE IF WE WORK WITH ISIN CODES OR NAMES OF MUTUAL FUNDS
    prices_df = data
    # MODIFY THE DATA
    prices_on_wed = prices_df[pd.DatetimeIndex(prices_df.index).weekday == 2]  # ty: ignore[unresolved-attribute] # Only wednesdays

    # Get weekly returns
    weekly_returns = prices_on_wed.pct_change()
    weekly_returns = weekly_returns.drop(weekly_returns.index[:1])  # drop first NaN row

    return weekly_returns
