"""Per-asset statistics and the asset-selection methods of :class:`ifunnel.TradeBot`.

Risk/return statistics over a window, consistent top performers across periods,
and diversified subsets by repeated minimum spanning trees or by clustering.
Figures are left to the caller.
"""

import numpy as np
import pandas as pd

from .clustering import cluster, pick_cluster
from .data_analyser import mean_an_returns
from .mst import minimum_spanning_tree

# Upper bound of the annualised return standard deviation of each risk class
_RISK_CLASSES = {
    "Risk Class 1": 0.005,
    "Risk Class 2": 0.02,
    "Risk Class 3": 0.05,
    "Risk Class 4": 0.10,
    "Risk Class 5": 0.15,
    "Risk Class 6": 0.25,
    "Risk Class 7": 1,
}


def _window(weekly_returns: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Return a copy of the rows dated from ``start_date`` to ``end_date``, both inclusive."""
    return weekly_returns[(weekly_returns.index >= start_date) & (weekly_returns.index <= end_date)].copy()


def asset_stats(weekly_returns: pd.DataFrame, names: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Compute annualised return, volatility and Sharpe ratio for every asset.

    See :meth:`ifunnel.TradeBot.get_stat`; ``weekly_returns`` and ``names`` are the
    bot's data.
    """
    # ANALYZE THE DATA for a given time period
    weekly_data = _window(weekly_returns, start_date, end_date)

    # Create table with summary statistics
    mu_ga = mean_an_returns(weekly_data)  # Annualised geometric mean of returns
    std_dev_a = weekly_data.std(axis=0) * np.sqrt(52)  # Annualised standard deviation of returns
    sharpe = round(mu_ga / std_dev_a, 2)  # Sharpe ratio of each financial product

    # Write all results into a data frame
    stat_df = pd.concat([mu_ga, std_dev_a, sharpe], axis=1)  # ty: ignore[no-matching-overload]
    stat_df.columns = [
        "Average Annual Returns",
        "Standard Deviation of Returns",
        "Sharpe Ratio",
    ]
    stat_df["ISIN"] = stat_df.index  # Add names into the table
    stat_df["Name"] = names
    stat_df["Size"] = 1
    stat_df["Type"] = "ETF"

    return stat_df


def top_performing_assets(
    weekly_returns: pd.DataFrame,
    tickers: list[str],
    names: list[str],
    time_periods: list[tuple[str, str]],
    top_percent: float = 0.2,
) -> list[str]:
    """Select assets that are consistent top performers across all given periods.

    See :meth:`ifunnel.TradeBot.get_top_performing_assets`; ``weekly_returns``,
    ``tickers`` and ``names`` are the bot's data.
    """
    stats_for_periods = [asset_stats(weekly_returns, names, *period) for period in time_periods]

    for data in stats_for_periods:
        data["Risk Class"] = pd.cut(
            data["Standard Deviation of Returns"],
            bins=[-1, *list(_RISK_CLASSES.values())],
            labels=list(_RISK_CLASSES.keys()),
            right=True,
        )
    # For each data_period and each risk class, find the top 20% best performing assets
    # mark them as True in column 'Top Performer'
    for data in stats_for_periods:
        for risk_class in _RISK_CLASSES:
            data.loc[
                data["Risk Class"] == risk_class,
                "Top Performer",
            ] = data.loc[data["Risk Class"] == risk_class, "Sharpe Ratio"].rank(pct=True) > (1 - top_percent)

    # ISIN codes for assets which were top performers in all n periods
    top_isins = stats_for_periods[0].loc[stats_for_periods[0]["Top Performer"], "ISIN"].values
    for data in stats_for_periods:
        top_isins = np.intersect1d(top_isins, data.loc[data["Top Performer"], "ISIN"].values)

    return [names[tickers.index(isin)] for isin in top_isins]


def mst_subset(
    weekly_returns: pd.DataFrame, start_date: str, end_date: str, n_mst_runs: int
) -> tuple[list, pd.DataFrame]:
    """Reduce the assets with ``n_mst_runs`` successive minimum spanning trees.

    See :meth:`ifunnel.TradeBot.mst`; ``weekly_returns`` is the bot's data.

    Returns:
        tuple: ``(subset, subset_returns)``: the selected tickers and their returns
            over the window (all assets' returns when ``n_mst_runs`` is 0).
    """
    subset_mst, subset_mst_df = [], _window(weekly_returns, start_date, end_date)
    for _i in range(n_mst_runs):
        subset_mst, subset_mst_df, _corr_mst_avg, _pdi_mst = minimum_spanning_tree(subset_mst_df)
    return subset_mst, subset_mst_df


def cluster_subset(
    weekly_returns: pd.DataFrame,
    names: list[str],
    start_date: str,
    end_date: str,
    n_clusters: int,
    n_assets: int,
) -> tuple[list, pd.DataFrame, str]:
    """Pick the best Sharpe-ratio assets of each correlation cluster.

    See :meth:`ifunnel.TradeBot.clustering`; ``weekly_returns`` and ``names`` are
    the bot's data.

    Returns:
        tuple: ``(subset, clusters, last_date)``: the selected tickers, the cluster
            of every asset, and the last date in the window.
    """
    dataset = _window(weekly_returns, start_date, end_date)
    # CLUSTER DATA
    clusters = cluster(dataset, n_clusters)

    # SELECT ASSETS
    last_date = str(dataset.index.date[-1])  # ty: ignore[unresolved-attribute]
    clustering_stats = asset_stats(weekly_returns, names, start_date, last_date)
    subset_clustering, _subset_clustering_df = pick_cluster(
        data=dataset, stat=clustering_stats, ml=clusters, n_assets=n_assets
    )  # Number of assets from each cluster

    return subset_clustering, clusters, last_date
