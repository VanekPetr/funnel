"""Main module for portfolio optimization and analysis.

This module provides :class:`TradeBot`, the user-facing interface for financial
portfolio optimization and analysis. Its methods delegate the work to the modules
that implement it (:mod:`.asset_selection`, :mod:`.backtest`,
:mod:`.lifecycle.analysis` and :mod:`.plotting`), so this module holds the public
API and its documentation rather than the algorithms.
"""

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from . import plotting
from .asset_selection import asset_stats, cluster_subset, mst_subset, top_performing_assets
from .backtest import run_backtest
from .data_analyser import final_stats
from .lifecycle.analysis import run_lifecycle_analysis


@lru_cache(maxsize=1)  # Cache the parsed file; each bot gets its own copy
def _load_weekly_returns(file: str | Path) -> pd.DataFrame:
    """Read a weekly returns parquet file whose columns are (ticker, name) pairs."""
    return pd.read_parquet(file)


def initialize_bot(file: str | Path | None = None) -> "TradeBot":
    """Initialize and return a trading bot instance.

    This function creates and returns a new TradeBot instance initialized with
    financial data from the specified file. The parsed file is cached to avoid
    reloading the data unnecessarily, but every call returns a separate bot.

    Args:
        file: Path to the parquet file containing financial return data.
             If None, uses the default file in the financial_data directory.

    Returns:
        TradeBot: An initialized trading bot instance ready for analysis.

    Example:
        Without a file, the ETF returns bundled with the package are loaded:

        >>> bot = initialize_bot()
        >>> len(bot.tickers) == bot.weekly_returns.shape[1]
        True
        >>> initialize_bot() is bot
        False
    """
    if file is None:
        root_dir = Path(__file__).parent.parent
        file = root_dir / "financial_data" / "all_etfs_rets.parquet.gzip"

    weekly_returns = _load_weekly_returns(file)

    tickers = [pair[0] for pair in weekly_returns.columns.values]
    names = [pair[1] for pair in weekly_returns.columns.values]
    return TradeBot(tickers, names, weekly_returns)


class TradeBot:
    """Trading bot for financial portfolio optimization and analysis.

    This class provides methods for analyzing financial assets, selecting diversified
    subsets using machine learning algorithms, optimizing portfolios using various
    mathematical models, and visualizing results. It supports:

    - Asset performance analysis and visualization
    - Diversification using minimum spanning tree and clustering algorithms
    - Portfolio optimization with Mean-Variance and CVaR models
    - Backtesting of optimization strategies
    - Lifecycle investment modeling and scenario analysis
    """

    def __init__(self, tickers: list[str], names: list[str], weekly_returns: pd.DataFrame) -> None:
        """Initialize the trading bot with financial data.

        Args:
            tickers: List of ticker symbols or identifiers for the assets
            names: List of human-readable names corresponding to the tickers
            weekly_returns: DataFrame containing weekly return data with dates as index
                           and assets as columns. It is copied, so the caller's frame
                           is left unchanged.
        """
        self.tickers = tickers
        self.names = names
        self.weekly_returns = weekly_returns.copy()
        self.weekly_returns.columns = tickers
        self.min_date = str(weekly_returns.index[0])
        self.max_date = str(weekly_returns.index[-1])

    @property
    def weeklyReturns(self) -> pd.DataFrame:  # noqa: N802  # deprecated camelCase name kept for compatibility
        """Deprecated alias for :attr:`weekly_returns`."""
        warnings.warn(
            "TradeBot.weeklyReturns is deprecated; use TradeBot.weekly_returns instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.weekly_returns

    @weeklyReturns.setter
    def weeklyReturns(self, value: pd.DataFrame) -> None:  # noqa: N802
        """Deprecated alias for setting :attr:`weekly_returns`."""
        warnings.warn(
            "TradeBot.weeklyReturns is deprecated; use TradeBot.weekly_returns instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.weekly_returns = value

    def get_stat(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Compute annualised return, volatility and Sharpe ratio for every asset.

        Args:
            start_date: First date of the window (inclusive), e.g. ``"2015-12-23"``.
            end_date: Last date of the window (inclusive).

        Returns:
            pd.DataFrame: One row per ticker with the columns ``Average Annual Returns``,
                ``Standard Deviation of Returns``, ``Sharpe Ratio``, ``ISIN``, ``Name``,
                ``Size`` and ``Type``.
        """
        return asset_stats(self.weekly_returns, self.names, start_date, end_date)

    def get_top_performing_assets(self, time_periods: list[tuple[str, str]], top_percent: float = 0.2) -> list[str]:
        """Select assets that are consistent top performers across all given periods.

        For each period, assets are grouped into risk classes by their return
        standard deviation and, within each class, ranked by Sharpe ratio. An asset
        is flagged a top performer for a period if its Sharpe-ratio rank falls in the
        top ``top_percent`` of its risk class. Only assets flagged in *every* period
        are returned.

        Args:
            time_periods: List of (start_date, end_date) string tuples, one per period.
            top_percent: Fraction (0-1) of each risk class to treat as top performers.

        Returns:
            list[str]: Names of the assets that were top performers in all periods.
        """
        return top_performing_assets(self.weekly_returns, self.tickers, self.names, time_periods, top_percent)

    def plot_dots(
        self,
        start_date: str,
        end_date: str,
        ml: str = "",
        ml_subset: list | pd.DataFrame | None = None,
        fund_set: list | None = None,
        top_performers: list | None = None,
        optimal_portfolio: list | None = None,
        benchmark: list | None = None,
    ) -> go.Figure:
        """Plot the risk/return overview of the financial products.

        Delegates figure construction to :func:`plotting.dots_figure`, passing
        the per-asset statistics computed for the requested window.
        """
        data = self.get_stat(start_date, end_date)
        return plotting.dots_figure(
            data,
            start_date,
            end_date,
            ml=ml,
            ml_subset=ml_subset,
            fund_set=fund_set,
            top_performers=top_performers,
            optimal_portfolio=optimal_portfolio,
            benchmark=benchmark,
            names=self.names,
            tickers=self.tickers,
        )

    def mst(self, start_date: str, end_date: str, n_mst_runs: int, plot: bool = False) -> tuple:
        """Select a diversified subset of assets with repeated minimum spanning trees.

        Each run keeps the leaf nodes of the minimum spanning tree built on the
        previous run's subset, so more runs give a smaller, more diversified set.

        Args:
            start_date: First date of the window (inclusive).
            end_date: Last date of the window (inclusive).
            n_mst_runs: Number of successive MST reductions to apply.
            plot: If True, also build a risk/return figure highlighting the subset.

        Returns:
            tuple: ``(fig, subset)`` where ``fig`` is a :class:`plotly.graph_objects.Figure`
                (``None`` unless ``plot`` is True and the subset is non-empty) and
                ``subset`` is the list of selected tickers.
        """
        fig = None
        subset_mst, subset_mst_df = mst_subset(self.weekly_returns, start_date, end_date, n_mst_runs)
        if plot and len(subset_mst) > 0:
            end_df_date = str(pd.DatetimeIndex(subset_mst_df.index).date[-1])  # ty: ignore[unresolved-attribute]
            fig = self.plot_dots(start_date=start_date, end_date=end_df_date, ml="MST", ml_subset=subset_mst)
        return fig, subset_mst

    def clustering(
        self,
        start_date: str,
        end_date: str,
        n_clusters: int,
        n_assets: int,
        plot: bool = False,
    ) -> tuple:
        """Select assets by hierarchical clustering and per-cluster Sharpe ratio.

        Assets are grouped into ``n_clusters`` clusters on their return
        correlations, and the ``n_assets`` assets with the highest Sharpe ratio
        are taken from each cluster (all of them when a cluster is smaller).

        Args:
            start_date: First date of the window (inclusive).
            end_date: Last date of the window (inclusive).
            n_clusters: Number of clusters to form.
            n_assets: Number of assets to take from each cluster.
            plot: If True, also build a risk/return figure coloured by cluster.

        Returns:
            tuple: ``(fig, subset)`` where ``fig`` is a :class:`plotly.graph_objects.Figure`
                (``None`` unless ``plot`` is True) and ``subset`` is the list of
                selected tickers.
        """
        fig = None
        subset_clustering, clusters, last_date = cluster_subset(
            self.weekly_returns, self.names, start_date, end_date, n_clusters, n_assets
        )
        if plot:
            fig = self.plot_dots(start_date=start_date, end_date=last_date, ml="Clustering", ml_subset=clusters)
        return fig, subset_clustering

    def backtest(
        self,
        start_train_date: str,
        start_test_date: str,
        end_test_date: str,
        subset_of_assets: list,
        benchmarks: list,
        scenarios_type: str,
        n_simulations: int,
        model: str,
        solver: str = "CLARABEL",
        lower_bound: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, go.Figure, go.Figure]:
        """Backtest an optimised portfolio against a benchmark.

        The portfolio is rebalanced every four weeks over the test window
        ``(start_test_date, end_test_date]``, each time on moments or scenarios
        estimated from a rolling window of the history that starts at
        ``start_train_date``. The budget is 100, transaction costs are 0.1% and the
        CVaR model uses a 5% tail. Risk targets are taken from the benchmark, so the
        optimised portfolio is compared with the benchmark at the benchmark's risk.

        Args:
            start_train_date: First date of the estimation data.
            start_test_date: Last date before the test window starts.
            end_test_date: Last date of the test window (inclusive).
            subset_of_assets: Tickers the portfolio may invest in, e.g. the output of
                :meth:`mst` or :meth:`clustering`.
            benchmarks: Asset *names* (not tickers) that make up the benchmark.
            scenarios_type: ``"MonteCarlo"`` or ``"Bootstrap"`` (``"Bootstrapping"`` is a
                deprecated alias).
            n_simulations: Number of scenarios generated per period.
            model: ``"Markowitz model"`` (volatility target) or ``"CVaR model"``.
            solver: CVXPY solver name, as listed by ``cvxpy.installed_solvers()``.
            lower_bound: Minimum weight of each asset the portfolio holds.

        Returns:
            tuple: ``(portfolio_stats, benchmark_stats, fig_performance, fig_composition)``:
                performance statistics of the optimised portfolio and of the benchmark,
                and figures of the value paths and of the portfolio composition.

        Raises:
            ValueError: If ``model`` or ``scenarios_type`` is not one of the values above,
                or a benchmark name is not in the data.
        """
        port_value, benchmark_value, port_allocation = run_backtest(
            self.weekly_returns,
            self.tickers,
            self.names,
            start_train_date=start_train_date,
            start_test_date=start_test_date,
            end_test_date=end_test_date,
            subset_of_assets=subset_of_assets,
            benchmarks=benchmarks,
            scenarios_type=scenarios_type,
            n_simulations=n_simulations,
            model=model,
            solver=solver,
            lower_bound=lower_bound,
        )
        fig_performance, fig_composition = plotting.plot_backtest(
            port_value.copy(), benchmark_value.copy(), port_allocation, self.names, self.tickers
        )
        return final_stats(port_value), final_stats(benchmark_value), fig_performance, fig_composition

    def lifecycle_scenario_analysis(
        self,
        subset_of_assets: list,
        scenarios_type: str,
        n_simulations: int,
        end_year: int,
        withdrawals: int,
        initial_risk_appetite: float,
        initial_budget: int,
        rng_seed: int = 0,
        test_split: float = False,
        start_year: int | None = None,
    ) -> tuple[dict, pd.DataFrame, go.Figure, go.Figure, dict, dict, go.Figure]:
        """Simulate a lifecycle investment with withdrawals along several glide paths.

        Glide paths that lower the portfolio risk from ``initial_risk_appetite``
        to 1% are generated, each is turned into yearly target allocations, and
        every allocation is simulated over ``n_simulations`` return scenarios while
        withdrawing a yearly amount that grows by 4%.

        Args:
            subset_of_assets: Tickers the portfolio may invest in.
            scenarios_type: ``"MonteCarlo"`` or ``"Bootstrap"`` (``"Bootstrapping"`` is a
                deprecated alias).
            n_simulations: Number of return scenarios.
            end_year: Year the horizon ends (exclusive): the analysis covers the
                ``end_year - start_year`` years ``start_year, ..., end_year - 1``.
            withdrawals: Withdrawal in the first year, increased by 4% per year.
            initial_risk_appetite: Portfolio volatility at the start of the glide paths.
            initial_budget: Starting wealth.
            rng_seed: Seed for the scenario generator; ``0`` leaves it unseeded.
            test_split: If non-zero, the fraction of the history used to generate
                scenarios; the remainder estimates the moments behind the targets.
                If zero, the whole history is used for both.
            start_year: First year of the horizon; defaults to the current year.

        Returns:
            tuple: ``(terminal_wealth, summary, fig_performance, fig_glidepaths,
                allocation_targets, fig_compositions, fig_compositions_all)``: per glide
                path, the scenario results and the target allocations (dicts keyed by
                glide path name); a DataFrame of analysis metrics with one column per
                glide path; and the terminal-wealth density, glide path and composition
                figures (``fig_compositions`` is a dict of figures).

        Raises:
            ValueError: If ``scenarios_type`` is not one of the values above,
                or ``end_year`` is not after ``start_year``.
        """
        wealth, summary, fig_glidepaths, targets = run_lifecycle_analysis(
            self.weekly_returns,
            subset_of_assets=subset_of_assets,
            scenarios_type=scenarios_type,
            n_simulations=n_simulations,
            end_year=end_year,
            withdrawals=withdrawals,
            initial_risk_appetite=initial_risk_appetite,
            initial_budget=initial_budget,
            rng_seed=rng_seed,
            test_split=test_split,
            start_year=start_year,
        )
        fig_performance, fig_compositions, fig_all = plotting.plot_portfolio_densities(
            wealth, targets, self.tickers, self.names
        )
        return wealth, summary, fig_performance, fig_glidepaths, targets, fig_compositions, fig_all


def __getattr__(name: str) -> Any:
    """Resolve the deprecated ``_TradeBot`` alias of :class:`TradeBot`."""
    if name == "_TradeBot":
        warnings.warn("_TradeBot is deprecated; use TradeBot instead", DeprecationWarning, stacklevel=2)
        return TradeBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # noqa: TRY003
