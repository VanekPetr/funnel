"""Main module for portfolio optimization and analysis.

This module provides a trading bot interface for financial portfolio optimization
and analysis. It integrates various optimization models, asset selection methods,
and visualization tools to help users build and analyze diversified portfolios.
The module supports backtesting, scenario analysis, and lifecycle investment modeling.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger

from . import plotting
from .clustering import cluster, pick_cluster
from .cvar_model import cvar_model
from .cvar_targets import get_cvar_targets
from .data_analyser import final_stats, mean_an_returns
from .lifecycle.glide_path_creator import generate_risk_profiles
from .lifecycle.mvo_lifecycle_model import (
    get_port_allocations,
    riskadjust_model_scen,
)
from .mst import minimum_spanning_tree
from .mvo_model import mvo_model
from .mvo_targets import get_mvo_targets
from .scenario_generation import MomentGenerator, ScenarioGenerator

pio.renderers.default = "browser"

# that's unfortunate but will be addressed later
# ROOT_DIR = Path(__file__).parent.parent
# Load our data
# weekly_returns = pd.read_parquet(os.path.join(ROOT_DIR, "financial_data/all_etfs_rets.parquet.gzip"))
# tickers = [pair[0] for pair in weekly_returns.columns.values]
# names = [pair[1] for pair in weekly_returns.columns.values]


@lru_cache(maxsize=1)  # Cache the result of this function
def initialize_bot(file: str | Path | None = None) -> "_TradeBot":
    """Initialize and return a trading bot instance.

    This function creates and returns a _TradeBot instance initialized with
    financial data from the specified file. The result is cached to avoid
    reloading the data unnecessarily.

    Args:
        file: Path to the parquet file containing financial return data.
             If None, uses the default file in the financial_data directory.

    Returns:
        _TradeBot: An initialized trading bot instance ready for analysis.
    """
    if file is None:
        root_dir = Path(__file__).parent.parent
        file = root_dir / "financial_data" / "all_etfs_rets.parquet.gzip"

    weekly_returns = pd.read_parquet(file)

    tickers = [pair[0] for pair in weekly_returns.columns.values]
    names = [pair[1] for pair in weekly_returns.columns.values]
    weekly_returns.columns = tickers
    return _TradeBot(tickers, names, weekly_returns)


class _TradeBot:
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
                           and assets as columns
        """
        self.tickers = tickers
        self.names = names
        self.weeklyReturns = weekly_returns
        self.min_date = str(weekly_returns.index[0])
        self.max_date = str(weekly_returns.index[-1])

        weekly_returns.columns = tickers

    @staticmethod
    def __plot_backtest(
        performance: pd.DataFrame,
        performance_benchmark: pd.DataFrame,
        composition: pd.DataFrame,
        names: list[str],
        tickers: list[str],
    ) -> tuple[go.Figure, go.Figure]:
        """Build the backtest performance and composition figures.

        Thin wrapper over :func:`plotting.plot_backtest` (kept on the class so
        the plotting logic lives in one module while the public/test call
        surface is preserved).
        """
        return plotting.plot_backtest(performance, performance_benchmark, composition, names, tickers)

    @staticmethod
    def __plot_portfolio_densities(
        portfolio_performance_dict: dict,
        compositions: dict[str, pd.DataFrame],
        tickers: list,
        names: list,
    ) -> tuple[go.Figure, dict[str, go.Figure], go.Figure]:
        """Build the lifecycle terminal-wealth density and composition figures.

        Thin wrapper over :func:`plotting.plot_portfolio_densities`.
        """
        return plotting.plot_portfolio_densities(portfolio_performance_dict, compositions, tickers, names)

    def get_stat(self, start_date: str, end_date: str) -> pd.DataFrame:
        """METHOD COMPUTING ANNUAL RETURNS, ANNUAL STD. DEV. & SHARPE RATIO OF ASSETS."""
        # ANALYZE THE DATA for a given time period
        weekly_data = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date) & (self.weeklyReturns.index <= end_date)
        ].copy()

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
        stat_df["Name"] = self.names
        stat_df["Size"] = 1
        stat_df["Type"] = "ETF"

        return stat_df

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
        stats_for_periods = {f"period_{i}": self.get_stat(*period) for i, period in enumerate(time_periods, 1)}

        # Create 'Risk class' column where the value is
        # 'Risk Class 1' if Standard Deviation of Returns <= 0.005
        # 'Risk Class 2' if > 0.005 and < 0.02
        # 'Risk Class 3' if > 0.02 and < 0.05
        # 'Risk Class 4' if > 0.05 and < 0.1
        # 'Risk Class 5' if > 0.1 and < 0.15
        # 'Risk Class 6' if > 0.15 and < 0.25 then
        # 'Risk Class 7' if > 0.25
        risk_level = {
            "Risk Class 1": 0.005,
            "Risk Class 2": 0.02,
            "Risk Class 3": 0.05,
            "Risk Class 4": 0.10,
            "Risk Class 5": 0.15,
            "Risk Class 6": 0.25,
            "Risk Class 7": 1,
        }
        for data in stats_for_periods.values():
            data["Risk Class"] = pd.cut(
                data["Standard Deviation of Returns"],
                bins=[-1, *list(risk_level.values())],
                labels=list(risk_level.keys()),
                right=True,
            )
        # For each data_period and each risk class, find the top 20% best performing assets
        # mark them as True in column 'Top Performer'
        for data in stats_for_periods.values():
            for risk_class in risk_level:
                data.loc[
                    data["Risk Class"] == risk_class,
                    "Top Performer",
                ] = data.loc[data["Risk Class"] == risk_class, "Sharpe Ratio"].rank(pct=True) > (1 - top_percent)
        # for each period, save the pandas dataframe into excel files
        # for index, data in enumerate(stats_for_periods.values()):
        #     data.to_excel(f"top_performers_{time_periods[index]}.xlsx")

        # ISIN codes for assets which were top performers in all n periods
        top_isins = stats_for_periods["period_1"].loc[stats_for_periods["period_1"]["Top Performer"], "ISIN"].values
        for data in stats_for_periods.values():
            top_isins = np.intersect1d(top_isins, data.loc[data["Top Performer"], "ISIN"].values)

        top_names = [self.names[self.tickers.index(isin)] for isin in top_isins]

        return top_names

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
        """METHOD TO RUN MST METHOD AND PRINT RESULTS."""
        fig, subset_mst = None, []

        # Starting subset of data for MST
        subset_mst_df = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date) & (self.weeklyReturns.index <= end_date)
        ].copy()

        for _i in range(n_mst_runs):
            subset_mst, subset_mst_df, _corr_mst_avg, _pdi_mst = minimum_spanning_tree(subset_mst_df)

        # PLOTTING RESULTS
        if plot and len(subset_mst) > 0:
            end_df_date = str(pd.DatetimeIndex(subset_mst_df.index).date[-1])  # ty: ignore[unresolved-attribute]
            fig = self.plot_dots(
                start_date=start_date,
                end_date=end_df_date,
                ml="MST",
                ml_subset=subset_mst,
            )

        return fig, subset_mst

    def clustering(
        self,
        start_date: str,
        end_date: str,
        n_clusters: int,
        n_assets: int,
        plot: bool = False,
    ) -> tuple:
        """METHOD TO RUN MST METHOD AND PRINT RESULTS."""
        fig = None
        dataset = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date) & (self.weeklyReturns.index <= end_date)
        ].copy()
        # CLUSTER DATA
        clusters = cluster(dataset, n_clusters)

        # SELECT ASSETS
        end_dataset_date = str(dataset.index.date[-1])
        clustering_stats = self.get_stat(start_date, end_dataset_date)
        subset_clustering, _subset_clustering_df = pick_cluster(
            data=dataset, stat=clustering_stats, ml=clusters, n_assets=n_assets
        )  # Number of assets from each cluster

        # PLOTTING DATA
        if plot:
            fig = self.plot_dots(
                start_date=start_date,
                end_date=end_dataset_date,
                ml="Clustering",
                ml_subset=clusters,
            )

            # fig.show()

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
        """METHOD TO COMPUTE THE BACKTEST."""
        # Find Benchmarks' ISIN codes
        benchmark_isin = [self.tickers[list(self.names).index(name)] for name in benchmarks]

        # Get train and testing datasets
        whole_dataset = self.weeklyReturns[
            (self.weeklyReturns.index >= start_train_date) & (self.weeklyReturns.index <= end_test_date)
        ].copy()
        test_dataset = self.weeklyReturns[
            (self.weeklyReturns.index > start_test_date) & (self.weeklyReturns.index <= end_test_date)
        ].copy()

        # SCENARIO GENERATION
        # ---------------------------------------------------------------------------------------------------
        # Create scenario generator
        sg = ScenarioGenerator(np.random.default_rng())

        if model == "Markowitz model" or scenarios_type == "MonteCarlo":
            sigma_lst, mu_lst = MomentGenerator.generate_sigma_mu_for_test_periods(
                data=whole_dataset[subset_of_assets], n_test=len(test_dataset.index)
            )

        if scenarios_type == "MonteCarlo":
            scenarios = sg.monte_carlo(
                data=whole_dataset[subset_of_assets],  # subsetMST_df or subsetCLUST_df
                n_simulations=n_simulations,
                n_test=len(test_dataset.index),
                sigma_lst=sigma_lst,
                mu_lst=mu_lst,
            )
        else:
            scenarios = sg.bootstrapping(
                data=whole_dataset[subset_of_assets],  # subsetMST or subsetCLUST
                n_simulations=n_simulations,  # number of scenarios per period
                n_test=len(test_dataset.index),
            )  # number of periods

        # TARGETS GENERATION
        # ---------------------------------------------------------------------------------------------------
        start_of_test_dataset = str(test_dataset.index.date[0])
        if model == "Markowitz model":
            targets, benchmark_port_val = get_mvo_targets(
                test_date=start_of_test_dataset,
                benchmark=benchmark_isin,
                budget=100,
                data=whole_dataset,
            )

        else:
            targets, benchmark_port_val = get_cvar_targets(
                test_date=start_of_test_dataset,
                benchmark=benchmark_isin,
                budget=100,
                cvar_alpha=0.05,
                data=whole_dataset,
                scgen=sg,
                n_simulations=n_simulations,
            )

        # MATHEMATICAL MODELING
        # ---------------------------------------------------------------------------------------------------
        if model == "Markowitz model":
            port_allocation, port_value, _port_cvar = mvo_model(
                test_ret=test_dataset[subset_of_assets],
                mu_lst=mu_lst,
                sigma_lst=sigma_lst,
                targets=targets,
                budget=100,
                trans_cost=0.001,
                max_weight=1,
                solver=solver,
                lower_bound=lower_bound,
            )
        #                                                      inaccurate=inaccurate_solution)

        else:
            port_allocation, port_value, _port_cvar = cvar_model(
                test_ret=test_dataset[subset_of_assets],
                scenarios=scenarios,  # Scenarios
                targets=targets,  # Target
                budget=100,
                cvar_alpha=0.05,
                trans_cost=0.001,
                max_weight=1,
                solver=solver,
                lower_bound=lower_bound,
            )
        #                                                       inaccurate=inaccurate_solution)

        # PLOTTING
        # ------------------------------------------------------------------
        fig_performance, fig_composition = self.__plot_backtest(
            performance=port_value.copy(),
            performance_benchmark=benchmark_port_val.copy(),
            composition=port_allocation,
            names=self.names,
            tickers=self.tickers,
        )

        # RETURN STATISTICS
        # ------------------------------------------------------------------
        optimal_portfolio_stat = final_stats(port_value)
        benchmark_stat = final_stats(benchmark_port_val)

        return optimal_portfolio_stat, benchmark_stat, fig_performance, fig_composition

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
    ) -> tuple[dict, pd.DataFrame, go.Figure, go.Figure, dict, dict, go.Figure]:
        """METHOD TO COMPUTE THE LIFECYCLE SCENARIO ANALYSIS."""
        # ------------------------------- INITIALIZE FUNCTION -------------------------------
        n_periods = end_year - 2023
        withdrawal_lst = [withdrawals * (1 + 0.04) ** i for i in range(n_periods)]

        # ------------------------------- PARAMETER INITIALIZATION -------------------------------
        if test_split != 0:
            sampling_set, estimating_set = MomentGenerator.split_dataset(
                data=self.weeklyReturns[subset_of_assets], sampling_ratio=test_split
            )

            _, _, sigma_weekly, mu_weekly = MomentGenerator.generate_annual_sigma_mu_with_risk_free(data=sampling_set)

            sigma, mu, _, _ = MomentGenerator.generate_annual_sigma_mu_with_risk_free(data=estimating_set)
        else:
            sigma, mu, sigma_weekly, mu_weekly = MomentGenerator.generate_annual_sigma_mu_with_risk_free(
                data=self.weeklyReturns[subset_of_assets]
            )

        # ------------------------------- SCENARIO GENERATION -------------------------------
        if rng_seed == 0:
            sg = ScenarioGenerator(np.random.default_rng())
        else:
            sg = ScenarioGenerator(np.random.default_rng(rng_seed))

        if scenarios_type == "MonteCarlo":
            scenarios = sg.mc_simulation_annual_from_weekly(
                weekly_mu=mu_weekly,
                weekly_sigma=sigma_weekly,
                n_simulations=n_simulations,
                n_years=n_periods,
            )

        elif scenarios_type == "Bootstrap":
            scenarios = sg.bootstrap_simulation_annual_from_weekly(
                historical_weekly_returns=self.weeklyReturns[subset_of_assets],
                n_simulations=n_simulations,
                n_years=n_periods,
            )

        else:
            raise ValueError(  # noqa: TRY003
                "It appears that a scenario method other than MonteCarlo or Bootstrap has been chosen. "
                "Please check for spelling mistakes."
            )

        # ------------------------------- Allocation Target Generation -------------------------------
        glide_paths_df, fig_glidepaths = generate_risk_profiles(
            n_periods=n_periods, initial_risk=initial_risk_appetite, minimum_risk=0.01
        )

        allocation_targets = {}
        for r in glide_paths_df.columns:
            targets = get_port_allocations(
                mu_lst=mu,
                sigma_lst=sigma,
                targets=glide_paths_df[r],
                max_weight=1 / 4,
                solver="CLARABEL",
            )
            allocation_targets[f"{r}"] = targets

        # ------------------------------- MATHEMATICAL MODELING -------------------------------
        exhibition_summary = pd.DataFrame()
        terminal_wealth_dict = {}

        for key, df in allocation_targets.items():
            logger.info(
                f"Optimizing portfolio for {key} over {n_simulations} scenarios. An info message will "
                f"appear, when we are halfway through the scenarios for the current strategy."
            )
            portfolio_df, _mean_allocations_df, analysis_metrics = riskadjust_model_scen(
                scen=scenarios[:, :, :],
                targets=df,
                budget=initial_budget,
                trans_cost=0.002,
                withdrawal_lst=withdrawal_lst,
                interest_rate=0.04,
            )

            # Add the analysis_metrics DataFrame as a new column in the storage DataFrame
            exhibition_summary[key] = analysis_metrics.squeeze()

            portfolio_df["Terminal Wealth"] = pd.to_numeric(portfolio_df["Terminal Wealth"], errors="coerce")
            terminal_wealth_dict[f"{key}"] = portfolio_df

        # ------------------------------- PLOTTING -------------------------------
        fig_performance, fig_compositions, fig_compositions_all = self.__plot_portfolio_densities(
            portfolio_performance_dict=terminal_wealth_dict,
            compositions=allocation_targets,
            tickers=self.tickers,
            names=self.names,
        )

        # ------------------------------- RETURN STATISTICS -------------------------------
        return (
            terminal_wealth_dict,
            exhibition_summary,
            fig_performance,
            fig_glidepaths,
            allocation_targets,
            fig_compositions,
            fig_compositions_all,
        )


if __name__ == "__main__":  # pragma: no cover
    # INITIALIZATION OF THE CLASS

    # that's unfortunate but will be addressed later
    # ROOT_DIR = Path(__file__).parent.parent
    # Load our data
    # weekly_returns = pd.read_parquet(ROOT_DIR / "financial_data" / "all_etfs_rets.parquet.gzip")
    # algo = build_bot(weekly_returns=weekly_returns)

    algo = initialize_bot()

    # algo = TradeBot()

    # Get top performing assets for given periods and measure
    top_assets = algo.get_top_performing_assets(
        time_periods=[
            (algo.min_date, "2017-01-01"),
            ("2017-01-02", "2020-01-01"),
            ("2020-01-02", algo.max_date),
        ],
        top_percent=0.2,
    )

    # PLOT INTERACTIVE GRAPH
    algo.plot_dots(start_date=algo.min_date, end_date=algo.max_date, top_performers=top_assets)

    # RUN THE MINIMUM SPANNING TREE METHOD
    _, mst_subset_of_assets = algo.mst(start_date="2000-01-01", end_date="2024-01-01", n_mst_runs=5, plot=False)

    # RUN THE CLUSTERING METHOD
    _, clustering_subset_of_assets = algo.clustering(
        start_date="2015-12-23",
        end_date="2017-07-01",
        n_clusters=3,
        n_assets=10,
        plot=True,
    )

    # RUN THE LIFECYCLE
    lifecycle = algo.lifecycle_scenario_analysis(
        subset_of_assets=mst_subset_of_assets,
        scenarios_type="MonteCarlo",
        n_simulations=1000,
        end_year=2050,
        withdrawals=51000,
        initial_risk_appetite=0.15,
        initial_budget=137000,
    )

    # RUN THE BACKTEST
    backtest = algo.backtest(
        start_train_date="2015-12-23",
        start_test_date="2018-09-24",
        end_test_date="2019-09-01",
        subset_of_assets=mst_subset_of_assets,
        benchmarks=["BankInvest Danske Aktier W"],
        scenarios_type="Bootstrapping",
        n_simulations=500,
        model="Markowitz model",
        lower_bound=0,
    )
