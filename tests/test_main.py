"""Comprehensive tests for the main module to achieve 100% coverage."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.main import _TradeBot, initialize_bot


@pytest.fixture
def sample_weekly_returns():
    """Create sample weekly returns data for testing."""
    dates = pd.date_range("2020-01-01", periods=200, freq="W")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            ("ISIN1", "Asset A"): np.random.normal(0.002, 0.02, 200),
            ("ISIN2", "Asset B"): np.random.normal(0.003, 0.03, 200),
            ("ISIN3", "Asset C"): np.random.normal(0.001, 0.01, 200),
            ("ISIN4", "Benchmark Fund"): np.random.normal(0.002, 0.015, 200),
        },
        index=dates,
    )
    data.columns = pd.MultiIndex.from_tuples(data.columns)
    return data


@pytest.fixture
def trade_bot(sample_weekly_returns):
    """Create a TradeBot instance for testing."""
    tickers = [col[0] for col in sample_weekly_returns.columns]
    names = [col[1] for col in sample_weekly_returns.columns]
    weekly_returns = sample_weekly_returns.copy()
    weekly_returns.columns = tickers
    return _TradeBot(tickers, names, weekly_returns)


class TestInitializeBot:
    """Tests for the initialize_bot function."""

    def test_initialize_bot_with_file(self, tmp_path, sample_weekly_returns):
        """Test initialize_bot with a provided file path."""
        # Save sample data to a parquet file
        file_path = tmp_path / "test_data.parquet"
        sample_weekly_returns.to_parquet(file_path)

        # Clear the cache before testing
        initialize_bot.cache_clear()

        bot = initialize_bot(file=str(file_path))

        assert isinstance(bot, _TradeBot)
        assert len(bot.tickers) == 4
        assert len(bot.names) == 4

    def test_initialize_bot_with_path_object(self, tmp_path, sample_weekly_returns):
        """Test initialize_bot with a Path object."""
        file_path = tmp_path / "test_data2.parquet"
        sample_weekly_returns.to_parquet(file_path)

        # Clear the cache
        initialize_bot.cache_clear()

        bot = initialize_bot(file=file_path)

        assert isinstance(bot, _TradeBot)


class TestTradeBot:
    """Tests for the _TradeBot class."""

    def test_init(self, trade_bot):
        """Test TradeBot initialization."""
        assert len(trade_bot.tickers) == 4
        assert len(trade_bot.names) == 4
        assert trade_bot.min_date is not None
        assert trade_bot.max_date is not None

    def test_get_stat(self, trade_bot):
        """Test get_stat method."""
        result = trade_bot.get_stat(trade_bot.min_date, trade_bot.max_date)

        assert isinstance(result, pd.DataFrame)
        assert "Average Annual Returns" in result.columns
        assert "Standard Deviation of Returns" in result.columns
        assert "Sharpe Ratio" in result.columns
        assert "ISIN" in result.columns
        assert "Name" in result.columns
        assert "Size" in result.columns
        assert "Type" in result.columns

    def test_get_top_performing_assets(self, trade_bot):
        """Test get_top_performing_assets method."""
        mid_date = str(trade_bot.weeklyReturns.index[100].date())

        result = trade_bot.get_top_performing_assets(
            time_periods=[(trade_bot.min_date, mid_date), (mid_date, trade_bot.max_date)],
            top_percent=0.5,
        )

        assert isinstance(result, list)

    def test_plot_dots_basic(self, trade_bot):
        """Test plot_dots method with basic parameters."""
        fig = trade_bot.plot_dots(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
        )

        assert fig is not None

    def test_plot_dots_risk_level_boundary(self):
        """Test plot_dots with data to trigger risk level boundary condition (line 555)."""
        # Create data where max std dev is below 0.25 (risk class 7 boundary)
        # Weekly std = annual_std / sqrt(52)
        # To get annual std of ~0.20 (below 0.25), use weekly std of ~0.028
        dates = pd.date_range("2020-01-01", periods=200, freq="W")
        np.random.seed(42)
        # Create assets with volatility levels all below risk class 7
        # max annual std will be around 0.20, so max(actual_risk_level) will be < 7
        data = pd.DataFrame(
            {
                ("ISIN1", "Low Vol Asset"): np.random.normal(0.001, 0.005, 200),  # ~2.5% annual std
                ("ISIN2", "Mid Vol Asset"): np.random.normal(0.001, 0.015, 200),  # ~7.5% annual std
                ("ISIN3", "Higher Vol Asset"): np.random.normal(0.001, 0.025, 200),  # ~12.5% annual std
            },
            index=dates,
        )
        data.columns = pd.MultiIndex.from_tuples(data.columns)

        tickers = [col[0] for col in data.columns]
        names = [col[1] for col in data.columns]
        weekly_returns = data.copy()
        weekly_returns.columns = tickers
        bot = _TradeBot(tickers, names, weekly_returns)

        fig = bot.plot_dots(
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
        )

        assert fig is not None

    def test_plot_dots_with_mst(self, trade_bot):
        """Test plot_dots method with MST highlighting."""
        fig = trade_bot.plot_dots(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            ml="MST",
            ml_subset=["ISIN1", "ISIN2"],
        )

        assert fig is not None

    def test_plot_dots_with_clustering(self, trade_bot):
        """Test plot_dots method with Clustering highlighting."""
        # Create a mock clustering DataFrame
        cluster_df = pd.DataFrame(
            {"Cluster": ["Cluster 1", "Cluster 2", "Cluster 1", "Cluster 2"]}, index=trade_bot.tickers
        )

        fig = trade_bot.plot_dots(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            ml="Clustering",
            ml_subset=cluster_df,
        )

        assert fig is not None

    def test_plot_dots_with_fund_set(self, trade_bot):
        """Test plot_dots method with fund_set parameter."""
        fig = trade_bot.plot_dots(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            fund_set=["Asset A"],
        )

        assert fig is not None

    def test_plot_dots_with_top_performers(self, trade_bot):
        """Test plot_dots method with top_performers parameter."""
        fig = trade_bot.plot_dots(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            top_performers=["Asset A"],
        )

        assert fig is not None

    def test_plot_dots_with_optimal_portfolio(self, trade_bot):
        """Test plot_dots method with optimal_portfolio parameter."""
        # The optimal_portfolio needs to match DataFrame columns:
        # [Average Annual Returns, Standard Deviation of Returns, Sharpe Ratio, ISIN, Name, Size, Type]
        # Index is optimal_portfolio[4] (ISIN column position when inserted)
        optimal_portfolio = [0.1, 0.05, 2.0, "OPT", "Optimal Portfolio", 3, "Optimal Portfolio"]

        fig = trade_bot.plot_dots(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            optimal_portfolio=optimal_portfolio,
        )

        assert fig is not None

    def test_plot_dots_with_benchmark(self, trade_bot):
        """Test plot_dots method with benchmark parameter."""
        # The benchmark needs to match DataFrame columns:
        # [Average Annual Returns, Standard Deviation of Returns, Sharpe Ratio, ISIN, Name, Size, Type]
        benchmark = [0.08, 0.04, 2.0, "BENCH", "Benchmark Portfolio", 3, "Benchmark Portfolio"]

        fig = trade_bot.plot_dots(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            benchmark=benchmark,
        )

        assert fig is not None

    def test_mst_basic(self, trade_bot):
        """Test mst method."""
        fig, subset = trade_bot.mst(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            n_mst_runs=1,
            plot=False,
        )

        assert fig is None  # No plot requested
        assert isinstance(subset, list)

    def test_mst_with_plot(self, trade_bot):
        """Test mst method with plotting."""
        fig, subset = trade_bot.mst(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            n_mst_runs=1,
            plot=True,
        )

        # If subset is not empty, fig should not be None
        if len(subset) > 0:
            assert fig is not None

    def test_clustering_basic(self, trade_bot):
        """Test clustering method."""
        fig, subset = trade_bot.clustering(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            n_clusters=2,
            n_assets=2,
            plot=False,
        )

        assert fig is None  # No plot requested
        assert isinstance(subset, list)

    def test_clustering_with_plot(self, trade_bot):
        """Test clustering method with plotting."""
        fig, subset = trade_bot.clustering(
            start_date=trade_bot.min_date,
            end_date=trade_bot.max_date,
            n_clusters=2,
            n_assets=2,
            plot=True,
        )

        assert fig is not None
        assert isinstance(subset, list)


class TestPlotBacktest:
    """Tests for the __plot_backtest static method."""

    def test_plot_backtest_basic(self, trade_bot):
        """Test the __plot_backtest method."""
        # Create sample performance data
        dates = pd.date_range("2021-01-01", periods=10, freq="W")
        performance = pd.DataFrame({"Portfolio_Value": np.linspace(100, 120, 10)}, index=dates)
        performance_benchmark = pd.DataFrame({"Benchmark_Value": np.linspace(100, 115, 10)}, index=dates)

        # Create sample composition data
        composition = pd.DataFrame(
            {"ISIN1": [0.5, 0.4, 0.3], "ISIN2": [0.3, 0.4, 0.5], "ISIN3": [0.2, 0.2, 0.2]}, index=[0, 1, 2]
        )

        fig_performance, fig_composition = _TradeBot._TradeBot__plot_backtest(
            performance=performance,
            performance_benchmark=performance_benchmark,
            composition=composition,
            names=trade_bot.names,
            tickers=trade_bot.tickers,
        )

        assert fig_performance is not None
        assert fig_composition is not None

    def test_plot_backtest_old_data_format(self, trade_bot):
        """Test the __plot_backtest method with old data format (triggers exception handling)."""
        # Create sample performance data with incompatible index
        dates = pd.date_range("2021-01-01", periods=10, freq="W")
        performance = pd.DataFrame({"Portfolio_Value": np.linspace(100, 120, 10)}, index=dates)

        # Create benchmark with different date format to trigger exception
        benchmark_dates = [d.date() for d in dates]
        performance_benchmark = pd.DataFrame({"Benchmark_Value": np.linspace(100, 115, 10)}, index=benchmark_dates)

        # Create sample composition data
        composition = pd.DataFrame(
            {"ISIN1": [0.5, 0.4, 0.3], "ISIN2": [0.3, 0.4, 0.5], "ISIN3": [0.2, 0.2, 0.2]}, index=[0, 1, 2]
        )

        fig_performance, fig_composition = _TradeBot._TradeBot__plot_backtest(
            performance=performance,
            performance_benchmark=performance_benchmark,
            composition=composition,
            names=trade_bot.names,
            tickers=trade_bot.tickers,
        )

        assert fig_performance is not None
        assert fig_composition is not None


class TestPlotPortfolioDensities:
    """Tests for the __plot_portfolio_densities static method."""

    def test_plot_portfolio_densities(self, trade_bot):
        """Test the __plot_portfolio_densities method."""
        # Create sample portfolio performance data
        np.random.seed(42)
        portfolio_performance_dict = {
            "Conservative": pd.DataFrame({"Terminal Wealth": np.random.normal(100000, 10000, 100)}),
            "Aggressive": pd.DataFrame({"Terminal Wealth": np.random.normal(120000, 20000, 100)}),
        }

        # Create sample composition data
        compositions = {
            "Conservative": pd.DataFrame({"ISIN1": [0.2, 0.2], "ISIN2": [0.3, 0.3], "Cash": [0.5, 0.5]}, index=[0, 1]),
            "Aggressive": pd.DataFrame({"ISIN1": [0.5, 0.4], "ISIN2": [0.4, 0.5], "Cash": [0.1, 0.1]}, index=[0, 1]),
        }

        fig, composition_figures, fig_subplots = _TradeBot._TradeBot__plot_portfolio_densities(
            portfolio_performance_dict=portfolio_performance_dict,
            compositions=compositions,
            tickers=trade_bot.tickers,
            names=trade_bot.names,
        )

        assert fig is not None
        assert isinstance(composition_figures, dict)
        assert fig_subplots is not None

    def test_plot_portfolio_densities_with_reverse(self, trade_bot):
        """Test __plot_portfolio_densities filtering out 'reverse' compositions."""
        np.random.seed(42)
        portfolio_performance_dict = {
            "Conservative": pd.DataFrame({"Terminal Wealth": np.random.normal(100000, 10000, 100)}),
        }

        compositions = {
            "Conservative": pd.DataFrame({"ISIN1": [0.5], "ISIN2": [0.3], "Cash": [0.2]}, index=[0]),
            "Conservative_reverse": pd.DataFrame({"ISIN1": [0.2], "ISIN2": [0.3], "Cash": [0.5]}, index=[0]),
        }

        fig, composition_figures, _fig_subplots = _TradeBot._TradeBot__plot_portfolio_densities(
            portfolio_performance_dict=portfolio_performance_dict,
            compositions=compositions,
            tickers=trade_bot.tickers,
            names=trade_bot.names,
        )

        assert fig is not None
        # "reverse" should be filtered out
        assert "Conservative_reverse" not in composition_figures


class TestBacktest:
    """Tests for the backtest method."""

    def test_backtest_markowitz(self, trade_bot):
        """Test backtest method with Markowitz model."""
        # Use shorter date ranges for faster testing
        mid_date = str(trade_bot.weeklyReturns.index[50].date())
        test_start = str(trade_bot.weeklyReturns.index[100].date())
        test_end = str(trade_bot.weeklyReturns.index[140].date())

        result = trade_bot.backtest(
            start_train_date=mid_date,
            start_test_date=test_start,
            end_test_date=test_end,
            subset_of_assets=["ISIN1", "ISIN2", "ISIN3"],
            benchmarks=["Benchmark Fund"],
            scenarios_type="MonteCarlo",
            n_simulations=50,
            model="Markowitz model",
            solver="CLARABEL",
            lower_bound=0,
        )

        optimal_stat, benchmark_stat, fig_performance, fig_composition = result
        assert optimal_stat is not None
        assert benchmark_stat is not None
        assert fig_performance is not None
        assert fig_composition is not None

    def test_backtest_cvar(self, trade_bot):
        """Test backtest method with CVaR model."""
        mid_date = str(trade_bot.weeklyReturns.index[50].date())
        test_start = str(trade_bot.weeklyReturns.index[100].date())
        test_end = str(trade_bot.weeklyReturns.index[140].date())

        result = trade_bot.backtest(
            start_train_date=mid_date,
            start_test_date=test_start,
            end_test_date=test_end,
            subset_of_assets=["ISIN1", "ISIN2", "ISIN3"],
            benchmarks=["Benchmark Fund"],
            scenarios_type="Bootstrapping",
            n_simulations=50,
            model="CVaR model",
            solver="CLARABEL",
            lower_bound=0,
        )

        optimal_stat, _benchmark_stat, _fig_performance, _fig_composition = result
        assert optimal_stat is not None


class TestLifecycleScenarioAnalysis:
    """Tests for the lifecycle_scenario_analysis method."""

    def test_lifecycle_montecarlo(self, trade_bot):
        """Test lifecycle_scenario_analysis with MonteCarlo scenarios."""
        # Need enough simulations to have variance in terminal wealth
        result = trade_bot.lifecycle_scenario_analysis(
            subset_of_assets=["ISIN1", "ISIN2", "ISIN3"],
            scenarios_type="MonteCarlo",
            n_simulations=100,  # More simulations for variance in terminal wealth
            end_year=2026,
            withdrawals=100,  # Smaller withdrawals relative to budget
            initial_risk_appetite=0.15,
            initial_budget=10000,
            rng_seed=123,  # Different seed for more variation
            test_split=0,
        )

        (
            terminal_wealth_dict,
            exhibition_summary,
            fig_performance,
            fig_glidepaths,
            _allocation_targets,
            _fig_compositions,
            _fig_compositions_all,
        ) = result

        assert isinstance(terminal_wealth_dict, dict)
        assert isinstance(exhibition_summary, pd.DataFrame)
        assert fig_performance is not None
        assert fig_glidepaths is not None

    def test_lifecycle_bootstrap(self, trade_bot):
        """Test lifecycle_scenario_analysis with Bootstrap scenarios."""
        result = trade_bot.lifecycle_scenario_analysis(
            subset_of_assets=["ISIN1", "ISIN2", "ISIN3"],
            scenarios_type="Bootstrap",
            n_simulations=100,  # More simulations for variance
            end_year=2026,
            withdrawals=100,
            initial_risk_appetite=0.15,
            initial_budget=10000,
            rng_seed=456,
            test_split=0,
        )

        terminal_wealth_dict, _exhibition_summary, _, _, _, _, _ = result

        assert isinstance(terminal_wealth_dict, dict)

    def test_lifecycle_with_test_split(self, trade_bot):
        """Test lifecycle_scenario_analysis with test_split."""
        result = trade_bot.lifecycle_scenario_analysis(
            subset_of_assets=["ISIN1", "ISIN2", "ISIN3"],
            scenarios_type="MonteCarlo",
            n_simulations=20,
            end_year=2026,
            withdrawals=1000,
            initial_risk_appetite=0.15,
            initial_budget=10000,
            rng_seed=42,
            test_split=0.5,
        )

        terminal_wealth_dict, _exhibition_summary, _, _, _, _, _ = result

        assert isinstance(terminal_wealth_dict, dict)

    def test_lifecycle_invalid_scenario_type(self, trade_bot):
        """Test lifecycle_scenario_analysis with invalid scenario type."""
        with pytest.raises(ValueError, match="scenario method"):
            trade_bot.lifecycle_scenario_analysis(
                subset_of_assets=["ISIN1", "ISIN2", "ISIN3"],
                scenarios_type="InvalidType",
                n_simulations=20,
                end_year=2026,
                withdrawals=1000,
                initial_risk_appetite=0.15,
                initial_budget=10000,
            )

    def test_lifecycle_no_rng_seed(self, trade_bot):
        """Test lifecycle_scenario_analysis without rng_seed (rng_seed=0 branch)."""
        # Mock the plotting to avoid gaussian_kde issues with low-variance data
        with patch.object(_TradeBot, "_TradeBot__plot_portfolio_densities") as mock_plot:
            # Return mock figures
            mock_plot.return_value = (MagicMock(), {}, MagicMock())

            result = trade_bot.lifecycle_scenario_analysis(
                subset_of_assets=["ISIN1", "ISIN2", "ISIN3"],
                scenarios_type="MonteCarlo",
                n_simulations=50,
                end_year=2026,
                withdrawals=100,
                initial_risk_appetite=0.15,
                initial_budget=10000,
                rng_seed=0,  # No seed - this triggers the specific branch
                test_split=0,
            )

            assert result is not None
            # Verify the rng_seed=0 path was taken (plot was called)
            mock_plot.assert_called_once()
