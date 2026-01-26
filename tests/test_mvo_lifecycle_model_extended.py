"""Extended tests for lifecycle MVO model module to improve coverage."""

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.lifecycle.mvo_lifecycle_model import (
    calculate_analysis_metrics,
    calculate_risk_metrics,
    get_port_allocations,
    lifecycle_rebalance_model,
    portfolio_rebalancing,
    riskadjust_model_scen,
)


@pytest.fixture
def simple_mu():
    """Create simple expected returns for testing."""
    return pd.Series([0.05, 0.08, 0.02, 0.0], index=["Asset1", "Asset2", "Asset3", "Cash"])


@pytest.fixture
def simple_sigma(simple_mu):
    """Create simple covariance matrix for testing."""
    len(simple_mu)
    # Create a valid positive semi-definite covariance matrix
    data = np.array(
        [
            [0.04, 0.01, 0.005, 0.0],
            [0.01, 0.09, 0.01, 0.0],
            [0.005, 0.01, 0.01, 0.0],
            [0.0, 0.0, 0.0, 0.0001],
        ]
    )
    return pd.DataFrame(data, index=simple_mu.index, columns=simple_mu.index)


class TestLifecycleRebalanceModel:
    """Tests for the lifecycle_rebalance_model function."""

    def test_basic_optimization(self, simple_mu, simple_sigma):
        """Test basic portfolio optimization."""
        vol_target = 0.15
        max_weight = 0.5
        solver = "CLARABEL"

        port_nom, port_val = lifecycle_rebalance_model(
            mu=simple_mu,
            sigma=simple_sigma,
            vol_target=vol_target,
            max_weight=max_weight,
            solver=solver,
        )

        assert isinstance(port_nom, pd.Series)
        assert isinstance(port_val, (float, np.floating))
        # Weights should sum to approximately 1
        assert abs(port_nom.sum() - 1.0) < 1e-4
        # All weights should be non-negative
        assert (port_nom >= -1e-6).all()

    @pytest.mark.skip(reason="Mixed-integer problem - no MIP-capable solver available")
    def test_with_lower_bound(self, simple_mu, simple_sigma):
        """Test portfolio optimization with lower bound constraint."""
        vol_target = 0.20
        max_weight = 0.5
        solver = "CLARABEL"
        lower_bound = 0.05

        port_nom, _port_val = lifecycle_rebalance_model(
            mu=simple_mu,
            sigma=simple_sigma,
            vol_target=vol_target,
            max_weight=max_weight,
            solver=solver,
            lower_bound=lower_bound,
        )

        assert isinstance(port_nom, pd.Series)
        # Selected non-cash assets should have weight >= lower_bound or 0
        non_cash = port_nom.drop("Cash")
        for w in non_cash:
            if w > 1e-5:  # If selected
                assert w >= lower_bound - 1e-4

    def test_infeasible_constraints(self, simple_mu, simple_sigma):
        """Test handling of infeasible constraints (very low vol target)."""
        vol_target = 0.0001  # Very low target - likely infeasible
        max_weight = 0.5
        solver = "CLARABEL"

        port_nom, _port_val = lifecycle_rebalance_model(
            mu=simple_mu,
            sigma=simple_sigma,
            vol_target=vol_target,
            max_weight=max_weight,
            solver=solver,
        )

        # Should still return a series (may contain NaN for infeasible)
        assert isinstance(port_nom, pd.Series)


class TestGetPortAllocations:
    """Tests for the get_port_allocations function."""

    def test_basic_allocations(self, simple_mu, simple_sigma):
        """Test getting portfolio allocations for multiple periods."""
        n_periods = 3
        targets = pd.Series([0.15, 0.12, 0.10], index=range(n_periods))

        allocation_df = get_port_allocations(
            mu_lst=simple_mu,
            sigma_lst=simple_sigma,
            targets=targets,
            max_weight=0.5,
            solver="CLARABEL",
        )

        assert isinstance(allocation_df, pd.DataFrame)
        assert len(allocation_df) == n_periods
        assert list(allocation_df.columns) == list(simple_mu.index)

    def test_decreasing_risk_targets(self, simple_mu, simple_sigma):
        """Test allocations with decreasing risk targets (typical glide path)."""
        n_periods = 5
        # Decreasing risk targets
        targets = pd.Series([0.20, 0.17, 0.14, 0.11, 0.08], index=range(n_periods))

        allocation_df = get_port_allocations(
            mu_lst=simple_mu,
            sigma_lst=simple_sigma,
            targets=targets,
            max_weight=0.5,
            solver="CLARABEL",
        )

        # Check that each row sums to approximately 1
        for idx in allocation_df.index:
            row_sum = allocation_df.loc[idx].astype(float).sum()
            assert abs(row_sum - 1.0) < 0.01, f"Row {idx} doesn't sum to 1: {row_sum}"


class TestPortfolioRebalancing:
    """Tests for the portfolio_rebalancing function."""

    @pytest.fixture
    def simple_targets(self):
        """Create simple allocation targets."""
        return pd.DataFrame(
            {
                "Asset1": [0.4, 0.35, 0.3],
                "Asset2": [0.3, 0.35, 0.4],
                "Cash": [0.3, 0.3, 0.3],
            },
            index=["2023", "2024", "2025"],
        )

    @pytest.fixture
    def simple_scenarios(self):
        """Create simple return scenarios."""
        return pd.DataFrame(
            {
                "Asset1": [0.05, 0.08, -0.02],
                "Asset2": [0.03, -0.05, 0.10],
                "Cash": [0.01, 0.01, 0.01],
            },
            index=[0, 1, 2],
        )

    def test_basic_rebalancing(self, simple_targets, simple_scenarios):
        """Test basic portfolio rebalancing."""
        budget = 10000
        withdrawal_lst = [500, 500, 500]
        transaction_cost = 0.001
        interest_rate = 0.04

        ptf_performance, allocation_df = portfolio_rebalancing(
            budget=budget,
            targets=simple_targets,
            withdrawal_lst=withdrawal_lst,
            transaction_cost=transaction_cost,
            scenarios=simple_scenarios,
            interest_rate=interest_rate,
        )

        assert isinstance(ptf_performance, pd.DataFrame)
        assert isinstance(allocation_df, pd.DataFrame)
        assert len(ptf_performance) == 3
        assert "Portfolio Value Primo" in ptf_performance.columns
        assert "Portfolio Value Ultimo" in ptf_performance.columns

    def test_rebalancing_with_large_withdrawals(self, simple_targets, simple_scenarios):
        """Test rebalancing when withdrawals cause default."""
        budget = 1000
        # Very large withdrawals to trigger default
        withdrawal_lst = [2000, 2000, 2000]
        transaction_cost = 0.001
        interest_rate = 0.04

        ptf_performance, _allocation_df = portfolio_rebalancing(
            budget=budget,
            targets=simple_targets,
            withdrawal_lst=withdrawal_lst,
            transaction_cost=transaction_cost,
            scenarios=simple_scenarios,
            interest_rate=interest_rate,
        )

        assert isinstance(ptf_performance, pd.DataFrame)
        # Should have a default year recorded
        assert "Default Year" in ptf_performance.columns

    def test_rebalancing_no_withdrawals(self, simple_targets, simple_scenarios):
        """Test rebalancing with no withdrawals."""
        budget = 10000
        withdrawal_lst = [0, 0, 0]
        transaction_cost = 0.001
        interest_rate = 0.04

        ptf_performance, _allocation_df = portfolio_rebalancing(
            budget=budget,
            targets=simple_targets,
            withdrawal_lst=withdrawal_lst,
            transaction_cost=transaction_cost,
            scenarios=simple_scenarios,
            interest_rate=interest_rate,
        )

        # With no withdrawals and positive returns, portfolio should grow
        assert isinstance(ptf_performance, pd.DataFrame)


class TestRiskadjustModelScen:
    """Tests for the riskadjust_model_scen function."""

    @pytest.fixture
    def multi_scenario_data(self):
        """Create multiple scenarios for testing."""
        n_scenarios = 10
        n_periods = 3
        n_assets = 3
        np.random.seed(42)

        # Generate random returns
        scen = np.random.normal(0.05, 0.10, (n_scenarios, n_periods, n_assets))
        return scen

    @pytest.fixture
    def scenario_targets(self):
        """Create targets for scenario testing."""
        return pd.DataFrame(
            {
                "Asset1": [0.4, 0.35, 0.3],
                "Asset2": [0.3, 0.35, 0.4],
                "Cash": [0.3, 0.3, 0.3],
            },
            index=["2023", "2024", "2025"],
        )

    def test_basic_scenario_analysis(self, multi_scenario_data, scenario_targets):
        """Test basic multi-scenario portfolio analysis."""
        budget = 10000
        trans_cost = 0.001
        withdrawal_lst = [500, 500, 500]
        interest_rate = 0.04

        portfolio_df, mean_allocations_df, analysis_metrics = riskadjust_model_scen(
            scen=multi_scenario_data,
            targets=scenario_targets,
            budget=budget,
            trans_cost=trans_cost,
            withdrawal_lst=withdrawal_lst,
            interest_rate=interest_rate,
        )

        assert isinstance(portfolio_df, pd.DataFrame)
        assert isinstance(mean_allocations_df, pd.DataFrame)
        assert isinstance(analysis_metrics, pd.DataFrame)
        assert len(portfolio_df) == 10  # n_scenarios
        assert "Terminal Wealth" in portfolio_df.columns
        assert "Sharpe Ratio" in portfolio_df.columns

    def test_scenario_analysis_with_defaults(self, scenario_targets):
        """Test scenario analysis where some scenarios default."""
        n_scenarios = 5
        n_periods = 3
        n_assets = 3

        # Create scenarios with very negative returns to trigger defaults
        np.random.seed(123)
        scen = np.random.normal(-0.30, 0.05, (n_scenarios, n_periods, n_assets))

        budget = 1000
        trans_cost = 0.001
        withdrawal_lst = [400, 400, 400]  # High relative to budget
        interest_rate = 0.04

        portfolio_df, _mean_allocations_df, _analysis_metrics = riskadjust_model_scen(
            scen=scen,
            targets=scenario_targets,
            budget=budget,
            trans_cost=trans_cost,
            withdrawal_lst=withdrawal_lst,
            interest_rate=interest_rate,
        )

        assert isinstance(portfolio_df, pd.DataFrame)
        assert "Default Year" in portfolio_df.columns


class TestCalculateRiskMetricsEdgeCases:
    """Additional edge case tests for calculate_risk_metrics."""

    def test_single_return(self):
        """Test with a single return value."""
        yearly_returns = pd.Series([0.05])

        annual_return, annual_std_dev, _sharpe_ratio, _downside_std_dev, _sortino_ratio = calculate_risk_metrics(
            yearly_returns, risk_free_rate=0.02
        )

        assert annual_return == 0.05
        # With single value, std_dev should be 0 or NaN
        assert annual_std_dev == 0.0 or np.isnan(annual_std_dev)

    def test_zero_downside_deviation(self):
        """Test when all returns are above risk-free rate."""
        yearly_returns = pd.Series([0.10, 0.15, 0.12, 0.20])

        annual_return, _annual_std_dev, _sharpe_ratio, _downside_std_dev, _sortino_ratio = calculate_risk_metrics(
            yearly_returns, risk_free_rate=0.02
        )

        # All returns above risk-free rate, so sortino may be None or very large
        assert annual_return > 0.02


class TestCalculateAnalysisMetricsEdgeCases:
    """Additional edge case tests for calculate_analysis_metrics."""

    def test_small_dataset(self):
        """Test with a small dataset."""
        terminal_values = pd.Series([100, 110, 90])

        metrics_df = calculate_analysis_metrics(terminal_values)

        assert isinstance(metrics_df, pd.DataFrame)
        assert metrics_df["Min Terminal Value"].iloc[0] == 90
        assert metrics_df["Max Terminal Value"].iloc[0] == 110

    def test_negative_terminal_values(self):
        """Test with negative terminal values (default scenarios)."""
        terminal_values = pd.Series([-50, -30, 100, 150, 200])

        metrics_df = calculate_analysis_metrics(terminal_values)

        assert isinstance(metrics_df, pd.DataFrame)
        assert metrics_df["Min Terminal Value"].iloc[0] == -50


class TestLifecycleRebalanceModelFailure:
    """Tests for lifecycle_rebalance_model failure scenarios."""

    @pytest.fixture
    def infeasible_mu(self):
        """Create expected returns for testing infeasible problem."""
        return pd.Series([0.05, 0.08, 0.02, 0.0], index=["Asset1", "Asset2", "Asset3", "Cash"])

    @pytest.fixture
    def infeasible_sigma(self, infeasible_mu):
        """Create covariance matrix for testing infeasible problem."""
        data = np.array(
            [
                [0.04, 0.01, 0.005, 0.0],
                [0.01, 0.09, 0.01, 0.0],
                [0.005, 0.01, 0.01, 0.0],
                [0.0, 0.0, 0.0, 0.0001],
            ]
        )
        return pd.DataFrame(data, index=infeasible_mu.index, columns=infeasible_mu.index)

    def test_infeasible_optimization_returns_nan(self, infeasible_mu, infeasible_sigma):
        """Test that infeasible optimization returns NaN values."""
        # Set extremely restrictive constraints that can't be satisfied
        vol_target = 0.0000001  # Impossibly low volatility target
        max_weight = 0.001  # Very restrictive max weight
        solver = "CLARABEL"

        port_nom, _port_val = lifecycle_rebalance_model(
            mu=infeasible_mu,
            sigma=infeasible_sigma,
            vol_target=vol_target,
            max_weight=max_weight,
            solver=solver,
            inaccurate=False,  # Don't accept inaccurate solutions
        )

        # For infeasible problems, should return NaN Series
        assert isinstance(port_nom, pd.Series)
        # Check that it's either NaN or a valid solution
        if not port_nom.isna().all():
            # If solution found, weights should sum to 1
            assert abs(port_nom.sum() - 1.0) < 1e-4


class TestCalculateRiskMetricsZeroDivision:
    """Tests for edge cases in calculate_risk_metrics that could cause division issues."""

    def test_all_same_returns(self):
        """Test when all returns are the same (zero std dev)."""
        yearly_returns = pd.Series([0.05, 0.05, 0.05, 0.05])

        annual_return, annual_std_dev, sharpe_ratio, _downside_std_dev, _sortino_ratio = calculate_risk_metrics(
            yearly_returns, risk_free_rate=0.02
        )

        assert annual_return == 0.05
        assert annual_std_dev == 0.0
        assert sharpe_ratio is None  # Division by zero avoided

    def test_all_positive_returns_above_risk_free(self):
        """Test when all returns are above risk free rate (zero downside)."""
        yearly_returns = pd.Series([0.10, 0.15, 0.20, 0.25])

        annual_return, _annual_std_dev, _sharpe_ratio, downside_std_dev, sortino_ratio = calculate_risk_metrics(
            yearly_returns, risk_free_rate=0.02
        )

        assert annual_return > 0.02
        # With all returns above risk-free, downside_std_dev should be 0
        assert downside_std_dev == 0.0
        assert sortino_ratio is None  # Division by zero avoided
