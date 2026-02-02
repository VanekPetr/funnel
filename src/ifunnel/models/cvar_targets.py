"""Module for Conditional Value at Risk (CVaR) target generation and calculation.

This module provides functions for calculating CVaR using the primal formula,
computing portfolio risk targets, and generating CVaR targets for portfolio optimization.
CVaR is a risk measure that quantifies the expected loss in the worst-case scenarios.
"""

import numpy as np
import pandas as pd
from loguru import logger

from .scenario_generation import ScenarioGenerator


# Primal CVaR formula
def cvar(alpha: float, p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Computes Conditional Value at Risk (CVaR) using the primal formula.

    This function calculates both Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    for a given confidence level alpha and distribution of losses.

    Args:
        alpha: Confidence level, typically between 0.9 and 0.99.
        p: Array of probabilities for each scenario.
        q: Array of losses for each scenario.

    Returns:
        tuple: A tuple containing:
            - float: Value at Risk (VaR) at the given confidence level.
            - float: Conditional Value at Risk (CVaR) at the given confidence level.

    Note:
        Inputs p and q must be numpy arrays.
    """
    # We need to be careful that math index starts from 1 but numpy starts from 0
    # (matters in formulas like ceil(alpha * T))
    # T = q.shape[0]
    sort_idx = np.argsort(q)
    sorted_q = q[sort_idx]
    sorted_p = p[sort_idx]

    # Starting index
    i_alpha = np.sort(np.nonzero(np.cumsum(sorted_p) >= alpha)[0])[0]

    # Weight of VaR component in CVaR
    lambda_alpha = (np.sum(sorted_p[: (i_alpha + 1)]) - alpha) / (1 - alpha)

    # CVaR
    var = sorted_q[i_alpha]
    cvar = lambda_alpha * sorted_q[i_alpha] + np.dot(sorted_p[(i_alpha + 1) :], sorted_q[(i_alpha + 1) :]) / (1 - alpha)

    return var, cvar


# FUNCTION RUNNING THE OPTIMIZATION
# ----------------------------------------------------------------------
def portfolio_risk_target(scenarios: pd.DataFrame, cvar_alpha: float) -> float:
    """Calculates the CVaR risk target for an equally-weighted portfolio.

    This function computes the Conditional Value at Risk (CVaR) for an equally-weighted
    portfolio based on the provided scenarios. It assumes equal allocation across all
    assets in the portfolio.

    Args:
        scenarios: DataFrame containing return scenarios for each asset.
        cvar_alpha: Confidence level for CVaR calculation, typically between 0.9 and 0.99.

    Returns:
        float: The CVaR value for the equally-weighted portfolio.
    """
    # Fixed equal weight x
    x = pd.Series(index=scenarios.columns, data=1 / scenarios.shape[1])

    # Number of scenarios
    scenario_n = scenarios.shape[0]

    # Portfolio loss scenarios
    losses = (-scenarios @ x).to_numpy()

    # Probabilities
    probs = np.ones(scenario_n) / scenario_n

    # CVaR
    _, portfolio_cvar = cvar(1 - cvar_alpha, probs, losses)

    return portfolio_cvar


# ----------------------------------------------------------------------
# Mathematical Optimization: TARGETS GENERATION
# ----------------------------------------------------------------------
def get_cvar_targets(
    test_date: str,
    benchmark: list,
    budget: int,
    cvar_alpha: float,
    data: pd.DataFrame,
    scgen: ScenarioGenerator,
    n_simulations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates CVaR targets and benchmark portfolio values for optimization.

    This function generates Conditional Value at Risk (CVaR) targets for portfolio
    optimization and calculates the benchmark portfolio values over time. It uses
    bootstrapping to generate scenarios for risk calculation.

    Args:
        test_date: Start date for the testing period.
        benchmark: List of ticker symbols for the benchmark portfolio.
        budget: Initial budget for the portfolio.
        cvar_alpha: Confidence level for CVaR calculation.
        data: DataFrame containing historical returns data.
        scgen: ScenarioGenerator instance for scenario generation.
        n_simulations: Number of scenarios to generate.

    Returns:
        tuple: A tuple containing two DataFrames:
            - targets: DataFrame with CVaR targets for each period.
            - portfolio_value: DataFrame with benchmark portfolio values over time.
    """
    logger.info(f"🎯 Generating CVaR targets for {benchmark}")

    # Define Benchmark
    tickers = benchmark
    # Get weekly return of our benchmark
    whole_dataset_benchmark = data[tickers].copy()

    # Get weekly data just for testing period
    test_dataset_benchmark = whole_dataset_benchmark[whole_dataset_benchmark.index >= test_date]

    # Number of weeks for testing
    weeks_n = len(test_dataset_benchmark.index)

    # Get scenarios
    # The Monte Carlo Method
    target_scenarios = scgen.bootstrapping(
        data=whole_dataset_benchmark,  # subsetMST or subsetCLUST
        n_simulations=n_simulations,
        n_test=weeks_n,
    )

    # Compute the optimal portfolio outperforming zero percentage return
    # ----------------------------------------------------------------------
    p_points = len(target_scenarios[:, 0, 0])  # number of periods
    s_points = len(target_scenarios[0, :, 0])  # number of scenarios

    # COMPUTE CVaR TARGETS
    list_targets = []
    for p in range(p_points):
        # create data frame with scenarios for a given period p
        scenario_df = pd.DataFrame(target_scenarios[p, :, :], columns=tickers, index=list(range(s_points)))

        # run CVaR model to compute CVaR targets
        cvar_target = portfolio_risk_target(scenarios=scenario_df, cvar_alpha=cvar_alpha)
        # save the result
        list_targets.append(cvar_target)

    # Generate new column so that dtype is set right.
    targets = pd.DataFrame(columns=["CVaR_Target"], data=list_targets)

    # COMPUTE PORTFOLIO VALUE
    list_portfolio_values = []
    for w in test_dataset_benchmark.index:
        budget_next = sum((budget / len(tickers)) * (1 + test_dataset_benchmark.loc[w, :]))
        list_portfolio_values.append(budget_next)
        budget = budget_next

    # Generate dataframe so that dtype is set right.
    portfolio_value = pd.DataFrame(
        columns=["Benchmark_Value"],
        index=test_dataset_benchmark.index,
        data=list_portfolio_values,
    )

    return targets, portfolio_value
