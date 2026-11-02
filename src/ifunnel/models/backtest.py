"""Backtest of an optimised portfolio against a benchmark.

The rolling-window rebalancing behind :meth:`ifunnel.TradeBot.backtest`: scenario
generation, benchmark-derived risk targets and the Markowitz or CVaR model.
Figures and summary statistics are left to the caller.
"""

import numpy as np
import pandas as pd

from .cvar_model import cvar_model
from .cvar_targets import get_cvar_targets
from .mvo_model import mvo_model
from .mvo_targets import get_mvo_targets
from .scenario_generation import MomentGenerator, ScenarioGenerator, resolve_scenarios_type

# Accepted values of the model switch
BACKTEST_MODELS = ("Markowitz model", "CVaR model")


def run_backtest(
    weekly_returns: pd.DataFrame,
    tickers: list[str],
    names: list[str],
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebalance an optimised portfolio over the test window and value its benchmark.

    See :meth:`ifunnel.TradeBot.backtest` for the meaning of the arguments, which are
    passed through unchanged; ``weekly_returns``, ``tickers`` and ``names`` are the
    bot's data.

    Returns:
        tuple: ``(port_value, benchmark_value, port_allocation)``: the value paths of
            the optimised portfolio and of the benchmark, and the portfolio
            composition at each rebalancing.

    Raises:
        ValueError: If ``model`` or ``scenarios_type`` is not an accepted value,
            or a benchmark name is not in ``names``.
    """
    if model not in BACKTEST_MODELS:
        raise ValueError(f"Unknown model {model!r}; choose one of {BACKTEST_MODELS}")  # noqa: TRY003
    scenarios_type = resolve_scenarios_type(scenarios_type)

    # Find Benchmarks' ISIN codes
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]

    # Get train and testing datasets
    whole_dataset = weekly_returns[
        (weekly_returns.index >= start_train_date) & (weekly_returns.index <= end_test_date)
    ].copy()
    test_dataset = weekly_returns[
        (weekly_returns.index > start_test_date) & (weekly_returns.index <= end_test_date)
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

    return port_value, benchmark_port_val, port_allocation
