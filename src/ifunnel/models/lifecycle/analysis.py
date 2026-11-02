"""Lifecycle investment simulation along several glide paths.

The scenario analysis behind :meth:`ifunnel.TradeBot.lifecycle_scenario_analysis`:
moment estimation, scenario generation, glide-path target allocations and the
per-path simulation with withdrawals. Figures of the results are left to the caller.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from ..scenario_generation import MomentGenerator, ScenarioGenerator, resolve_scenarios_type
from .glide_path_creator import generate_risk_profiles
from .mvo_lifecycle_model import get_port_allocations, resolve_start_year, riskadjust_model_scen


def run_lifecycle_analysis(
    weekly_returns: pd.DataFrame,
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
) -> tuple[dict, pd.DataFrame, go.Figure, dict]:
    """Simulate every glide path's target allocations over the return scenarios.

    See :meth:`ifunnel.TradeBot.lifecycle_scenario_analysis` for the meaning of the
    arguments, which are passed through unchanged; ``weekly_returns`` is the bot's data.

    Returns:
        tuple: ``(terminal_wealth, summary, fig_glidepaths, allocation_targets)``: per
            glide path, the scenario results (dict keyed by glide path name); a
            DataFrame of analysis metrics with one column per glide path; the glide
            path figure; and the target allocations (dict keyed by glide path name).

    Raises:
        ValueError: If ``scenarios_type`` is not an accepted value, or ``end_year``
            is not after ``start_year``.
    """
    # ------------------------------- INITIALIZE FUNCTION -------------------------------
    scenarios_type = resolve_scenarios_type(scenarios_type)
    start_year = resolve_start_year(start_year)
    n_periods = end_year - start_year
    if n_periods < 1:
        raise ValueError(f"end_year ({end_year}) must be after start_year ({start_year})")  # noqa: TRY003
    withdrawal_lst = [withdrawals * (1 + 0.04) ** i for i in range(n_periods)]

    # ------------------------------- PARAMETER INITIALIZATION -------------------------------
    if test_split != 0:
        sampling_set, estimating_set = MomentGenerator.split_dataset(
            data=weekly_returns[subset_of_assets], sampling_ratio=test_split
        )

        _, _, sigma_weekly, mu_weekly = MomentGenerator.generate_annual_sigma_mu_with_risk_free(data=sampling_set)

        sigma, mu, _, _ = MomentGenerator.generate_annual_sigma_mu_with_risk_free(data=estimating_set)
    else:
        sigma, mu, sigma_weekly, mu_weekly = MomentGenerator.generate_annual_sigma_mu_with_risk_free(
            data=weekly_returns[subset_of_assets]
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

    else:  # "Bootstrap"
        scenarios = sg.bootstrap_simulation_annual_from_weekly(
            historical_weekly_returns=weekly_returns[subset_of_assets],
            n_simulations=n_simulations,
            n_years=n_periods,
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
            start_year=start_year,
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
            start_year=start_year,
        )

        # Add the analysis_metrics DataFrame as a new column in the storage DataFrame
        exhibition_summary[key] = analysis_metrics.squeeze()

        portfolio_df["Terminal Wealth"] = pd.to_numeric(portfolio_df["Terminal Wealth"], errors="coerce")
        terminal_wealth_dict[f"{key}"] = portfolio_df

    return terminal_wealth_dict, exhibition_summary, fig_glidepaths, allocation_targets
