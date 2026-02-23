"""Module for Mean-Variance Optimization (MVO) portfolio modeling.

This module provides functions for portfolio optimization using the Mean-Variance
Optimization approach. It includes methods for computing Cholesky decomposition,
rebalancing portfolios with transaction costs, and running the MVO model over
multiple periods to generate optimal portfolio allocations.
"""

import pickle  # nosec B403

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
from cvxpy import norm as cp_norm  # type: ignore[attr-defined]
from cvxpy import sum as cp_sum  # type: ignore[attr-defined]
from loguru import logger


def cholesky_psd(m):
    """Computes the Cholesky decomposition for positive semidefinite matrices.

    This function performs a Cholesky decomposition on matrices that may not be
    positive definite, but are at least positive semidefinite. It uses LDL decomposition
    and applies a fix for non-positive eigenvalues if necessary.

    Args:
        m: Input matrix to decompose, should be symmetric and positive semidefinite.

    Returns:
        numpy.ndarray: The Cholesky factor C such that C.T @ C approximates the input matrix.

    Raises:
        AssertionError: If the D matrix from LDL decomposition is not diagonal.
    """
    lu, d, _perm = sp.linalg.ldl(m)
    if np.max(np.abs(d - np.diag(np.diag(d)))) >= 1e-12:
        raise ValueError("Matrix 'd' is not diagonal!")  # noqa: TRY003

    # Do non-negativity fix
    min_eig = np.min(np.diag(d))
    if min_eig < 0:
        d -= 5 * min_eig * np.eye(*d.shape)

    sqrtd = sp.linalg.sqrtm(d)
    c = (lu @ sqrtd).T
    return c


# ----------------------------------------------------------------------
# MODEL FOR OPTIMIZING THE BACKTEST PERIODS
# ----------------------------------------------------------------------
def rebalancing_model(
    mu,
    covariance,
    vty_target,
    cash,
    x_old,
    trans_cost,
    max_weight,
    solver,
    inaccurate,
    lower_bound,
):
    """Finds the optimal enhanced index portfolio according to a benchmark.

    This function optimizes a portfolio to maximize expected return while respecting
    volatility constraints and transaction costs. The portfolio corresponds to the
    tangency portfolio where risk is evaluated according to the volatility of the
    tracking error. The model is formulated using quadratic programming.

    Args:
        mu: pandas.Series with float values representing asset point forecasts.
        covariance: pandas.DataFrame with asset covariances.
        vty_target: Target volatility for the optimal portfolio.
        cash: Additional budget for the portfolio.
        x_old: Previous portfolio allocation.
        trans_cost: Transaction costs as a fraction of traded value.
        max_weight: Maximum allowed weight for any single asset.
        solver: The name of the solver to use, as returned by cvxpy.installed_solvers().
        inaccurate: Whether to also accept solutions with status "optimal_inaccurate".
        lower_bound: Minimum weight given to each selected asset.

    Returns:
        tuple: A tuple containing:
            - pd.Series: Optimal portfolio weights.
            - float: Resulting portfolio volatility.
            - float: Portfolio value.
            - float: Remaining cash.

    Raises:
        Exception: If the solver does not find an optimal solution.
    """
    # Number of assets
    n = covariance.shape[1]

    # Variable transaction costs
    c = trans_cost

    # Factorize the covariance
    # G = cholesky_psd(covariance)
    g = np.linalg.cholesky(covariance)

    # Define variables
    # - portfolio
    x = cp.Variable(n, name="x", nonneg=True)
    # - |x - x_old|
    absdiff = cp.Variable(n, name="absdiff", nonneg=True)
    # - cost
    cost = cp.Variable(name="cost", nonneg=True)

    # Define objective (max expected portfolio return)
    objective = cp.Maximize(mu.to_numpy() @ x)

    # Define constraints
    constraints = [
        # - Volatility limit
        cp_norm(g @ x, 2) <= vty_target,
        # - Cost of rebalancing
        c * cp_sum(absdiff) == cost,
        x - x_old <= absdiff,
        x - x_old >= -absdiff,
        # - Budget
        x_old.sum() + cash - cp_sum(x) - cost == 0,
        # - Concentration limits
        x <= max_weight * cp_sum(x),
    ]

    if lower_bound != 0:  # pragma: no cover (requires MIP solver)
        z = cp.Variable(n, boolean=True)  # Binary variable indicates if asset is selected
        upper_bound = 100

        constraints.append(lower_bound * z <= x)
        constraints.append(x <= upper_bound * z)
        constraints.append(cp_sum(z) >= 1)

    # Define model
    model = cp.Problem(objective=objective, constraints=constraints)

    # Solve
    model.solve(solver=solver, verbose=False)

    # Get positions
    accepted_statuses = ["optimal"]
    if inaccurate:
        accepted_statuses.append("optimal_inaccurate")
    if model.status in accepted_statuses:
        opt_port = pd.Series(x.value, index=mu.index)

        # Set floating data points to zero and normalize
        opt_port[np.abs(opt_port) < 0.000001] = 0
        port_val = np.sum(opt_port)
        vty_result_p = np.linalg.norm(g @ x.value, 2) / port_val
        opt_port = opt_port / port_val

        # Remaining cash
        cash = cash - (port_val + cost.value - x_old.sum())

        # return portfolio, CVaR, and alpha
        return opt_port, vty_result_p, port_val, cash

    else:
        # Save inputs, so that failing problem can be investigated separately e.g. in a notebook
        inputs = {
            "mu": mu,
            "covariance": covariance,
            "vty_target": vty_target,
            "cash": cash,
            "x_old": x_old,
            "trans_cost": trans_cost,
            "max_weight": max_weight,
            "lower_bound": lower_bound,
        }
        with open("rebalance_inputs.pkl", "wb") as file:
            pickle.dump(inputs, file)

        # Print an error if the model is not optimal
        logger.exception(f"❌ Solver does not find optimal solution. Status code is {model.status}")


# ----------------------------------------------------------------------
# Mathematical Optimization: RUN THE MVO MODEL
# ----------------------------------------------------------------------
def mvo_model(
    test_ret: pd.DataFrame,
    mu_lst: list,
    sigma_lst: list,
    targets: pd.DataFrame,
    budget: float,
    trans_cost: float,
    max_weight: float,
    solver: str,
    lower_bound: int,
    inaccurate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs the Mean-Variance Optimization model over multiple periods.

    This function executes the MVO model for a sequence of investment periods,
    rebalancing the portfolio at each period based on updated forecasts and targets.
    It tracks portfolio allocations, values, and volatility over time.

    Args:
        test_ret: DataFrame containing asset returns for the test period.
        mu_lst: List of expected return vectors (pandas.Series) for each period.
        sigma_lst: List of covariance matrices (pandas.DataFrame) for each period.
        targets: DataFrame containing volatility targets for each period.
        budget: Initial budget for the portfolio.
        trans_cost: Transaction costs as a fraction of traded value.
        max_weight: Maximum allowed weight for any single asset.
        solver: The name of the solver to use for optimization.
        lower_bound: Minimum weight given to each selected asset.
        inaccurate: Whether to accept solutions with status "optimal_inaccurate".

    Returns:
        tuple: A tuple containing three DataFrames:
            - portfolio_allocation: Portfolio weights for each period.
            - portfolio_value: Portfolio values over time.
            - portfolio_vty: Portfolio volatility for each period.
    """
    p_points = len(mu_lst)  # number of periods

    assets = test_ret.columns  # names of all assets

    # LIST TO STORE VOLATILITY TARGETS
    list_portfolio_vty = []
    # LIST TO STORE VALUE OF THE PORTFOLIO
    list_portfolio_value = []
    # LIST TO STORE PORTFOLIO ALLOCATION
    list_portfolio_allocation = []

    x_old = pd.Series(0, index=assets)
    cash = budget
    portfolio_value_w = budget

    logger.debug(f"🤖 Selected solver is {solver}")
    for p in range(p_points):
        logger.info(f"🚀 Optimizing period {p + 1} out of {p_points}.")

        # Get MVO parameters
        mu = mu_lst[p]
        sigma = sigma_lst[p]

        # run CVaR model
        p_alloc, vty_val, port_val, cash = rebalancing_model(
            mu=mu,
            covariance=sigma,
            vty_target=targets.loc[p, "Vty_Target"] * portfolio_value_w,
            cash=cash,
            x_old=x_old,
            trans_cost=trans_cost,
            max_weight=max_weight,
            solver=solver,
            inaccurate=inaccurate,
            lower_bound=lower_bound,
        )

        # save the result
        list_portfolio_vty.append(vty_val)
        # save allocation
        list_portfolio_allocation.append(p_alloc)

        portfolio_value_w = port_val
        # COMPUTE PORTFOLIO VALUE
        for w in test_ret.index[(p * 4) : (4 + p * 4)]:
            portfolio_value_w = sum(p_alloc * portfolio_value_w * (1 + test_ret.loc[w, assets]))
            list_portfolio_value.append((w, portfolio_value_w))

        x_old = p_alloc * portfolio_value_w

    portfolio_vty = pd.DataFrame(columns=["Volatility"], data=list_portfolio_vty)
    portfolio_value = pd.DataFrame(columns=["Date", "Portfolio_Value"], data=list_portfolio_value).set_index(
        "Date", drop=True
    )
    portfolio_allocation = pd.DataFrame(columns=assets, data=list_portfolio_allocation)

    return portfolio_allocation, portfolio_value, portfolio_vty
