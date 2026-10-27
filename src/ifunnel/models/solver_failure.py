"""Shared handling of a failed solve for the optimization models.

The CVaR, MVO and lifecycle rebalancing models all stop the same way when the
solver returns a non-optimal status: optionally save the inputs for offline
investigation, log the status, and raise.
"""

import pickle  # nosec B403
from pathlib import Path
from typing import Any, NoReturn

from loguru import logger


def raise_solver_failure(
    status: str,
    inputs: dict[str, Any] | None = None,
    debug_dump: str | Path | None = None,
    hint: str = "",
) -> NoReturn:
    """Raise because the solver did not return an optimal solution.

    Args:
        status: The status reported by the CVXPY problem.
        inputs: The model inputs, saved to ``debug_dump`` when one is given.
        debug_dump: Optional file to pickle ``inputs`` to, so the failing problem
            can be investigated separately, e.g. in a notebook. Nothing is
            written unless the caller asks for it.
        hint: Optional extra sentence appended to the logged error.

    Raises:
        RuntimeError: Always, naming the solver status.

    Example:
        >>> raise_solver_failure("infeasible")
        Traceback (most recent call last):
        ...
        RuntimeError: Solver did not find an optimal solution (status: infeasible)
    """
    if debug_dump is not None and inputs is not None:
        with open(debug_dump, "wb") as file:
            pickle.dump(inputs, file)
        logger.info(f"Saved the failing solver inputs to {debug_dump}")

    logger.error(f"❌ Solver does not find optimal solution. Status code is {status}. {hint}".rstrip())
    raise RuntimeError(f"Solver did not find an optimal solution (status: {status})")  # noqa: TRY003
