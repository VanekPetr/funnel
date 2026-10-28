"""Module for fetching financial data from AlgoStrata API.

This module provides functions to retrieve financial asset data from the
AlgoStrata API, including asset identifiers, ISIN codes, names, and price
history. It handles authentication, pagination, and data transformation.
"""

import os
from collections.abc import Iterator, Sequence
from typing import TypeVar

import dateutil.parser
import numpy as np
import pandas as pd
import requests
from loguru import logger

from ..settings import settings

T = TypeVar("T")


# BATCH FUNCTION
# ----------------------------------------------------------------------
def batch(iterable: Sequence[T], n: int = 1) -> Iterator[Sequence[T]]:
    """Split an iterable into batches of specified size.

    This function yields consecutive batches from the input iterable,
    each of size n (except possibly the last one which may be smaller).

    Args:
        iterable: The sequence to be batched
        n: The size of each batch (default: 1)

    Yields:
        Iterator[Sequence[T]]: Consecutive batches from the iterable

    Example:
        The last batch holds whatever is left over:

        >>> list(batch([1, 2, 3, 4, 5, 6, 7], 3))
        [[1, 2, 3], [4, 5, 6], [7]]
        >>> list(batch("abcde", 2))
        ['ab', 'cd', 'e']
    """
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


# Get IDs and ISIN codes
# ----------------------------------------------------------------------
def get_algostrata_data() -> pd.DataFrame:
    """Fetch financial asset data from AlgoStrata API.

    This function retrieves asset information and price history from the
    AlgoStrata API using the credentials specified in settings. It:
    1. Fetches asset metadata (IDs, ISINs, names)
    2. Retrieves price data for each asset in batches
    3. Processes and combines the data into a single DataFrame

    Returns:
        pd.DataFrame: A DataFrame containing daily price data for all assets,
                     with dates as index and multi-level columns for ISIN and name

    Raises:
        requests.RequestException: If the asset names request fails
        ValueError: If no asset returned any price data, e.g. because every
            prices request failed

    Example:
        Needs network access and an AlgoStrata API key (``ALGOSTRATA_KEY``), so it is
        not run as part of the tests:

        >>> prices = get_algostrata_data()  # doctest: +SKIP
        >>> prices.columns.nlevels  # doctest: +SKIP
        2
    """
    headers = {
        "X-Api-Key": settings.ALGOSTRATA_KEY,
        "content-type": "application/json",
    }
    # GET ASSET NAME DATA
    response = requests.get(settings.ALGOSTRATA_NAMES_URL, headers=headers, timeout=30)
    response.raise_for_status()
    assets = response.json()  # list of {"id", "isin", "name"}

    # Get the price data, a few assets per request
    # ----------------------------------------------------------------------
    batch_size = 3  # size of a batch
    round_rep = int(np.ceil(len(assets) / batch_size))  # number of iterations
    columns = []  # one single-column frame per asset with price data
    for rep, sub_assets in enumerate(batch(assets, batch_size)):
        # GET ASSET PRICE DATA
        logger.debug(f"Starting round {rep + 1} out of {round_rep}")
        response = requests.post(
            settings.ALGOSTRATA_PRICES_URL,
            json={"idList": [asset["id"] for asset in sub_assets]},
            headers=headers,
            timeout=30,
        )

        if response.status_code != 200:
            logger.warning(f"Round {rep + 1} out of {round_rep} failed: {response.reason}, content {response.text}")
            continue

        # Results come back in the order of the requested ids, so each one is
        # labelled by the asset it was requested for rather than by a running offset
        for asset, result in zip(sub_assets, response.json()["result"], strict=False):
            if result["priceData"] is None:
                continue
            reinvested_prices = result["priceData"]["reInvestedPrices"]
            df = pd.DataFrame(
                [x["unit_DKK"] for x in reinvested_prices],
                index=pd.DatetimeIndex([dateutil.parser.parse(x["date"]) for x in reinvested_prices]),
                columns=pd.MultiIndex.from_tuples([(asset["isin"], asset["name"])]),
            )
            # Keep the asset only if it has at least one price
            if not df.isnull().values.all():
                columns.append(df)

    if not columns:
        raise ValueError("No price data was downloaded from AlgoStrata")  # noqa: TRY003
    return pd.concat(columns, axis=1)


if __name__ == "__main__":  # pragma: no cover
    # Download raw data
    data_algostrata = get_algostrata_data()
    # Save daily_prices into parquet file
    data_algostrata.to_parquet(os.path.join(os.path.dirname(os.getcwd()), "financial_data/daily_price.parquet"))
