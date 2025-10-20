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
        requests.RequestException: If API requests fail
    """
    id_list = []  # empty list of IDs
    isin_list = []  # empty list of isin codes
    name_list = []  # empty list of names
    # GET ASSET NAME DATA
    response = requests.get(
        settings.ALGOSTRATA_NAMES_URL,
        headers={
            "X-Api-Key": settings.ALGOSTRATA_KEY,
            "content-type": "application/json",
        },
    )
    data = response.json()  # downloaded data
    # SAVE IDs and ISIN CODES INTO LISTS
    for asset in data:
        id_list.append(asset["id"])
        isin_list.append(asset["isin"])
        name_list.append(asset["name"])

    # Get the price data with index
    # ----------------------------------------------------------------------
    batch_size = 3  # size of a batch
    round_rep = int(np.ceil(len(id_list) / batch_size))  # number of iterations
    rep = 0  # current iteration
    first_run = True
    # LOAD DATASET BY STEP, EACH STEP XY ASSETS
    for sub_id_list in batch(id_list, batch_size):
        # GET ASSET PRICE DATA
        print("---- Starting round", rep + 1, "out of", round_rep, "----")
        response = requests.post(
            settings.ALGOSTRATA_PRICES_URL,
            json={"idList": sub_id_list},
            headers={
                "X-Api-Key": settings.ALGOSTRATA_KEY,
                "content-type": "application/json",
            },
        )

        if response.status_code != 200:
            print(f"Code {response.reason}, content {response.text}")
            print("---- Error round", rep + 1, "out of", round_rep, "----")
            continue

        data = response.json()  # downloaded data

        # CREATE PANDAS TABLE WITH ALL PRICE DATA
        for num, asset in enumerate(data["result"]):
            # IF WE HAVE A PRICE DATA THEN
            if asset["priceData"] is not None:
                price_data = asset["priceData"]
                reinvested_prices = price_data["reInvestedPrices"]
                dates = list(map(lambda x: dateutil.parser.parse(x["date"]), reinvested_prices))
                prices = list(map(lambda x: x["unit_DKK"], reinvested_prices))

                # IF THE FIRST RUN, THEN CREATE A TABLE
                if first_run:
                    daily_prices = pd.DataFrame(prices, index=dates, columns=[isin_list[0:1], name_list[0:1]])
                    first_run = False
                # IF NOT THE FIRST RUN, JUST CONCAT THE COLUMN INTO EXISTING TABLE
                else:
                    df = pd.DataFrame(
                        prices,
                        index=dates,
                        columns=[
                            isin_list[rep * batch_size + num : rep * batch_size + num + 1],
                            name_list[rep * batch_size + num : rep * batch_size + num + 1],
                        ],
                    )
                    # IF THE PRICE DATA ARE NOT ALL NaN, THEN
                    if not df.isnull().values.all():
                        daily_prices = pd.concat([daily_prices, df], axis=1)
        rep += 1

    return daily_prices


if __name__ == "__main__":
    # Download raw data
    data_algostrata = get_algostrata_data()
    # Save daily_prices into parquet file
    data_algostrata.to_parquet(os.path.join(os.path.dirname(os.getcwd()), "financial_data/daily_price.parquet"))
