"""Pytest fixtures for the ifunnel project tests.

This module contains fixtures that are shared across multiple test files.
These fixtures provide test data, model instances, and other resources
needed for testing the various components of the ifunnel project.

Security Notes:
- S101 (assert usage): Asserts are appropriate in test code for validation.
- S603 (subprocess calls): Any subprocess usage in tests uses explicit argument lists, not shell=True.
- S607 (partial executable path): Known commands (e.g. git) resolved from PATH are safe in test context.
"""

from pathlib import Path

import matplotlib as mpl

# Force a non-interactive backend before pyplot is imported anywhere. Some model
# code (e.g. clustering.cluster) calls plt.show(); on a GUI backend such as macOS's
# 'macosx' that enters a blocking mainloop and hangs the suite (fatally so under
# xdist, where the per-test signal timeout cannot interrupt the native loop).
mpl.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.mst import minimum_spanning_tree
from ifunnel.models.scenario_generation import MomentGenerator, ScenarioGenerator


@pytest.fixture(scope="module", name="root_dir")
def root_fixture():
    """Provide the path to the project root directory.

    Returns:
        Path: Path to the project root directory.
    """
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """Provide the path to the test resources directory.

    Returns:
        Path: Path to the directory containing test resources.
    """
    return Path(__file__).parent / "resources"


@pytest.fixture
def rng():
    """Provide a random number generator with a fixed seed for reproducibility.

    Returns:
        numpy.random.Generator: Random number generator with seed 42.
    """
    test_rng = np.random.default_rng(seed=42)
    return test_rng


@pytest.fixture
def scgen(rng):
    """Provide a ScenarioGenerator instance for generating test scenarios.

    Args:
        rng: Random number generator with a fixed seed.

    Returns:
        ScenarioGenerator: Instance for generating financial scenarios.
    """
    sg = ScenarioGenerator(rng)
    return sg


@pytest.fixture(scope="session")
def weekly_returns(resource_dir):
    """Provide historical weekly returns data for testing.

    Args:
        resource_dir: Path to the test resources directory.

    Returns:
        pd.DataFrame: DataFrame containing historical weekly returns.
    """
    weekly_returns = pd.read_parquet(resource_dir / "data/all_etfs_rets.parquet.gzip")
    return weekly_returns


@pytest.fixture(scope="session")
def tickers(weekly_returns):
    """Provide the list of ticker symbols from the weekly returns data.

    Args:
        weekly_returns: DataFrame containing historical weekly returns.

    Returns:
        numpy.ndarray: Array of ticker symbols.
    """
    tickers = weekly_returns.columns.values
    return tickers


@pytest.fixture(scope="session")
def names(resource_dir):
    """Provide the list of ETF names from the test data.

    Args:
        resource_dir: Path to the test resources directory.

    Returns:
        numpy.ndarray: Array of ETF names.
    """
    df_names = pd.read_parquet(resource_dir / "data/all_etfs_rets_name.parquet.gzip")
    names = df_names.columns.values
    return names


@pytest.fixture(scope="session")
def start_train_date():
    """Provide the start date for the training period.

    Returns:
        pandas.Timestamp: Start date for the training dataset.
    """
    return pd.to_datetime("2014-06-11")


@pytest.fixture(scope="session")
def end_train_date():
    """Provide the end date for the training period.

    Returns:
        pandas.Timestamp: End date for the training dataset.
    """
    return pd.to_datetime("2017-07-01")


@pytest.fixture(scope="session")
def start_test_date():
    """Provide the start date for the testing period.

    Returns:
        pandas.Timestamp: Start date for the test dataset.
    """
    return pd.to_datetime("2017-07-01")


@pytest.fixture(scope="session")
def end_test_date():
    """Provide the end date for the testing period.

    Returns:
        pandas.Timestamp: End date for the test dataset.
    """
    return pd.to_datetime("2022-07-20")


@pytest.fixture(scope="session")
def benchmark_isin_1(tickers, names):
    """Provide the first benchmark portfolio for testing.

    This fixture returns a list containing the ticker for the
    iShares MSCI ACWI ETF, which is used as a benchmark in tests.

    Args:
        tickers: Array of ticker symbols.
        names: Array of ETF names.

    Returns:
        list: List containing the ticker for the benchmark ETF.
    """
    benchmarks = ["iShares MSCI ACWI ETF"]
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]
    return benchmark_isin


@pytest.fixture(scope="session")
def benchmark_isin_2(tickers, names):
    """Provide the second benchmark portfolio for testing.

    This fixture returns a list containing the tickers for two ETFs:
    iShares MSCI All Country Asia ex Japan Index Fund ETF and
    iShares MSCI ACWI ETF, which are used as benchmarks in tests.

    Args:
        tickers: Array of ticker symbols.
        names: Array of ETF names.

    Returns:
        list: List containing the tickers for the benchmark ETFs.
    """
    benchmarks = [
        "iShares MSCI All Country Asia ex Japan Index Fund ETF",
        "iShares MSCI ACWI ETF",
    ]
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]
    return benchmark_isin


@pytest.fixture(scope="session")
def whole_wide_dataset(weekly_returns, start_train_date, end_test_date):
    """Provide the complete dataset for both training and testing periods.

    This fixture returns a DataFrame containing weekly returns for all assets
    from the start of the training period to the end of the testing period.

    Args:
        weekly_returns: DataFrame containing historical weekly returns.
        start_train_date: Start date for the training dataset.
        end_test_date: End date for the test dataset.

    Returns:
        pd.DataFrame: DataFrame containing weekly returns for the entire period.
    """
    whole_dataset = weekly_returns[
        (weekly_returns.index >= start_train_date) & (weekly_returns.index <= end_test_date)
    ].copy()
    return whole_dataset


@pytest.fixture(scope="session")
def train_wide_dataset(weekly_returns, start_train_date, end_train_date):
    """Provide the training dataset with all assets.

    This fixture returns a DataFrame containing weekly returns for all assets
    during the training period only.

    Args:
        weekly_returns: DataFrame containing historical weekly returns.
        start_train_date: Start date for the training dataset.
        end_train_date: End date for the training dataset.

    Returns:
        pd.DataFrame: DataFrame containing weekly returns for the training period.
    """
    train_dataset = weekly_returns[
        (weekly_returns.index >= start_train_date) & (weekly_returns.index <= end_train_date)
    ].copy()
    return train_dataset


@pytest.fixture(scope="session")
def subset_of_assets(train_wide_dataset):
    """Provide a subset of assets selected using minimum spanning tree.

    This fixture applies the minimum spanning tree algorithm twice to select
    a subset of assets that are representative of the full dataset.

    Args:
        train_wide_dataset: DataFrame containing training data for all assets.

    Returns:
        list: List of selected asset identifiers.
    """
    n_mst_runs = 2
    subset_mst_df = train_wide_dataset
    for _i in range(n_mst_runs):
        subset_mst, subset_mst_df, _, _ = minimum_spanning_tree(subset_mst_df)
    return subset_mst


@pytest.fixture(scope="session")
def whole_narrow_dataset(whole_wide_dataset, subset_of_assets):
    """Provide the complete dataset for the subset of assets.

    This fixture returns a DataFrame containing weekly returns for the selected
    subset of assets from the start of the training period to the end of the testing period.

    Args:
        whole_wide_dataset: DataFrame containing data for all assets and periods.
        subset_of_assets: List of selected asset identifiers.

    Returns:
        pd.DataFrame: DataFrame containing weekly returns for the subset of assets.
    """
    whole_dataset = whole_wide_dataset[subset_of_assets]
    return whole_dataset


@pytest.fixture(scope="session")
def test_narrow_dataset(weekly_returns, start_test_date, end_test_date, subset_of_assets):
    """Provide the test dataset for the subset of assets.

    This fixture returns a DataFrame containing weekly returns for the selected
    subset of assets during the testing period only.

    Args:
        weekly_returns: DataFrame containing historical weekly returns.
        start_test_date: Start date for the test dataset.
        end_test_date: End date for the test dataset.
        subset_of_assets: List of selected asset identifiers.

    Returns:
        pd.DataFrame: DataFrame containing test data for the subset of assets.
    """
    test_dataset = weekly_returns[
        (weekly_returns.index > start_test_date) & (weekly_returns.index <= end_test_date)
    ].copy()
    test_dataset = test_dataset[subset_of_assets]
    return test_dataset


@pytest.fixture(scope="session")
def length_test_dataset(test_narrow_dataset):
    """Provide the number of periods in the test dataset.

    Args:
        test_narrow_dataset: DataFrame containing test data.

    Returns:
        int: Number of periods (rows) in the test dataset.
    """
    return test_narrow_dataset.shape[0]


@pytest.fixture(scope="session")
def n_simulations():
    """Provide the number of simulations to use for scenario generation.

    Returns:
        int: Number of simulations (250).
    """
    return 250


@pytest.fixture
def mc_scenarios(moments, whole_narrow_dataset, length_test_dataset, n_simulations, scgen):
    """Generate Monte Carlo scenarios for testing.

    This fixture uses the ScenarioGenerator to create Monte Carlo simulations
    based on the provided moments (mean and covariance) for each period.

    Args:
        moments: Tuple containing (sigma_lst, mu_lst) for each period.
        whole_narrow_dataset: DataFrame containing data for the subset of assets.
        length_test_dataset: Number of periods in the test dataset.
        n_simulations: Number of simulations to generate.
        scgen: ScenarioGenerator instance.

    Returns:
        numpy.ndarray: 3D array of simulated returns.
    """
    sigma_lst, mu_lst = moments

    scenarios = scgen.monte_carlo(
        data=whole_narrow_dataset,
        n_simulations=n_simulations,
        n_test=length_test_dataset,
        sigma_lst=sigma_lst,
        mu_lst=mu_lst,
    )

    return scenarios


@pytest.fixture(scope="session")
def moments(whole_narrow_dataset, length_test_dataset):
    """Generate statistical moments (mean and covariance) for each test period.

    This fixture computes the mean vectors and covariance matrices for each
    investment period using the MomentGenerator.

    Args:
        whole_narrow_dataset: DataFrame containing data for the subset of assets.
        length_test_dataset: Number of periods in the test dataset.

    Returns:
        tuple: A tuple containing:
            - sigma_lst: List of covariance matrices for each period.
            - mu_lst: List of mean arrays for each period.
    """
    sigma_lst, mu_lst = MomentGenerator.generate_sigma_mu_for_test_periods(
        data=whole_narrow_dataset, n_test=length_test_dataset
    )

    return sigma_lst, mu_lst
