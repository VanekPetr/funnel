"""Tests for scenario generation module."""

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.scenario_generation import MomentGenerator, ScenarioGenerator


def test_moment_generator_alpha_numerator():
    """Test the _alpha_numerator method."""
    np.random.seed(42)
    zz = np.random.randn(5, 10)
    ss = np.cov(zz, rowvar=True)
    result = MomentGenerator._alpha_numerator(zz, ss)
    assert isinstance(result, (float, np.floating))
    assert result >= 0


def test_moment_generator_ledoit_wolf_shrinkage():
    """Test Ledoit-Wolf shrinkage method."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])
    s = np.cov(data.T)
    result = MomentGenerator._ledoit_wolf_shrinkage(data, s)
    assert isinstance(result, np.ndarray)
    assert result.shape == s.shape


def test_moment_generator_ledoit_wolf_single_asset():
    """Test Ledoit-Wolf shrinkage with single asset."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 1), columns=["A"])
    s = np.cov(data.T)
    result = MomentGenerator._ledoit_wolf_shrinkage(data, s)
    assert np.array_equal(result, s)


def test_moment_generator_jorion_shrinkage():
    """Test Jorion shrinkage method."""
    mu = 0.05
    mu_star = 0.03
    lambda_ = 0.5
    result = MomentGenerator._jorion_shrinkage(mu, mu_star, lambda_)
    expected = lambda_ * mu_star + (1 - lambda_) * mu
    assert result == expected


def test_compute_annualized_covariance():
    """Test compute_annualized_covariance method."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])
    result = MomentGenerator.compute_annualized_covariance(data)
    # Method returns a DataFrame now
    assert isinstance(result, (np.ndarray, pd.DataFrame))
    if isinstance(result, pd.DataFrame):
        assert result.shape == (5, 5)
    else:
        assert result.shape == (5, 5)


def test_generate_sigma_mu_for_test_periods():
    """Test generate_sigma_mu_for_test_periods method."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(200, 3), columns=["A", "B", "C"])
    n_test = 40
    sigma_lst, mu_lst = MomentGenerator.generate_sigma_mu_for_test_periods(data, n_test)
    assert isinstance(sigma_lst, list)
    assert isinstance(mu_lst, list)
    assert len(sigma_lst) > 0
    assert len(mu_lst) > 0
    assert len(sigma_lst) == len(mu_lst)


def test_split_dataset():
    """Test split_dataset method."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])
    sampling_set, estimating_set = MomentGenerator.split_dataset(data, sampling_ratio=0.6)
    assert len(sampling_set) == 60
    assert len(estimating_set) == 40
    assert len(sampling_set) + len(estimating_set) == len(data)


def test_split_dataset_invalid_ratio():
    """Test split_dataset with invalid ratio."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])
    with pytest.raises(ValueError):
        MomentGenerator.split_dataset(data, sampling_ratio=1.5)
    with pytest.raises(ValueError):
        MomentGenerator.split_dataset(data, sampling_ratio=0)


def test_generate_annual_sigma_mu_with_risk_free():
    """Test generate_annual_sigma_mu_with_risk_free method."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    sigma_annual, mu_annual, sigma_weekly, mu_weekly = MomentGenerator.generate_annual_sigma_mu_with_risk_free(
        data, risk_free_rate_annual=0.02
    )
    assert isinstance(sigma_annual, pd.DataFrame)
    assert isinstance(mu_annual, pd.Series)
    assert isinstance(sigma_weekly, pd.DataFrame)
    assert isinstance(mu_weekly, pd.Series)
    assert "Cash" in mu_annual.index
    assert "Cash" in sigma_annual.index


def test_scenario_generator_monte_carlo():
    """Test monte_carlo scenario generation."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    gen = ScenarioGenerator(rng)
    
    data = pd.DataFrame(np.random.randn(200, 3), columns=["A", "B", "C"])
    n_simulations = 10
    n_test = 16
    sigma_lst, mu_lst = MomentGenerator.generate_sigma_mu_for_test_periods(data, n_test)
    
    result = gen.monte_carlo(data, n_simulations, n_test, sigma_lst, mu_lst)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == n_simulations
    assert result.shape[2] == data.shape[1]


def test_scenario_generator_bootstrapping():
    """Test bootstrapping scenario generation."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    gen = ScenarioGenerator(rng)
    
    data = pd.DataFrame(np.random.randn(200, 3), columns=["A", "B", "C"])
    n_simulations = 10
    n_test = 16
    
    result = gen.bootstrapping(data, n_simulations, n_test)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == n_simulations
    assert result.shape[2] == data.shape[1]


def test_mc_simulation_annual_from_weekly():
    """Test MC simulation for annual returns from weekly data."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    gen = ScenarioGenerator(rng)
    
    weekly_mu = pd.Series([0.001, 0.002, 0.015], index=["A", "B", "Cash"])
    weekly_sigma = pd.DataFrame(
        [[0.01, 0.005, 0], [0.005, 0.02, 0], [0, 0, 0]], index=["A", "B", "Cash"], columns=["A", "B", "Cash"]
    )
    
    result = gen.mc_simulation_annual_from_weekly(
        weekly_mu, weekly_sigma, n_simulations=10, n_years=2, cash_return_annual=0.015
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 2, 3)
    # Check that Cash returns are constant
    assert np.allclose(result[:, :, 2], 0.015)


def test_bootstrap_simulation_annual_from_weekly():
    """Test bootstrap simulation for annual returns from weekly data."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    gen = ScenarioGenerator(rng)
    
    data = pd.DataFrame(np.random.randn(100, 2) * 0.01, columns=["A", "B"])
    
    result = gen.bootstrap_simulation_annual_from_weekly(data, n_simulations=10, n_years=2, cash_return_annual=0.015)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 2, 3)
    # Check that Cash returns are constant
    assert np.allclose(result[:, :, 2], 0.015)
