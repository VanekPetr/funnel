"""Tests for lifecycle MVO model module."""

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.lifecycle.mvo_lifecycle_model import (
    calculate_analysis_metrics,
    calculate_risk_metrics,
)


def test_calculate_risk_metrics_basic():
    """Test basic risk metrics calculation."""
    yearly_returns = pd.Series([0.05, 0.10, -0.02, 0.08, 0.06])
    
    annual_return, annual_std_dev, sharpe_ratio, downside_std_dev, sortino_ratio = calculate_risk_metrics(
        yearly_returns, risk_free_rate=0.02
    )
    
    assert isinstance(annual_return, (float, np.floating))
    assert isinstance(annual_std_dev, (float, np.floating))
    assert isinstance(sharpe_ratio, (float, np.floating, type(None)))
    assert isinstance(downside_std_dev, (float, np.floating))
    assert isinstance(sortino_ratio, (float, np.floating, type(None)))
    
    # Check that annual return is the mean
    assert abs(annual_return - yearly_returns.mean()) < 1e-10


def test_calculate_risk_metrics_positive_returns():
    """Test risk metrics with all positive returns."""
    yearly_returns = pd.Series([0.05, 0.10, 0.08, 0.12, 0.06])
    
    annual_return, annual_std_dev, sharpe_ratio, downside_std_dev, sortino_ratio = calculate_risk_metrics(
        yearly_returns, risk_free_rate=0.02
    )
    
    assert annual_return > 0
    assert annual_std_dev > 0
    assert sharpe_ratio > 0
    # Downside deviation should be very small or zero since all returns are above risk-free rate
    assert downside_std_dev >= 0


def test_calculate_risk_metrics_negative_returns():
    """Test risk metrics with negative returns."""
    yearly_returns = pd.Series([-0.05, -0.10, -0.02, -0.08, -0.06])
    
    annual_return, annual_std_dev, sharpe_ratio, downside_std_dev, sortino_ratio = calculate_risk_metrics(
        yearly_returns, risk_free_rate=0.02
    )
    
    assert annual_return < 0
    assert annual_std_dev > 0
    assert sharpe_ratio < 0  # Negative Sharpe when returns < risk-free rate
    assert downside_std_dev > 0


def test_calculate_risk_metrics_zero_std_dev():
    """Test risk metrics with zero standard deviation."""
    yearly_returns = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05])
    
    annual_return, annual_std_dev, sharpe_ratio, downside_std_dev, sortino_ratio = calculate_risk_metrics(
        yearly_returns, risk_free_rate=0.02
    )
    
    assert annual_return == 0.05
    assert annual_std_dev == 0
    assert sharpe_ratio is None  # Should be None when std_dev is 0


def test_calculate_risk_metrics_different_risk_free_rate():
    """Test risk metrics with different risk-free rate."""
    yearly_returns = pd.Series([0.05, 0.10, -0.02, 0.08, 0.06])
    
    # Test with higher risk-free rate
    _, _, sharpe_high, _, sortino_high = calculate_risk_metrics(yearly_returns, risk_free_rate=0.08)
    
    # Test with lower risk-free rate
    _, _, sharpe_low, _, sortino_low = calculate_risk_metrics(yearly_returns, risk_free_rate=0.01)
    
    # Higher risk-free rate should result in lower Sharpe and Sortino ratios
    assert sharpe_low > sharpe_high


def test_calculate_analysis_metrics_basic():
    """Test basic analysis metrics calculation."""
    terminal_values = pd.Series([100, 110, 95, 120, 105, 115, 90, 125, 98, 112])
    
    metrics_df = calculate_analysis_metrics(terminal_values)
    
    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 1
    assert "Mean Terminal Value" in metrics_df.columns
    assert "Standard Deviation Terminal Value" in metrics_df.columns
    assert "Max Terminal Value" in metrics_df.columns
    assert "Min Terminal Value" in metrics_df.columns
    assert "Lower Decile Average" in metrics_df.columns
    assert "Upper Decile Average" in metrics_df.columns
    assert "Lower Quartile Average" in metrics_df.columns
    assert "Upper Quartile Average" in metrics_df.columns


def test_calculate_analysis_metrics_values():
    """Test that analysis metrics have correct values."""
    terminal_values = pd.Series([100, 110, 95, 120, 105, 115, 90, 125, 98, 112])
    
    metrics_df = calculate_analysis_metrics(terminal_values)
    
    # Check mean
    assert abs(metrics_df["Mean Terminal Value"].iloc[0] - terminal_values.mean()) < 1e-10
    
    # Check standard deviation (np.std uses different defaults than pandas)
    expected_std = np.std(terminal_values)
    assert abs(metrics_df["Standard Deviation Terminal Value"].iloc[0] - expected_std) < 1e-10
    
    # Check max and min
    assert metrics_df["Max Terminal Value"].iloc[0] == 125
    assert metrics_df["Min Terminal Value"].iloc[0] == 90
    
    # Check that upper values > lower values
    assert metrics_df["Upper Decile Average"].iloc[0] > metrics_df["Lower Decile Average"].iloc[0]
    assert metrics_df["Upper Quartile Average"].iloc[0] > metrics_df["Lower Quartile Average"].iloc[0]


def test_calculate_analysis_metrics_uniform_values():
    """Test analysis metrics with uniform values."""
    terminal_values = pd.Series([100] * 20)
    
    metrics_df = calculate_analysis_metrics(terminal_values)
    
    # All metrics should be 100 except std dev which should be 0
    assert metrics_df["Mean Terminal Value"].iloc[0] == 100
    assert metrics_df["Standard Deviation Terminal Value"].iloc[0] == 0
    assert metrics_df["Max Terminal Value"].iloc[0] == 100
    assert metrics_df["Min Terminal Value"].iloc[0] == 100
    assert metrics_df["Lower Decile Average"].iloc[0] == 100
    assert metrics_df["Upper Decile Average"].iloc[0] == 100


def test_calculate_analysis_metrics_large_dataset():
    """Test analysis metrics with a large dataset."""
    np.random.seed(42)
    terminal_values = pd.Series(np.random.uniform(80, 150, 1000))
    
    metrics_df = calculate_analysis_metrics(terminal_values)
    
    # Check that metrics are within reasonable ranges
    assert 80 <= metrics_df["Mean Terminal Value"].iloc[0] <= 150
    assert metrics_df["Standard Deviation Terminal Value"].iloc[0] > 0
    assert metrics_df["Lower Decile Average"].iloc[0] < metrics_df["Mean Terminal Value"].iloc[0]
    assert metrics_df["Upper Decile Average"].iloc[0] > metrics_df["Mean Terminal Value"].iloc[0]
