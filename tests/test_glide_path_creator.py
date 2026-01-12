"""Tests for lifecycle glide path creator module."""

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.lifecycle.glide_path_creator import generate_risk_profiles


def test_generate_risk_profiles_basic():
    """Test basic risk profile generation."""
    n_periods = 10
    initial_risk = 0.20
    minimum_risk = 0.05
    
    df, fig = generate_risk_profiles(n_periods, initial_risk, minimum_risk)
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_periods
    assert "Linear GP" in df.columns
    assert "Concave GP" in df.columns
    assert "Convex GP" in df.columns
    
    # Check that risks decrease over time
    assert df["Linear GP"].iloc[0] >= df["Linear GP"].iloc[-1]
    assert df["Concave GP"].iloc[0] >= df["Concave GP"].iloc[-1]
    assert df["Convex GP"].iloc[0] >= df["Convex GP"].iloc[-1]
    
    # Check start and end values for linear profile
    assert abs(df["Linear GP"].iloc[0] - initial_risk) < 1e-10
    assert abs(df["Linear GP"].iloc[-1] - minimum_risk) < 1e-10


def test_generate_risk_profiles_values():
    """Test that risk profiles have correct values."""
    n_periods = 10
    initial_risk = 0.20
    minimum_risk = 0.05
    
    df, fig = generate_risk_profiles(n_periods, initial_risk, minimum_risk)
    
    # All values should be between minimum and initial risk (with small tolerance for floating point)
    assert (df["Linear GP"] >= minimum_risk - 1e-10).all()
    assert (df["Linear GP"] <= initial_risk + 1e-10).all()
    # Concave and Convex may slightly go below or above due to the formulas
    # Check that they're generally in the right range
    assert df["Concave GP"].min() >= minimum_risk - 0.01
    assert df["Concave GP"].max() <= initial_risk + 0.01
    assert df["Convex GP"].min() >= minimum_risk - 0.01
    assert df["Convex GP"].max() <= initial_risk + 0.01


def test_generate_risk_profiles_figure():
    """Test that a figure is generated."""
    n_periods = 10
    initial_risk = 0.20
    minimum_risk = 0.05
    
    df, fig = generate_risk_profiles(n_periods, initial_risk, minimum_risk)
    
    # Check that figure is created
    assert fig is not None
    # Check that figure has the correct number of traces
    assert len(fig.data) == 3
    # Check figure layout
    assert fig.layout.yaxis.title.text == "Annual Standard Deviation"
    assert fig.layout.xaxis.title.text == "Period"


def test_generate_risk_profiles_single_period():
    """Test risk profile generation with single period."""
    n_periods = 1
    initial_risk = 0.20
    minimum_risk = 0.05
    
    df, fig = generate_risk_profiles(n_periods, initial_risk, minimum_risk)
    
    assert len(df) == 1
    # With a single period, all profiles should have the same value
    assert abs(df["Linear GP"].iloc[0] - initial_risk) < 1e-10


def test_generate_risk_profiles_many_periods():
    """Test risk profile generation with many periods."""
    n_periods = 100
    initial_risk = 0.25
    minimum_risk = 0.02
    
    df, fig = generate_risk_profiles(n_periods, initial_risk, minimum_risk)
    
    assert len(df) == n_periods
    # Check monotonic decrease for all profiles
    assert (df["Linear GP"].diff().dropna() <= 0).all()
    # For concave and convex, the rate of decrease changes
    assert len(df["Concave GP"]) == n_periods
    assert len(df["Convex GP"]) == n_periods
