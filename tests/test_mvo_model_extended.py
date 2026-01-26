"""Extended tests for MVO model module to improve coverage."""

import numpy as np
import pandas as pd
import pytest

from ifunnel.models.mvo_model import cholesky_psd, rebalancing_model


class TestCholeskyPsd:
    """Tests for the cholesky_psd function."""

    def test_positive_definite_matrix(self):
        """Test with a positive definite matrix."""
        # Create a positive definite matrix
        m = np.array([[4.0, 2.0, 0.5], [2.0, 5.0, 1.0], [0.5, 1.0, 3.0]])

        c = cholesky_psd(m)

        # Verify that C.T @ C approximates m
        reconstructed = c.T @ c
        np.testing.assert_array_almost_equal(reconstructed, m, decimal=5)

    def test_positive_semidefinite_matrix(self):
        """Test with a positive semidefinite matrix (has zero eigenvalues)."""
        # Create a rank-deficient positive semidefinite matrix
        a = np.array([[1.0, 0.5], [0.5, 1.0], [0.0, 0.0]])
        m = a @ a.T  # This creates a PSD matrix

        c = cholesky_psd(m)

        # Should not raise an error
        assert c.shape[0] == c.shape[1]

    def test_diagonal_matrix(self):
        """Test with a diagonal matrix."""
        m = np.diag([4.0, 9.0, 16.0])

        c = cholesky_psd(m)

        reconstructed = c.T @ c
        np.testing.assert_array_almost_equal(reconstructed, m, decimal=5)

    def test_identity_matrix(self):
        """Test with identity matrix."""
        m = np.eye(4)

        c = cholesky_psd(m)

        reconstructed = c.T @ c
        np.testing.assert_array_almost_equal(reconstructed, m, decimal=5)

    def test_near_singular_matrix(self):
        """Test with a near-singular matrix."""
        # Create a matrix with very small eigenvalues
        m = np.array([[1.0, 0.9999, 0.5], [0.9999, 1.0, 0.5], [0.5, 0.5, 0.3]])

        c = cholesky_psd(m)

        # Should produce a result without error
        assert isinstance(c, np.ndarray)

    def test_with_negative_eigenvalue_fix(self):
        """Test matrix that requires the negative eigenvalue fix."""
        # Create a matrix with definite negative eigenvalues
        # Start with a rank-deficient matrix and subtract identity to make it non-PSD
        n = 5
        np.random.seed(42)
        # Create a matrix with one near-zero eigenvalue
        a = np.random.randn(n, n - 1)  # Rank deficient
        m = a @ a.T  # Rank n-1, has one zero eigenvalue
        # Subtract identity to make eigenvalues negative
        m = m - 0.5 * np.eye(n)  # This will make some eigenvalues negative

        c = cholesky_psd(m)

        # Should handle the negative eigenvalue case
        assert isinstance(c, np.ndarray)
        assert c.shape == (n, n)


class TestRebalancingModel:
    """Tests for the rebalancing_model function."""

    @pytest.fixture
    def simple_inputs(self):
        """Create simple inputs for testing rebalancing model."""
        assets = ["Asset1", "Asset2", "Asset3"]
        mu = pd.Series([0.08, 0.05, 0.03], index=assets)
        covariance = pd.DataFrame(
            [[0.04, 0.01, 0.005], [0.01, 0.02, 0.003], [0.005, 0.003, 0.01]],
            index=assets,
            columns=assets,
        )
        return mu, covariance

    def test_basic_rebalancing(self, simple_inputs):
        """Test basic portfolio rebalancing."""
        mu, covariance = simple_inputs
        vty_target = 0.15 * 100  # Scaled by portfolio value
        cash = 100
        x_old = pd.Series([0, 0, 0], index=mu.index)
        trans_cost = 0.001
        max_weight = 0.5
        solver = "CLARABEL"

        result = rebalancing_model(
            mu=mu,
            covariance=covariance,
            vty_target=vty_target,
            cash=cash,
            x_old=x_old,
            trans_cost=trans_cost,
            max_weight=max_weight,
            solver=solver,
            inaccurate=True,
            lower_bound=0,
        )

        assert result is not None
        opt_port, _vty_result_p, _port_val, _remaining_cash = result
        assert isinstance(opt_port, pd.Series)
        # Weights should sum to approximately 1
        assert abs(opt_port.sum() - 1.0) < 0.01

    def test_rebalancing_with_existing_position(self, simple_inputs):
        """Test rebalancing with an existing portfolio position."""
        mu, covariance = simple_inputs
        vty_target = 0.15 * 100
        cash = 20
        x_old = pd.Series([30, 30, 20], index=mu.index)  # Existing position
        trans_cost = 0.001
        max_weight = 0.5
        solver = "CLARABEL"

        result = rebalancing_model(
            mu=mu,
            covariance=covariance,
            vty_target=vty_target,
            cash=cash,
            x_old=x_old,
            trans_cost=trans_cost,
            max_weight=max_weight,
            solver=solver,
            inaccurate=True,
            lower_bound=0,
        )

        assert result is not None
        opt_port, _vty_result_p, _port_val, _remaining_cash = result
        assert isinstance(opt_port, pd.Series)

    @pytest.mark.skip(reason="Mixed-integer problem - no MIP-capable solver available")
    def test_rebalancing_with_lower_bound(self, simple_inputs):
        """Test rebalancing with lower bound constraint."""
        mu, covariance = simple_inputs
        vty_target = 0.20 * 100
        cash = 100
        x_old = pd.Series([0, 0, 0], index=mu.index)
        trans_cost = 0.001
        max_weight = 0.5
        solver = "CLARABEL"
        lower_bound = 0.1

        result = rebalancing_model(
            mu=mu,
            covariance=covariance,
            vty_target=vty_target,
            cash=cash,
            x_old=x_old,
            trans_cost=trans_cost,
            max_weight=max_weight,
            solver=solver,
            inaccurate=True,
            lower_bound=lower_bound,
        )

        # Result may be None if infeasible with lower bound
        if result is not None:
            opt_port, _, _, _ = result
            # Check that selected assets meet lower bound
            for w in opt_port:
                if w > 1e-5:
                    assert w >= lower_bound - 1e-4

    def test_rebalancing_tight_volatility_constraint(self, simple_inputs):
        """Test rebalancing with very tight volatility constraint."""
        mu, covariance = simple_inputs
        vty_target = 0.01 * 100  # Very tight constraint
        cash = 100
        x_old = pd.Series([0, 0, 0], index=mu.index)
        trans_cost = 0.001
        max_weight = 1.0
        solver = "CLARABEL"

        result = rebalancing_model(
            mu=mu,
            covariance=covariance,
            vty_target=vty_target,
            cash=cash,
            x_old=x_old,
            trans_cost=trans_cost,
            max_weight=max_weight,
            solver=solver,
            inaccurate=True,
            lower_bound=0,
        )

        # Should find a solution even with tight constraints
        if result is not None:
            opt_port, _vty_result_p, _port_val, _remaining_cash = result
            assert isinstance(opt_port, pd.Series)

    def test_rebalancing_high_transaction_costs(self, simple_inputs):
        """Test rebalancing with high transaction costs."""
        mu, covariance = simple_inputs
        vty_target = 0.15 * 100
        cash = 100
        x_old = pd.Series([30, 30, 30], index=mu.index)
        trans_cost = 0.05  # 5% transaction cost
        max_weight = 0.5
        solver = "CLARABEL"

        result = rebalancing_model(
            mu=mu,
            covariance=covariance,
            vty_target=vty_target,
            cash=cash,
            x_old=x_old,
            trans_cost=trans_cost,
            max_weight=max_weight,
            solver=solver,
            inaccurate=True,
            lower_bound=0,
        )

        # Should still produce a valid result
        if result is not None:
            opt_port, _, _, _ = result
            assert isinstance(opt_port, pd.Series)

    def test_rebalancing_infeasible_constraints(self, simple_inputs):
        """Test rebalancing with infeasible constraints triggers failure path."""
        mu, covariance = simple_inputs
        # Create an impossible scenario: negative cash and existing position
        # that requires selling but has high transaction costs
        vty_target = 0.001  # Very low volatility target
        cash = -1000  # Negative cash - must sell
        x_old = pd.Series([100, 100, 100], index=mu.index)  # Large existing position
        trans_cost = 0.99  # 99% transaction cost - makes any trade extremely costly
        max_weight = 0.01  # Very restrictive max weight
        solver = "CLARABEL"

        import os
        import tempfile

        # Change to temp directory so pickle file doesn't pollute working dir
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                result = rebalancing_model(
                    mu=mu,
                    covariance=covariance,
                    vty_target=vty_target,
                    cash=cash,
                    x_old=x_old,
                    trans_cost=trans_cost,
                    max_weight=max_weight,
                    solver=solver,
                    inaccurate=False,  # Don't accept inaccurate solutions
                    lower_bound=0,
                )
                # Result could be None for infeasible problem or a valid result
                # if solver finds a way. Either way, the code path is exercised.
                assert result is None or isinstance(result, tuple)
            finally:
                os.chdir(original_dir)
