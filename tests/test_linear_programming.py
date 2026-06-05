import numpy as np
import pytest
from unittest.mock import MagicMock

from vnn import linear_programming as lp


def _is_gurobi_available():
    try:
        lp._get_gurobi_env()
        return True
    except Exception:
        return False


requires_gurobi = pytest.mark.skipif(
    not _is_gurobi_available(),
    reason="Gurobi license not available",
)


@pytest.fixture
def simple_lp_problem():
    # Objective: Minimize -x - 2y
    c = np.array([-1, -2])
    # Constraints:
    # x <= 1
    # y <= 2
    # -x <= 0 (x >= 0)
    # -y <= 0 (y >= 0)
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 2, 0, 0])

    # Expected solution: x = 1, y = 2
    expected_x = np.array([1, 2])
    return c, A, b, expected_x


def test_solve_scipy(simple_lp_problem):
    c, A, b, expected_x = simple_lp_problem
    x = lp.solve_scipy(c, A, b)
    assert np.allclose(x, expected_x)


@requires_gurobi
def test_solve_gurobi(simple_lp_problem):
    c, A, b, expected_x = simple_lp_problem
    x = lp.solve_gurobi(c, A, b)
    assert np.allclose(x, expected_x)


@requires_gurobi
def test_get_gurobi_env():
    # Test that it acts as a singleton
    env1 = lp._get_gurobi_env()
    env2 = lp._get_gurobi_env()
    assert env1 is env2


def test_cleanup_gurobi_env():
    """Test that cleanup properly disposes the Gurobi environment."""
    mock_env = MagicMock()
    original_env = lp._GUROBI_ENV

    try:
        # Set a mock env
        lp._GUROBI_ENV = mock_env
        lp._cleanup_gurobi_env()
        mock_env.dispose.assert_called_once()
        assert lp._GUROBI_ENV is None

        # Calling cleanup again when already None should be safe
        lp._cleanup_gurobi_env()
        assert lp._GUROBI_ENV is None
    finally:
        lp._GUROBI_ENV = original_env


def test_cleanup_gurobi_env_exception():
    """Test that cleanup handles exceptions during dispose."""
    mock_env = MagicMock()
    mock_env.dispose.side_effect = Exception("dispose error")
    original_env = lp._GUROBI_ENV

    try:
        lp._GUROBI_ENV = mock_env
        # Should not raise even though dispose throws
        lp._cleanup_gurobi_env()
        assert lp._GUROBI_ENV is None
    finally:
        lp._GUROBI_ENV = original_env


@requires_gurobi
def test_solve_gurobi_infeasible():
    """Test that infeasible problems raise RuntimeError."""
    # Infeasible: minimize x s.t. x <= -1 and x >= 1
    c = np.array([1, 0])
    A = np.array([[1, 0], [-1, 0]])
    b = np.array([-1, -1])  # x <= -1 and -x <= -1 (x >= 1) → infeasible
    with pytest.raises(RuntimeError, match="No solution found"):
        lp.solve_gurobi(c, A, b)


def test_solve_scipy_returns_tuple(simple_lp_problem):
    c, A, b, expected_x = simple_lp_problem
    result = lp.solve_scipy(c, A, b)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert np.isclose(result[0], expected_x[0])
    assert np.isclose(result[1], expected_x[1])


@requires_gurobi
def test_solve_default_gurobi(simple_lp_problem):
    """Test that solve() defaults to Gurobi."""
    c, A, b, expected_x = simple_lp_problem
    x = lp.solve(c, A, b)
    assert np.allclose(x, expected_x)


def test_solve_with_scipy(simple_lp_problem):
    """Test that solve() dispatches to scipy when requested."""
    c, A, b, expected_x = simple_lp_problem
    x = lp.solve(c, A, b, solver=lp.Solver.SCIPY)
    assert np.allclose(x, expected_x)


def test_solve_explicit_scipy(simple_lp_problem):
    c, A, b, expected_x = simple_lp_problem
    x = lp.solve(c, A, b, solver=lp.Solver.SCIPY)
    assert np.allclose(x, expected_x)


@requires_gurobi
def test_solve_explicit_gurobi(simple_lp_problem):
    c, A, b, expected_x = simple_lp_problem
    x = lp.solve(c, A, b, solver=lp.Solver.GUROBI)
    assert np.allclose(x, expected_x)


@requires_gurobi
def test_solvers_agree(simple_lp_problem):
    """Test that both solvers produce the same result."""
    c, A, b, _ = simple_lp_problem
    scipy_result = lp.solve(c, A, b, solver=lp.Solver.SCIPY)
    gurobi_result = lp.solve(c, A, b, solver=lp.Solver.GUROBI)
    assert np.allclose(scipy_result, gurobi_result)


def test_solve_unknown_solver(simple_lp_problem):
    """Test that solve() raises ValueError for an unknown solver."""
    c, A, b, _ = simple_lp_problem
    with pytest.raises(ValueError, match="Unknown solver"):
        lp.solve(c, A, b, solver="invalid")
