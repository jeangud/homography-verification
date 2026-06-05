"""Linear programming solvers."""

import atexit
import logging
from enum import Enum

import gurobipy as gp
import scipy.optimize as opt
from gurobipy import GRB

LOGGER = logging.getLogger(__name__)


class Solver(Enum):
    GUROBI = "gurobi"
    SCIPY = "scipy"


# Singleton Gurobi environment for efficiency
_GUROBI_ENV = None


def _get_gurobi_env():
    """Get or create a Gurobi environment singleton."""
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam("OutputFlag", 0)
        _GUROBI_ENV.setParam("LogToConsole", 0)
        _GUROBI_ENV.start()
    return _GUROBI_ENV


def _cleanup_gurobi_env():
    """Clean up the Gurobi environment at program exit."""
    global _GUROBI_ENV
    if _GUROBI_ENV is not None:
        try:
            _GUROBI_ENV.dispose()
        except Exception:
            LOGGER.debug(
                "Failed to dispose Gurobi environment during shutdown", exc_info=True
            )
        _GUROBI_ENV = None


# Properly clean up memory upon exit
atexit.register(_cleanup_gurobi_env)


def solve(c, A, b, solver: Solver = Solver.GUROBI):
    """Solve a linear program using the specified solver."""
    if solver == Solver.GUROBI:
        return solve_gurobi(c, A, b)
    elif solver == Solver.SCIPY:
        return solve_scipy(c, A, b)
    else:
        raise ValueError(f"Unknown solver: {solver}")


def solve_scipy(c, A, b):
    """Solve a linear program using scipy."""
    # Set bounds to (None, None) to allow negative params (slope, intercept)
    res = opt.linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * len(c))
    return res.x[0], res.x[1]


def solve_gurobi(c, A, b):
    """Solve a linear program using Gurobi."""
    env = _get_gurobi_env()

    with gp.Model(env=env) as model:
        model.setParam("OutputFlag", 0)

        # Define decision variables
        n = len(c)
        x = model.addVars(n, name="x", lb=-GRB.INFINITY)

        # Set objective function
        model.setObjective(gp.LinExpr(c, [x[i] for i in range(n)]), sense=GRB.MINIMIZE)

        # Add inequality constraints
        for i in range(len(A)):
            model.addConstr(
                gp.LinExpr(A[i], [x[j] for j in range(n)]) <= b[i],
                f"constraint_{i}",
            )

        # Optimize the model
        model.optimize()

        # Results
        if model.status != GRB.OPTIMAL:
            raise RuntimeError("No solution found.")

        # Extract solution
        solution = model.getVars()[0].X, model.getVars()[1].X

    # NOTE: we don't return model.* directly because we need to free the memory
    return solution
