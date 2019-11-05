from cvxopt import matrix
from cvxopt import solvers
from cvxopt.modeling import variable, op
import numpy as np
from numpy import ndarray

from typing import List


def main() -> None:
    midpoint_minimization()

    assert False

    # standard form:
    # minimize: x    (1/2) x^(T) Px + q^(T)x

    # subject to: Gx element-wise< h
    #  " " "    : Ax = b

    # objective function:
    # quadratic coefficients (square matrix of all vars to be minimized)
    p = matrix(np.diag(np.array([1, 0])), tc="d")
    # linear coefficients
    q = matrix(np.array([3, 4]), tc="d")
    # constraints:
    # left hand side of inequality (coefficients of x and y terms)
    g = matrix(np.array([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]]), tc="d")
    # right hand side of inequality (scalars)
    h = matrix(np.array([0, 0, -15, 100, 80]), tc="d")

    sol = solvers.qp(p, q, g, h)
    print(sol["x"])


class PositionSolver:
    POSITION_CONSTRAINTS = {
        "X_RANGE": (-10.0, 10.0),
        "Y_RANGE": (-10.0, 5.0),
        "Z_RANGE_GROUND": (0.0, 0.0),
        "Z_RANGE_FLY": (2.0, 5.0),
        "Z_RANGE_OTHER": (0.0, 3.0),
    }

    CLOSE_DISTANCE = 0.25

    def __init__(self, main_point: ndarray, points: List[ndarray]) -> None:
        self.main_position = main_point
        self.positions = points


class RelativePositionSolver:
    pass


def midpoint_minimization():
    # objective function:

    # quadratic components: none
    # P = matrix(np.diag(np.array([0, 0, 0])), tc="d")
    # linear coefficients: all ones
    # q = matrix(np.array([1, 1, 1]), tc="d")
    # these are absolute value stand-ins u, v, w

    # u = variable(1, 'u')
    # v = variable(1, 'v')
    # w = variable(1, 'w')

    x = variable(1, "x")
    y = variable(1, "y")
    z = variable(1, "z")

    f = abs(x) + abs(y) + abs(z)

    print(f)

    # constraints:
    c_x_1 = x <= 10.0
    c_x_2 = x >= -10.0
    c_y_1 = y <= 5.0
    c_y_2 = y >= -5.0
    c_z_1 = z <= 2.0
    c_z_2 = z >= 0.0

    prob = op(f, [c_x_1, c_x_2, c_y_1, c_y_2, c_z_1, c_z_2], "test_prob")
    prob.solve()
    print(prob.status)

    # G =


if __name__ == "__main__":
    main()
