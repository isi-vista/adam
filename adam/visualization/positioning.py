from cvxopt import matrix
from cvxopt import solvers
import numpy as np

from typing import(
    List,
    Tuple,
)
import attr
from attr import attrs
from math import sqrt
from adam.visualization.utils import Vector3


class PositionSolver:
    POSITION_CONSTRAINTS = {
        "X_RANGE": (-10.0, 10.0),
        "Y_RANGE": (-10.0, 5.0),
        "Z_RANGE_GROUND": (0.0, 0.0),
        "Z_RANGE_FLY": (2.0, 5.0),
        "Z_RANGE_OTHER": (0.0, 3.0),
    }

    CLOSE_DISTANCE = 0.25

    def __init__(self, main_point: Vector3, points: List[Vector3]):
        self.main_position = main_point
        self.positions = points


class RelativePositionSolver:
    pass


if __name__ == "__main__":

    # standard form:
    # minimize: x    (1/2) x^(T) Px + q^(T)x

    # subject to: Gx element-wise< h
    #  " " "    : Ax = b

    # objective function:
    # quadratic coefficients (square matrix of all vars to be minimized)
    P = matrix(np.diag(np.array([1, 0])), tc='d')
    # linear coefficients
    q = matrix(np.array([3, 4]), tc='d')
    # constraints:
    # left hand side of inequality (coefficients of x and y terms)
    G = matrix(np.array([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]]), tc='d')
    # right hand side of inequality (scalars)
    h = matrix(np.array([0, 0, -15, 100, 80]), tc='d')

    sol = solvers.qp(P, q, G, h)
    print(sol['x'])
