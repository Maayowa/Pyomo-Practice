import pyomo.environ as pyo
import numpy as np


# Output Cost
def lhs_obj(W, Q, Z, xa, u, ref=None):
    result = (W @ xa) + np.r_[
        [
            sum([Z[row, col] * u[col] for col in range(Z.shape[1])])
            for row in range(Z.shape[0])
        ]
    ]
    res = (ref - result)[:, None]

    return (
        sum(
            sum(res[row] * Q[col, row] for row in range(Q.shape[0])) * res[col]
            for col in range(Q.shape[1])
        )
    )[0]


# Input Cost
def rhs_obj(R, u):
    return sum(
        sum(u[row] * R[col, row] for row in range(R.shape[0])) * u[col]  # ()
        for col in range(R.shape[1])
    )


def cost_func(m):
    return lhs_obj(m.W, m.Q, m.Z, m.xa, m.u, ref=m.Ref) / 2 + rhs_obj(m.R, m.u) / 2


def mpc_lti_system(aug_mat=None, Np=10):
    m = pyo.ConcreteModel()

    # Indices
    m.k = pyo.RangeSet(0, Np - 1)
    m.Np = Np

    # Parameters
    m.Ref = 2 * np.ones((Np))
    m.Q = aug_mat.Q  # ["Q"] # Penalty parameter1 checked
    m.R = aug_mat.R  # ["R"] # Penalty parameter2 checked
    m.Z = aug_mat.Z  # ["Z"] #
    m.W = aug_mat.W  # ["W"]
    m.A = aug_mat.Ao  # ["Ao"]
    m.b = aug_mat.b  # ["b"]
    m.xa = aug_mat.xa  # ["x_a"]

    # Variables
    m.u = pyo.Var(m.k, initialize=aug_mat.xo, bounds=(-0.5, 0.5))

    # Constraints
    m.const = pyo.ConstraintList()
    for row in range(m.A.shape[0]):
        m.const.add(
            expr=sum(m.A[row, col] * m.u[col] for col in range(m.A.shape[1]))
            <= m.b[row][0]
        )

    # Objective
    m.obj = pyo.Objective(rule=cost_func, sense=pyo.minimize)

    return m
