import numpy as np


class SampleMPC:
    """
    For more description of the problem, see the instructional material in folder
    """

    def __init__(self, Np=10, k_q=1, k_r=0.1):
        # Problem is well defined and only time horizon needs to be changed

        # Problem definition
        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([[0.5], [1]])
        self.C = np.array([1, 0])
        self.D = np.array([0])

        # Dimensions
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.ny = len(self.C.shape)

        # Prediction horizon
        self.Np = Np

        # Weights
        self.Q = k_q * np.eye(self.Np)
        self.R = k_r * np.eye(self.Np)

    def init_augmented_matrix(self, k_q=1, k_r=0.1):
        # Computed Penalty Weights Parameters
        R = k_r * np.eye(self.Np)
        Q = k_q * np.eye(self.Np)

        # Reference point
        Ref = 2 * np.ones((self.Np))  # reference point over the prediction horizon

        ## Initial Conditions
        self.uk = 0
        self.xk = np.zeros(self.nx)
        self.yk = self.C @ self.xk
        self.xa = np.hstack([self.xk, self.yk])

        ## Augmented Matrix - integral action approach
        Phi_a = np.vstack(
            [
                np.hstack([self.A, np.zeros((self.nx, self.ny))]),
                np.hstack([(self.C @ self.A)[None, :], np.eye(self.ny)]),
            ]
        )

        self.Gamma_a = np.vstack([self.B, self.C @ self.B])

        self.C_a = np.hstack([np.zeros((self.ny, self.nx)), np.eye(self.ny)])

        [self.r1, self.c1] = self.C_a.shape  # get the size of C_a for creating Z,W
        [self.a1, self.b1] = self.Gamma_a.shape

        # Precompute matrix powers
        self.Phi_powers = np.r_[
            [np.linalg.matrix_power(Phi_a, i) for i in range(self.Np + 1)]
        ]

        # Solve W and Z
        self.W = (self.C_a @ self.Phi_powers[1:]).reshape(self.Np * self.ny, -1)
        self.Z = self.precalc_Z()

        ## Solve H and E Matrices
        # U = E*u(k-1) + H*DU
        Inu = np.eye(self.nu)  # identity matrix the size of uk to solve H and E

        self.E = np.repeat(Inu, self.Np, axis=0)
        self.H = np.tril(np.tile(Inu, (self.Np, self.Np)))

        return "Completed"

    def precalc_Z(self):
        Z = np.zeros((self.Np * self.ny, self.Np * self.nu))
        Z[:, : self.nu] = (self.C_a @ self.Phi_powers[:-1] @ self.Gamma_a)[:, 0]
        n = 1
        for i in range(self.b1, self.Np * self.b1):
            Z[i : self.Np * self.r1, i : i + self.b1] = Z[
                : self.Np * self.r1 - n * self.r1, : self.b1
            ]
            n = n + 1
        return Z

    def init_system_matrix(self, uk=None, xa=None, xo=None):
        """Returns final A and B matrix for optimization model"""
        # Optimization problem
        if uk is None:
            uk = self.uk
        if xa is not None:
            self.xa = xa
        if xo is None:
            self.xo = np.ones((self.Np * self.nu)) * uk  # initial guess
        else:
            self.xo = xo

        # Constraints

        self.dumin = 0.5
        self.dumax = 0.5

        self.umin = -0.5
        self.umax = 1
        self.ymin = 0
        self.ymax = 3

        bu1 = -self.umin + np.dot(self.E, uk)
        bu2 = self.umax - np.dot(self.E, uk)
        by1 = -self.ymin + np.dot(self.W, self.xa)
        by2 = self.ymax - np.dot(self.W, self.xa)

        # Inequality constraints

        self.Ao = np.vstack((-self.H, self.H, -self.Z, self.Z))
        self.b = np.vstack(
            (bu1, bu2, by1.reshape(by1.shape[0], -1), by2.reshape(by2.shape[0], -1))
        )

        # Bounds
        self.lb = -0.5 * np.ones((self.Np * self.nu))
        self.ub = 0.5 * np.ones((self.Np * self.nu))

        return (self.Ao, self.b), (self.lb, self.ub)
