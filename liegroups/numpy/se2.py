import numpy as np

from . import base
from .so2 import SO2


class SE2(base.SpecialEuclideanBase):
    """Homogeneous transformation matrix in SE(2) using active (alibi) transformations."""
    dim = 3
    dof = 3
    RotationType = SO2

    def __init__(self, rot, trans):
        super().__init__(rot, trans)

    @classmethod
    def wedge(cls, xi):
        xi = np.atleast_2d(xi)
        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = np.zeros([xi.shape[0], cls.dof, cls.dof])
        Xi[:, 0:2, 0:2] = cls.RotationType.wedge(xi[:, 2])
        Xi[:, 0:2, 2] = xi[:, 0:2]

        return np.squeeze(Xi)

    @classmethod
    def vee(cls, Xi):
        if Xi.ndim < 3:
            Xi = np.expand_dims(Xi, axis=0)

        if Xi.shape[1:3] != (cls.dof, cls.dof):
            raise ValueError("Xi must have shape ({},{}) or (N,{},{})".format(
                cls.dof, cls.dof, cls.dof, cls.dof))

        xi = np.empty([Xi.shape[0], cls.dof])
        xi[:, 0:2] = Xi[:, 0:2, 2]
        xi[:, 2] = cls.RotationType.vee(Xi[:, 0:2, 0:2])
        return np.squeeze(xi)

    @classmethod
    def left_jacobian(cls, xi):
        raise NotImplementedError

    @classmethod
    def inv_left_jacobian(cls, xi):
        raise NotImplementedError

    @classmethod
    def exp(cls, xi):
        if len(xi) != cls.dof:
            raise ValueError("xi must have length {}".format(cls.dof))

        rho = xi[0:2]
        phi = xi[2]
        return cls(cls.RotationType.exp(phi),
                   cls.RotationType.left_jacobian(phi).dot(rho))

    def log(self):
        phi = self.rot.log()
        rho = self.RotationType.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    def adjoint(self):
        rot_part = self.rot.as_matrix()
        trans_part = np.array([self.trans[1], -self.trans[0]]).reshape((2, 1))
        return np.vstack([np.hstack([rot_part, trans_part]),
                          [0, 0, 1]])

    @classmethod
    def odot(cls, p, directional=False):
        p = np.atleast_2d(p)
        result = np.zeros([p.shape[0], p.shape[1], cls.dof])

        if p.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless p is a direction
            # vector, in which case the scale is 0
            if not directional:
                result[:, 0:2, 0:2] = np.eye(2)

            result[:, 0:2, 2] = cls.RotationType.wedge(1).dot(p.T).T

        elif p.shape[1] == cls.dim:
            result[:, 0:2, 0:2] = p[:, 2] * np.eye(2)
            result[:, 0:2, 2] = cls.RotationType.wedge(1).dot(p[:, 0:2].T).T

        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return np.squeeze(result)