# Copyright (c) 2022 Mathis Gerdes
# Licensed under the MIT license (see LICENSE for details).

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
from .util import cyclic_corr
from functools import partial
import math


@partial(jax.jit, static_argnames=('average',))
def two_point(phis: jnp.ndarray, average: bool = True) -> jnp.ndarray:
    """Estimate ``G(x) = <phi(0) phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
        ``mean_y <phi(y) phi(x+y)>``
    using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.
        average: If false, average over samples is not executed.

    Returns:
        Array of shape ``(L_1, ..., L_d)`` if ``average`` is true, otherwise
        of shape ``(batch size, L_1, ..., L_d)``.
    """
    corr = jax.vmap(cyclic_corr)(phis, phis)
    return jnp.mean(corr, axis=0) if average else corr


@jax.jit
def two_point_central(phis: jnp.ndarray) -> jnp.ndarray:
    """Estimate ``G_c(x) = <phi(0) phi(x)> - <phi(0)> <phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
        ``mean_y <phi(y) phi(x+y)> - <phi(x)> mean_y <phi(x+y)>``
    using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.

    Returns:
        Array of shape ``(L_1, ..., L_d)``.
    """
    phis_mean = jnp.mean(phis, axis=0)
    outer = phis_mean * jnp.mean(phis_mean)

    return two_point(phis, True) - outer


@jax.jit
def correlation_length(G):
    """Estimator for the correlation length.

    Args:
        G: Centered two-point function.

    Returns:
        Scalar. Estimate of correlation length.
    """
    Gs = jnp.mean(G, axis=0)
    arg = (jnp.roll(Gs, 1) + jnp.roll(Gs, -1)) / (2 * Gs)
    mp = jnp.arccosh(arg[1:])
    return 1 / jnp.nanmean(mp)


@jax.jit
def ising_action(x: jnp.ndarray,
                 Kinv: jnp.ndarray) -> jnp.ndarray:
    """Compute the Euclidean action for the Ising theory after 
       Hubbard-Stratonovich transformation.

       0.5 * x @ Kinv @ x - sum logcosh(x)

    Args:
        x: Single field configuration of shape L^d.

    Returns:
        Scalar, the action of the field configuration..
    """
    x = jnp.ravel(x)
    return jnp.sum(0.5 * jnp.matmul(x, Kinv) * x \
           - (jax.nn.softplus(2.*x) - x - math.log(2.)))


@jax.jit
def ising_action_x(x: jnp.ndarray,
                   K: jnp.ndarray) -> jnp.ndarray:
    """Compute the Euclidean action for the Ising theory after 
       Hubbard-Stratonovich transformation.

       0.5 * x @ K @ x - sum logcosh(Kx)

    Args:
        x: Single field configuration of shape L^d.

    Returns:
        Scalar, the action of the field configuration..
    """
    x = jnp.ravel(x)
    Kx = jnp.matmul(x, K)
    return 0.5 * jnp.sum(x*Kx) - jnp.sum(jax.nn.softplus(2.*Kx) - Kx - math.log(2.))


class Lattice:
    def __init__(self,L, d, BC='periodic'):
        self.L = L 
        self.d = d
        self.shape = [L]*d 
        self.Nsite = L**d 
        self.BC = BC

    def move(self, idx, d, shift):
        coord = self.index2coord(idx)
        coord[d] += shift

        if self.BC != 'periodic':
            if (coord[d]>=self.L) or (coord[d]<0):
                return None
        #wrap around because of the PBC
        if (coord[d]>=self.L): coord[d] -= self.L; 
        if (coord[d]<0): coord[d] += self.L; 

        return self.coord2index(coord)

    def index2coord(self, idx):
        coord = np.zeros(self.d, int) 
        for d in range(self.d):
            coord[self.d-d-1] = idx%self.L;
            idx /= self.L
        return coord 

    def coord2index(self, coord):
        idx = coord[0]
        for d in range(1, self.d):
            idx *= self.L; 
            idx += coord[d]
        return idx 


class Hypercube(Lattice):
    def __init__(self,L, d, BC='periodic'):
        super(Hypercube, self).__init__(L, d, BC)
        self.Adj = np.zeros((self.Nsite,self.Nsite), int)
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)

                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0


@chex.dataclass
class IsingTheory:
    """Ising theory after Hubbard-Stratonovich transformation."""
    Kinv: jnp.ndarray
    L: chex.Scalar
    dim: chex.Scalar = 2

    @property
    def lattice_size(self):
        return self.L ** self.dim

    def action(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the Ising action.

        Args:
            x: Either a single field configuration of shape L^d or
               a batch of those field configurations.

        Returns:
            Either a scalar value or a 1d array of actions for the
            field configuration(s).
        """

        # check whether x are a batch or a single sample
        if x.ndim == self.dim:
            chex.assert_shape(x, [self.L] * self.dim)
            return ising_action(x, self.Kinv)
        else:
            chex.assert_shape(x[0], [self.L] * self.dim)
            act = partial(ising_action, Kinv=self.Kinv)
            return jax.vmap(act)(x)


@chex.dataclass
class IsingTheory_T:
    """Ising theory for different temperatures after Hubbard-Stratonovich transformation."""
    Adj: jnp.ndarray
    L: chex.Scalar
    T: chex.Scalar = None
    dim: chex.Scalar = 2
    eps: chex.Scalar = 0.1

    @property
    def lattice_size(self):
        return self.L ** self.dim

    def action(self, x: jnp.ndarray, T: chex.Scalar = None) -> jnp.ndarray:
        """Compute the Ising action.

        Args:
            x: Either a single field configuration of shape L^d or
               a batch of those field configurations.

        Returns:
            Either a scalar value or a 1d array of actions for the
            field configuration(s).
        """
        T = self.T if T is None else T

        K = self.Adj / T
        w, v = jnp.linalg.eigh(K)
        offset = self.eps-jnp.min(w)
        K += jnp.eye(w.size)*offset
        Kinv = jnp.linalg.inv(K)

        # check whether x are a batch or a single sample
        if x.ndim == self.dim:
            chex.assert_shape(x, [self.L] * self.dim)
            return ising_action(x, Kinv)
        else:
            chex.assert_shape(x[0], [self.L] * self.dim)
            act = partial(ising_action, Kinv=Kinv)
            return jax.vmap(act)(x)


@chex.dataclass
class IsingTheory_x:
    """Ising theory after Hubbard-Stratonovich transformation."""
    K: jnp.ndarray
    L: chex.Scalar
    dim: chex.Scalar = 2

    @property
    def lattice_size(self):
        return self.L ** self.dim

    def action(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the Ising action.

        Args:
            x: Either a single field configuration of shape L^d or
               a batch of those field configurations.

        Returns:
            Either a scalar value or a 1d array of actions for the
            field configuration(s).
        """

        # check whether x are a batch or a single sample
        if x.ndim == self.dim:
            chex.assert_shape(x, [self.L] * self.dim)
            return ising_action_x(x, self.K)
        else:
            chex.assert_shape(x[0], [self.L] * self.dim)
            act = partial(ising_action_x, K=self.K)
            return jax.vmap(act)(x)


@chex.dataclass
class IsingTheory_xT:
    """Ising theory for different temperatures after Hubbard-Stratonovich transformation."""
    Adj: jnp.ndarray
    L: chex.Scalar
    T: chex.Scalar = None
    dim: chex.Scalar = 2
    eps: chex.Scalar = 1e-4

    @property
    def lattice_size(self):
        return self.L ** self.dim

    def action(self, x: jnp.ndarray, T: chex.Scalar = None) -> jnp.ndarray:
        """Compute the Ising action.

        Args:
            x: Either a single field configuration of shape L^d or
               a batch of those field configurations.

        Returns:
            Either a scalar value or a 1d array of actions for the
            field configuration(s).
        """
        T = self.T if T is None else T

        K = self.Adj / T
        w, v = jnp.linalg.eigh(K)
        offset = self.eps - jnp.min(w)
        K += jnp.eye(w.size) * offset

        # check whether x are a batch or a single sample
        if x.ndim == self.dim:
            chex.assert_shape(x, [self.L] * self.dim)
            return ising_action_x(x, K)
        else:
            chex.assert_shape(x[0], [self.L] * self.dim)
            act = partial(ising_action_x, K=K)
            return jax.vmap(act)(x)
