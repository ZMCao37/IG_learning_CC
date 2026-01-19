"""
Statistical Manifold Classes

This module provides classes for working with statistical manifolds,
including computation of:
- Fisher metric (Riemannian metric)
- Geodesics
- Parallel transport
- Curvature
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, List
from .distributions import GaussianDistribution


class StatisticalManifold(ABC):
    """
    Abstract base class for statistical manifolds.

    A statistical manifold is a space of probability distributions
    equipped with the Fisher information metric.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the manifold."""
        pass

    @abstractmethod
    def fisher_metric(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Fisher information matrix at point theta.

        Parameters:
        -----------
        theta : np.ndarray
            Point on the manifold (parameter values)

        Returns:
        --------
        np.ndarray of shape (dim, dim)
            Fisher information matrix (positive definite)
        """
        pass

    def metric_tensor(self, theta: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Compute inner product <v, w>_θ using Fisher metric.

        Parameters:
        -----------
        theta : np.ndarray
            Point on the manifold
        v, w : np.ndarray
            Tangent vectors at theta

        Returns:
        --------
        float
            Inner product g_θ(v, w) = v^T I(θ) w
        """
        I = self.fisher_metric(theta)
        return float(v @ I @ w)

    def norm(self, theta: np.ndarray, v: np.ndarray) -> float:
        """
        Compute norm ||v||_θ using Fisher metric.
        """
        return np.sqrt(self.metric_tensor(theta, v, v))

    def distance_infinitesimal(self, theta: np.ndarray, dtheta: np.ndarray) -> float:
        """
        Compute infinitesimal distance ds² = dθ^T I(θ) dθ.
        """
        return self.norm(theta, dtheta)

    @abstractmethod
    def christoffel_symbols(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^k_ij at point theta.

        Returns:
        --------
        np.ndarray of shape (dim, dim, dim)
            Christoffel symbols where result[k, i, j] = Γ^k_ij
        """
        pass

    def geodesic_equation(self, theta: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute geodesic acceleration: d²θ^k/dt² = -Γ^k_ij (dθ^i/dt)(dθ^j/dt)

        Parameters:
        -----------
        theta : np.ndarray
            Current position on manifold
        v : np.ndarray
            Current velocity (tangent vector)

        Returns:
        --------
        np.ndarray
            Acceleration (second derivative of geodesic)
        """
        Gamma = self.christoffel_symbols(theta)
        n = self.dim
        acc = np.zeros(n)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    acc[k] -= Gamma[k, i, j] * v[i] * v[j]
        return acc

    def geodesic(self, theta0: np.ndarray, v0: np.ndarray,
                 t_max: float = 1.0, n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute geodesic starting from theta0 with initial velocity v0.

        Uses simple Euler integration (can be improved with RK4).

        Parameters:
        -----------
        theta0 : np.ndarray
            Starting point
        v0 : np.ndarray
            Initial velocity (tangent vector)
        t_max : float
            Maximum time
        n_steps : int
            Number of integration steps

        Returns:
        --------
        t : np.ndarray of shape (n_steps,)
            Time values
        path : np.ndarray of shape (n_steps, dim)
            Points along geodesic
        """
        dt = t_max / n_steps
        t = np.linspace(0, t_max, n_steps)
        path = np.zeros((n_steps, self.dim))

        theta = theta0.copy()
        v = v0.copy()
        path[0] = theta

        for i in range(1, n_steps):
            # Euler integration of geodesic equation
            acc = self.geodesic_equation(theta, v)
            theta = theta + v * dt
            v = v + acc * dt
            path[i] = theta

        return t, path


class GaussianManifold(StatisticalManifold):
    """
    Statistical manifold of univariate Gaussian distributions N(μ, σ²).

    This is a 2-dimensional Riemannian manifold with:
    - Coordinates: (μ, σ) where μ ∈ ℝ, σ > 0
    - Fisher metric: I = [[1/σ², 0], [0, 2/σ²]]

    The manifold is isometric to the hyperbolic plane (Poincaré half-plane)
    with constant negative curvature K = -1/2.
    """

    @property
    def dim(self) -> int:
        return 2

    def fisher_metric(self, theta: np.ndarray) -> np.ndarray:
        """
        Fisher information matrix at point (μ, σ).

        I(μ, σ) = [[1/σ², 0], [0, 2/σ²]]
        """
        mu, sigma = theta
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        return np.array([
            [1 / sigma**2, 0],
            [0, 2 / sigma**2]
        ])

    def fisher_metric_inverse(self, theta: np.ndarray) -> np.ndarray:
        """
        Inverse Fisher metric.

        I^{-1} = [[σ², 0], [0, σ²/2]]
        """
        mu, sigma = theta
        return np.array([
            [sigma**2, 0],
            [0, sigma**2 / 2]
        ])

    def christoffel_symbols(self, theta: np.ndarray) -> np.ndarray:
        """
        Christoffel symbols for the Gaussian manifold.

        Non-zero symbols:
        - Γ^1_12 = Γ^1_21 = -1/σ
        - Γ^2_11 = 1/(2σ)
        - Γ^2_22 = -1/σ
        """
        mu, sigma = theta
        Gamma = np.zeros((2, 2, 2))

        # Γ^μ_μσ = Γ^μ_σμ = -1/σ
        Gamma[0, 0, 1] = -1 / sigma
        Gamma[0, 1, 0] = -1 / sigma

        # Γ^σ_μμ = 1/(2σ)
        Gamma[1, 0, 0] = 1 / (2 * sigma)

        # Γ^σ_σσ = -1/σ
        Gamma[1, 1, 1] = -1 / sigma

        return Gamma

    def riemann_curvature(self, theta: np.ndarray) -> np.ndarray:
        """
        Riemann curvature tensor R^l_ijk.

        For Gaussian manifold, the scalar curvature is K = -1/2 (constant).
        """
        mu, sigma = theta
        R = np.zeros((2, 2, 2, 2))
        # The non-zero components encode curvature -1/2
        # R^μ_σμσ = -1/(2σ²)
        R[0, 1, 0, 1] = -1 / (2 * sigma**2)
        R[0, 1, 1, 0] = 1 / (2 * sigma**2)
        R[1, 0, 0, 1] = 1 / (2 * sigma**2)
        R[1, 0, 1, 0] = -1 / (2 * sigma**2)
        return R

    def scalar_curvature(self, theta: np.ndarray) -> float:
        """
        Scalar curvature (constant for Gaussian manifold).

        K = -1/2
        """
        return -0.5

    def kl_divergence(self, theta1: np.ndarray, theta2: np.ndarray) -> float:
        """
        KL divergence D_KL(N(μ1,σ1²) || N(μ2,σ2²)).
        """
        mu1, sigma1 = theta1
        mu2, sigma2 = theta2
        return (np.log(sigma2 / sigma1)
                + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
                - 0.5)

    def fisher_rao_distance(self, theta1: np.ndarray, theta2: np.ndarray) -> float:
        """
        Fisher-Rao distance (geodesic distance) between two Gaussians.

        For the Gaussian manifold (isometric to hyperbolic plane):
        d(p1, p2) = √2 · arccosh(1 + (μ1-μ2)²/(2σ1σ2) + (σ1²+σ2²)/(2σ1σ2) - 1)

        Simplified formula using hyperbolic distance.
        """
        mu1, sigma1 = theta1
        mu2, sigma2 = theta2

        # Using the formula for hyperbolic distance in the Poincaré half-plane model
        # scaled appropriately for the Fisher metric
        delta_mu = mu1 - mu2
        term = 1 + (delta_mu**2 + (sigma1 - sigma2)**2) / (2 * sigma1 * sigma2)
        return np.sqrt(2) * np.arccosh(term)

    def exp_map(self, theta: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: Exp_θ(v) = γ(1) where γ is geodesic with γ(0)=θ, γ'(0)=v.

        For small v, approximately θ + v.
        """
        _, path = self.geodesic(theta, v, t_max=1.0, n_steps=100)
        return path[-1]

    def parallel_transport(self, theta: np.ndarray, v: np.ndarray,
                           path: np.ndarray) -> List[np.ndarray]:
        """
        Parallel transport vector v along path.

        Uses simple Euler integration of parallel transport equation:
        dV^k/dt + Γ^k_ij V^i (dγ^j/dt) = 0
        """
        n_steps = len(path)
        transported = [v.copy()]

        for i in range(1, n_steps):
            theta_curr = path[i-1]
            theta_next = path[i]
            dtheta = theta_next - theta_curr

            Gamma = self.christoffel_symbols(theta_curr)
            V = transported[-1].copy()

            # Update rule from parallel transport equation
            dV = np.zeros(self.dim)
            for k in range(self.dim):
                for l in range(self.dim):
                    for j in range(self.dim):
                        dV[k] -= Gamma[k, l, j] * V[l] * dtheta[j]

            transported.append(V + dV)

        return transported

    def get_distribution(self, theta: np.ndarray) -> GaussianDistribution:
        """Get GaussianDistribution object for given parameters."""
        return GaussianDistribution(mu=theta[0], sigma=theta[1])

    def __repr__(self) -> str:
        return "GaussianManifold(dim=2)"


# Utility functions for manifold operations

def fisher_metric_numerical(dist_class, theta: np.ndarray,
                            n_samples: int = 100000, eps: float = 1e-6) -> np.ndarray:
    """
    Numerically estimate Fisher information matrix using Monte Carlo.

    I_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j]
    """
    # Create distribution
    dist = dist_class(*theta)

    # Sample
    samples = dist.sample(n_samples)

    # Compute score function
    scores = dist.score(samples)

    # Estimate Fisher matrix as covariance of score
    I = np.cov(scores.T)

    return I
