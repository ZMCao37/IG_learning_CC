"""
Probability Distribution Classes with Information Geometry Properties

This module provides probability distribution classes that include:
- Probability density/mass functions
- Log-likelihood functions
- Score functions (gradient of log-likelihood)
- Fisher information matrices
- KL divergence calculations
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional
import warnings


class Distribution(ABC):
    """Abstract base class for probability distributions."""

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density/mass function."""
        pass

    @abstractmethod
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Log probability density/mass function."""
        pass

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray:
        """Score function: gradient of log-likelihood w.r.t. parameters."""
        pass

    @abstractmethod
    def fisher_information(self) -> np.ndarray:
        """Fisher information matrix."""
        pass

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Generate random samples."""
        pass


class GaussianDistribution(Distribution):
    """
    Univariate Gaussian Distribution N(μ, σ²)

    Parameters:
    -----------
    mu : float
        Mean parameter
    sigma : float
        Standard deviation (must be positive)

    Information Geometry Properties:
    - Fisher information matrix: I = [[1/σ², 0], [0, 2/σ²]]
    - Natural parameters: θ = (μ/σ², -1/(2σ²))
    - Expectation parameters: η = (μ, μ² + σ²)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self._params = np.array([mu, sigma])

    @property
    def params(self) -> np.ndarray:
        """Return parameters as array [μ, σ]."""
        return self._params

    @params.setter
    def params(self, value: np.ndarray):
        """Set parameters from array [μ, σ]."""
        if len(value) != 2:
            raise ValueError("Expected 2 parameters [mu, sigma]")
        if value[1] <= 0:
            raise ValueError("sigma must be positive")
        self.mu = value[0]
        self.sigma = value[1]
        self._params = np.array(value)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function."""
        x = np.asarray(x)
        coef = 1 / (self.sigma * np.sqrt(2 * np.pi))
        return coef * np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Log probability density function."""
        x = np.asarray(x)
        return (-np.log(self.sigma) - 0.5 * np.log(2 * np.pi)
                - 0.5 * ((x - self.mu) / self.sigma) ** 2)

    def score(self, x: np.ndarray) -> np.ndarray:
        """
        Score function: [∂log p/∂μ, ∂log p/∂σ]

        Returns:
        --------
        np.ndarray of shape (n, 2) where n = len(x)
        """
        x = np.asarray(x)
        score_mu = (x - self.mu) / self.sigma**2
        score_sigma = -1/self.sigma + (x - self.mu)**2 / self.sigma**3
        return np.column_stack([score_mu, score_sigma])

    def fisher_information(self) -> np.ndarray:
        """
        Fisher information matrix.

        I(μ, σ) = [[1/σ², 0], [0, 2/σ²]]
        """
        return np.array([
            [1 / self.sigma**2, 0],
            [0, 2 / self.sigma**2]
        ])

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate n random samples."""
        return np.random.normal(self.mu, self.sigma, n)

    def kl_divergence(self, other: 'GaussianDistribution') -> float:
        """
        KL divergence D_KL(self || other).

        D_KL(N(μ1,σ1²) || N(μ2,σ2²)) =
            log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        """
        return (np.log(other.sigma / self.sigma)
                + (self.sigma**2 + (self.mu - other.mu)**2) / (2 * other.sigma**2)
                - 0.5)

    def natural_parameters(self) -> np.ndarray:
        """
        Natural (canonical) parameters θ = (θ1, θ2).

        θ1 = μ/σ²
        θ2 = -1/(2σ²)
        """
        return np.array([self.mu / self.sigma**2, -1 / (2 * self.sigma**2)])

    def expectation_parameters(self) -> np.ndarray:
        """
        Expectation parameters η = (η1, η2).

        η1 = E[x] = μ
        η2 = E[x²] = μ² + σ²
        """
        return np.array([self.mu, self.mu**2 + self.sigma**2])

    @classmethod
    def from_natural_parameters(cls, theta: np.ndarray) -> 'GaussianDistribution':
        """Create distribution from natural parameters."""
        theta1, theta2 = theta
        sigma = np.sqrt(-1 / (2 * theta2))
        mu = theta1 * sigma**2
        return cls(mu, sigma)

    def __repr__(self) -> str:
        return f"GaussianDistribution(mu={self.mu:.4f}, sigma={self.sigma:.4f})"


class BernoulliDistribution(Distribution):
    """
    Bernoulli Distribution Ber(p)

    Parameters:
    -----------
    p : float
        Success probability (must be in (0, 1))

    Information Geometry Properties:
    - Fisher information: I(p) = 1/(p(1-p))
    - Natural parameter: θ = log(p/(1-p)) (log-odds)
    - Expectation parameter: η = p
    """

    def __init__(self, p: float = 0.5):
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")
        self.p = p

    @property
    def params(self) -> np.ndarray:
        """Return parameters as array [p]."""
        return np.array([self.p])

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability mass function."""
        x = np.asarray(x)
        return np.where(x == 1, self.p, np.where(x == 0, 1 - self.p, 0))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Log probability mass function."""
        x = np.asarray(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return x * np.log(self.p) + (1 - x) * np.log(1 - self.p)

    def score(self, x: np.ndarray) -> np.ndarray:
        """
        Score function: ∂log P/∂p = (x - p) / (p(1-p))
        """
        x = np.asarray(x)
        return ((x - self.p) / (self.p * (1 - self.p))).reshape(-1, 1)

    def fisher_information(self) -> np.ndarray:
        """
        Fisher information.

        I(p) = 1 / (p(1-p))
        """
        return np.array([[1 / (self.p * (1 - self.p))]])

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate n random samples."""
        return np.random.binomial(1, self.p, n)

    def kl_divergence(self, other: 'BernoulliDistribution') -> float:
        """KL divergence D_KL(self || other)."""
        p, q = self.p, other.p
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def natural_parameters(self) -> np.ndarray:
        """Natural parameter θ = log(p/(1-p))."""
        return np.array([np.log(self.p / (1 - self.p))])

    @classmethod
    def from_natural_parameters(cls, theta: np.ndarray) -> 'BernoulliDistribution':
        """Create distribution from natural parameters."""
        p = 1 / (1 + np.exp(-theta[0]))
        return cls(p)

    def __repr__(self) -> str:
        return f"BernoulliDistribution(p={self.p:.4f})"


class PoissonDistribution(Distribution):
    """
    Poisson Distribution Poisson(λ)

    Parameters:
    -----------
    lam : float
        Rate parameter (must be positive)

    Information Geometry Properties:
    - Fisher information: I(λ) = 1/λ
    - Natural parameter: θ = log(λ)
    - Expectation parameter: η = λ
    """

    def __init__(self, lam: float = 1.0):
        if lam <= 0:
            raise ValueError("lambda must be positive")
        self.lam = lam

    @property
    def params(self) -> np.ndarray:
        """Return parameters as array [λ]."""
        return np.array([self.lam])

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability mass function."""
        x = np.asarray(x)
        from scipy.special import factorial
        return (self.lam ** x) * np.exp(-self.lam) / factorial(x)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Log probability mass function."""
        x = np.asarray(x)
        from scipy.special import gammaln
        return x * np.log(self.lam) - self.lam - gammaln(x + 1)

    def score(self, x: np.ndarray) -> np.ndarray:
        """
        Score function: ∂log P/∂λ = x/λ - 1
        """
        x = np.asarray(x)
        return ((x / self.lam - 1)).reshape(-1, 1)

    def fisher_information(self) -> np.ndarray:
        """
        Fisher information.

        I(λ) = 1/λ
        """
        return np.array([[1 / self.lam]])

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate n random samples."""
        return np.random.poisson(self.lam, n)

    def kl_divergence(self, other: 'PoissonDistribution') -> float:
        """KL divergence D_KL(self || other)."""
        return (self.lam * np.log(self.lam / other.lam)
                - self.lam + other.lam)

    def natural_parameters(self) -> np.ndarray:
        """Natural parameter θ = log(λ)."""
        return np.array([np.log(self.lam)])

    @classmethod
    def from_natural_parameters(cls, theta: np.ndarray) -> 'PoissonDistribution':
        """Create distribution from natural parameters."""
        return cls(np.exp(theta[0]))

    def __repr__(self) -> str:
        return f"PoissonDistribution(lam={self.lam:.4f})"


class ExponentialFamily(ABC):
    """
    Abstract base class for exponential family distributions.

    p(x|θ) = h(x) exp(θᵀT(x) - ψ(θ))

    where:
    - θ: natural parameters
    - T(x): sufficient statistics
    - ψ(θ): log-partition function (cumulant generating function)
    - h(x): base measure

    Key properties:
    - E[T(x)] = ∇ψ(θ)  (expectation parameters)
    - Cov[T(x)] = ∇²ψ(θ) = I(θ)  (Fisher information)
    """

    @abstractmethod
    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Sufficient statistics T(x)."""
        pass

    @abstractmethod
    def log_partition(self) -> float:
        """Log-partition function ψ(θ)."""
        pass

    @abstractmethod
    def base_measure(self, x: np.ndarray) -> np.ndarray:
        """Base measure h(x)."""
        pass

    def log_pdf_exponential_form(self, x: np.ndarray) -> np.ndarray:
        """
        Log PDF in exponential family form:
        log p(x|θ) = log h(x) + θᵀT(x) - ψ(θ)
        """
        x = np.asarray(x)
        theta = self.natural_parameters()
        T = self.sufficient_statistics(x)
        psi = self.log_partition()
        h = self.base_measure(x)

        return np.log(h) + np.dot(T, theta) - psi
