"""
Information Geometry Learning - Reusable Code Module

This module provides reusable classes and functions for studying
information geometry concepts.

Modules:
- distributions: Probability distribution classes with Fisher information
- manifolds: Statistical manifold classes
- visualization: Visualization utilities for information geometry
"""

from .distributions import (
    GaussianDistribution,
    BernoulliDistribution,
    PoissonDistribution,
    ExponentialFamily,
)
from .manifolds import (
    StatisticalManifold,
    GaussianManifold,
)
from .visualization import (
    plot_fisher_ellipse,
    plot_kl_contours,
    plot_distribution_family,
    plot_geodesic,
)

__version__ = "0.1.0"
__all__ = [
    # Distributions
    "GaussianDistribution",
    "BernoulliDistribution",
    "PoissonDistribution",
    "ExponentialFamily",
    # Manifolds
    "StatisticalManifold",
    "GaussianManifold",
    # Visualization
    "plot_fisher_ellipse",
    "plot_kl_contours",
    "plot_distribution_family",
    "plot_geodesic",
]
