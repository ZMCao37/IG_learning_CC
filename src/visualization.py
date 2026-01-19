"""
Visualization Utilities for Information Geometry

This module provides functions for visualizing:
- Fisher metric ellipses
- KL divergence contours
- Distribution families
- Geodesics on statistical manifolds
- Parameter space structure
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Optional, Tuple, List, Callable
import warnings


def plot_fisher_ellipse(ax: plt.Axes, mu: float, sigma: float,
                        scale: float = 0.2, color: str = 'blue',
                        linewidth: float = 1.5, label: Optional[str] = None,
                        fill: bool = False, alpha: float = 0.3) -> None:
    """
    Plot Fisher metric ellipse at point (μ, σ) on Gaussian manifold.

    The ellipse represents the "unit ball" in the Fisher metric:
    {v : v^T I(θ) v = const}

    For Gaussian: I = [[1/σ², 0], [0, 2/σ²]]
    Ellipse semi-axes: a = σ (μ direction), b = σ/√2 (σ direction)

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    mu, sigma : float
        Center point (parameters)
    scale : float
        Scale factor for ellipse size
    color : str
        Color for the ellipse
    linewidth : float
        Line width
    label : str, optional
        Label for legend
    fill : bool
        Whether to fill the ellipse
    alpha : float
        Transparency for fill
    """
    theta = np.linspace(0, 2 * np.pi, 100)

    # Semi-axes from Fisher metric (inverse eigenvalues)
    a = sigma * scale  # μ direction
    b = sigma / np.sqrt(2) * scale  # σ direction

    x = a * np.cos(theta) + mu
    y = b * np.sin(theta) + sigma

    if fill:
        ax.fill(x, y, color=color, alpha=alpha)
    ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    ax.plot(mu, sigma, 'o', color=color, markersize=6)


def plot_kl_contours(ax: plt.Axes, mu0: float = 0, sigma0: float = 1,
                     mu_range: Tuple[float, float] = (-2, 2),
                     sigma_range: Tuple[float, float] = (0.3, 2),
                     levels: Optional[List[float]] = None,
                     n_points: int = 100, cmap: str = 'viridis',
                     show_colorbar: bool = True,
                     mode: str = 'forward') -> None:
    """
    Plot KL divergence contours from reference point.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    mu0, sigma0 : float
        Reference point
    mu_range, sigma_range : tuple
        Range for μ and σ axes
    levels : list of float, optional
        Contour levels (default: auto)
    n_points : int
        Number of grid points
    cmap : str
        Colormap
    show_colorbar : bool
        Whether to show colorbar
    mode : str
        'forward': D_KL(p0 || p), 'reverse': D_KL(p || p0)
    """
    mu_vals = np.linspace(mu_range[0], mu_range[1], n_points)
    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], n_points)
    MU, SIGMA = np.meshgrid(mu_vals, sigma_vals)

    if mode == 'forward':
        # D_KL(N(μ0,σ0²) || N(μ,σ²))
        KL = (np.log(SIGMA / sigma0)
              + (sigma0**2 + (mu0 - MU)**2) / (2 * SIGMA**2)
              - 0.5)
    else:
        # D_KL(N(μ,σ²) || N(μ0,σ0²))
        KL = (np.log(sigma0 / SIGMA)
              + (SIGMA**2 + (MU - mu0)**2) / (2 * sigma0**2)
              - 0.5)

    if levels is None:
        levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    cs = ax.contour(MU, SIGMA, KL, levels=levels, cmap=cmap)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

    # Mark reference point
    ax.plot(mu0, sigma0, 'r*', markersize=15, label='Reference')

    if show_colorbar:
        plt.colorbar(cs, ax=ax, label='KL divergence')


def plot_distribution_family(ax: plt.Axes, mu_values: List[float],
                             sigma_values: List[float],
                             x_range: Tuple[float, float] = (-5, 5),
                             n_points: int = 200,
                             colors: Optional[List[str]] = None,
                             labels: Optional[List[str]] = None) -> None:
    """
    Plot multiple Gaussian distributions for comparison.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    mu_values, sigma_values : list of float
        Parameters for each distribution
    x_range : tuple
        Range for x-axis
    n_points : int
        Number of points for smooth curves
    colors : list of str, optional
        Colors for each distribution
    labels : list of str, optional
        Labels for each distribution
    """
    x = np.linspace(x_range[0], x_range[1], n_points)

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(mu_values)))

    if labels is None:
        labels = [f'N({mu:.1f}, {sigma:.1f}²)'
                  for mu, sigma in zip(mu_values, sigma_values)]

    for mu, sigma, color, label in zip(mu_values, sigma_values, colors, labels):
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        ax.plot(x, y, color=color, linewidth=2, label=label)

    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_geodesic(ax: plt.Axes, path: np.ndarray,
                  color: str = 'blue', linewidth: float = 2,
                  marker_start: str = 'o', marker_end: str = 's',
                  label: Optional[str] = None,
                  show_tangent: bool = False) -> None:
    """
    Plot geodesic path on parameter space.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    path : np.ndarray of shape (n_steps, 2)
        Points along geodesic
    color : str
        Line color
    linewidth : float
        Line width
    marker_start, marker_end : str
        Markers for start and end points
    label : str, optional
        Label for legend
    show_tangent : bool
        Whether to show tangent vectors
    """
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth, label=label)
    ax.plot(path[0, 0], path[0, 1], marker_start, color=color, markersize=12)
    ax.plot(path[-1, 0], path[-1, 1], marker_end, color=color, markersize=12)

    if show_tangent and len(path) > 1:
        # Show tangent at start
        tangent = path[1] - path[0]
        tangent = tangent / np.linalg.norm(tangent) * 0.3  # Scale for visibility
        ax.annotate('', xy=path[0] + tangent, xytext=path[0],
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


def plot_parameter_space_grid(ax: plt.Axes,
                              mu_range: Tuple[float, float] = (-2, 2),
                              sigma_range: Tuple[float, float] = (0.3, 2),
                              n_mu: int = 10, n_sigma: int = 10,
                              show_fisher_ellipses: bool = True,
                              ellipse_scale: float = 0.1) -> None:
    """
    Plot parameter space with grid and optional Fisher ellipses.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    mu_range, sigma_range : tuple
        Range for μ and σ axes
    n_mu, n_sigma : int
        Number of grid points in each direction
    show_fisher_ellipses : bool
        Whether to show Fisher metric ellipses
    ellipse_scale : float
        Scale for ellipses
    """
    mu_vals = np.linspace(mu_range[0], mu_range[1], n_mu)
    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], n_sigma)

    # Draw grid
    for mu in mu_vals:
        ax.axvline(x=mu, color='lightgray', linestyle='-', alpha=0.5)
    for sigma in sigma_vals:
        ax.axhline(y=sigma, color='lightgray', linestyle='-', alpha=0.5)

    # Draw Fisher ellipses at grid points
    if show_fisher_ellipses:
        for mu in mu_vals[::2]:  # Every other point
            for sigma in sigma_vals[::2]:
                plot_fisher_ellipse(ax, mu, sigma, scale=ellipse_scale,
                                    color='blue', alpha=0.2, fill=True)

    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_xlim(mu_range)
    ax.set_ylim(sigma_range)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)


def plot_natural_gradient_vs_euclidean(ax: plt.Axes,
                                       loss_fn: Callable[[np.ndarray], float],
                                       grad_fn: Callable[[np.ndarray], np.ndarray],
                                       theta0: np.ndarray,
                                       n_steps: int = 20,
                                       lr_euclidean: float = 0.1,
                                       lr_natural: float = 0.1) -> None:
    """
    Compare Euclidean gradient descent with natural gradient descent.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    loss_fn : callable
        Loss function L(θ)
    grad_fn : callable
        Gradient function ∇L(θ)
    theta0 : np.ndarray
        Starting point [μ, σ]
    n_steps : int
        Number of optimization steps
    lr_euclidean, lr_natural : float
        Learning rates
    """
    from .manifolds import GaussianManifold
    manifold = GaussianManifold()

    # Euclidean gradient descent
    path_euclidean = [theta0.copy()]
    theta = theta0.copy()
    for _ in range(n_steps):
        grad = grad_fn(theta)
        theta = theta - lr_euclidean * grad
        theta[1] = max(theta[1], 0.1)  # Keep σ positive
        path_euclidean.append(theta.copy())
    path_euclidean = np.array(path_euclidean)

    # Natural gradient descent
    path_natural = [theta0.copy()]
    theta = theta0.copy()
    for _ in range(n_steps):
        grad = grad_fn(theta)
        I_inv = manifold.fisher_metric_inverse(theta)
        natural_grad = I_inv @ grad
        theta = theta - lr_natural * natural_grad
        theta[1] = max(theta[1], 0.1)  # Keep σ positive
        path_natural.append(theta.copy())
    path_natural = np.array(path_natural)

    # Plot paths
    ax.plot(path_euclidean[:, 0], path_euclidean[:, 1], 'b-o',
            linewidth=2, markersize=4, label='Euclidean GD')
    ax.plot(path_natural[:, 0], path_natural[:, 1], 'r-s',
            linewidth=2, markersize=4, label='Natural GD')
    ax.plot(theta0[0], theta0[1], 'k*', markersize=15, label='Start')

    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_info_geometry_figure(figsize: Tuple[float, float] = (14, 5),
                                n_cols: int = 2) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a figure with consistent styling for information geometry plots.

    Parameters:
    -----------
    figsize : tuple
        Figure size (width, height)
    n_cols : int
        Number of subplot columns

    Returns:
    --------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, list(axes)


def animate_geodesic(path: np.ndarray, save_path: Optional[str] = None,
                     fps: int = 10) -> None:
    """
    Create animation of movement along geodesic.

    Parameters:
    -----------
    path : np.ndarray of shape (n_steps, 2)
        Points along geodesic
    save_path : str, optional
        Path to save animation (requires ffmpeg)
    fps : int
        Frames per second
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        warnings.warn("Animation requires matplotlib.animation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Parameter space
    ax1 = axes[0]
    ax1.set_xlim(path[:, 0].min() - 0.5, path[:, 0].max() + 0.5)
    ax1.set_ylim(0, path[:, 1].max() + 0.5)
    ax1.set_xlabel('μ')
    ax1.set_ylabel('σ')
    ax1.set_title('Geodesic in parameter space')
    ax1.grid(True, alpha=0.3)

    line, = ax1.plot([], [], 'b-', linewidth=2)
    point, = ax1.plot([], [], 'ro', markersize=10)

    # Distribution space
    ax2 = axes[1]
    x = np.linspace(-5, 5, 200)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(0, 1.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('p(x)')
    ax2.set_title('Corresponding distribution')
    ax2.grid(True, alpha=0.3)

    dist_line, = ax2.plot([], [], 'b-', linewidth=2)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        dist_line.set_data([], [])
        return line, point, dist_line

    def update(frame):
        # Parameter space
        line.set_data(path[:frame+1, 0], path[:frame+1, 1])
        point.set_data([path[frame, 0]], [path[frame, 1]])

        # Distribution
        mu, sigma = path[frame]
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        dist_line.set_data(x, y)

        return line, point, dist_line

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(path), interval=1000/fps, blit=True)

    if save_path:
        anim.save(save_path, fps=fps)
    else:
        plt.show()

    return anim
