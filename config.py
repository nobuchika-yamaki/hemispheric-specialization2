"""
Configuration parameters for hemispheric specialization simulations.

This module defines a SimulationConfig dataclass used across the codebase.
"""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Grid / environment parameters
    grid_size: int = 20        # 20 x 20 grid
    anomaly_size: int = 3      # 3 x 3 local anomaly patch
    spatial_sigma: float = 1.0 # Gaussian blur sigma for spatial condition

    # Evolutionary parameters
    population_size: int = 20
    n_generations: int = 20
    n_trials_per_generation: int = 8
    selection_fraction: float = 0.5

    # Mutation parameters
    mutation_sd_a: float = 0.05
    mutation_sd_lambda: float = 0.05
    min_a_value: float = 0.1

    # Initial parameter ranges
    init_a_min: float = 1.4
    init_a_max: float = 1.6
    init_lambda_min: float = 0.2
    init_lambda_max: float = 0.4

    # Developmental gains (symmetric in this study)
    m_left: float = 1.0
    m_right: float = 1.0

    # Numerical constant to avoid division by zero
    epsilon: float = 1e-6

    # Number of independent lineages (per condition)
    n_lineages: int = 10
