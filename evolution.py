"""
Evolutionary simulation of hemispheric specialization.

This module implements a simple generational evolutionary algorithm with:
- Truncation selection (top fraction by fitness)
- Arithmetic recombination
- Independent Gaussian mutation of parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import SimulationConfig
from envs import SpatialEnvironment, PlanarEnvironment
from model import HemisphericAgent


@dataclass
class GenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_specialization: float
    best_a_left: float
    best_a_right: float
    best_lambda: float
    best_mean_rt: float


def initialise_population(
    config: SimulationConfig, rng: np.random.Generator
) -> List[HemisphericAgent]:
    """Initialise a population of HemisphericAgent instances."""
    agents: List[HemisphericAgent] = []
    for _ in range(config.population_size):
        a_left = rng.uniform(config.init_a_min, config.init_a_max)
        a_right = rng.uniform(config.init_a_min, config.init_a_max)
        lam = rng.uniform(config.init_lambda_min, config.init_lambda_max)
        agents.append(HemisphericAgent(a_left=a_left, a_right=a_right, lam=lam))
    return agents


def evaluate_population(
    agents: List[HemisphericAgent],
    env,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate all individuals in the population.

    Returns
    -------
    fitnesses : np.ndarray
        Fitness values (negative mean RT) for each agent.
    mean_rts : np.ndarray
        Mean RT for each agent.
    """
    fitnesses = np.zeros(len(agents), dtype=float)
    mean_rts = np.zeros(len(agents), dtype=float)

    for i, agent in enumerate(agents):
        fitness, mean_rt = agent.evaluate_environment(env, config, rng)
        fitnesses[i] = fitness
        mean_rts[i] = mean_rt

    return fitnesses, mean_rts


def select_parents(
    agents: List[HemisphericAgent],
    fitnesses: np.ndarray,
    config: SimulationConfig,
) -> List[HemisphericAgent]:
    """Select the top fraction of the population as parents."""
    n = len(agents)
    n_parents = max(2, int(n * config.selection_fraction))
    idx_sorted = np.argsort(fitnesses)[::-1]  # descending
    parent_indices = idx_sorted[:n_parents]
    parents = [agents[int(i)] for i in parent_indices]
    return parents


def reproduce(
    parents: List[HemisphericAgent],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> List[HemisphericAgent]:
    """
    Create a new population from parents via arithmetic recombination
    and Gaussian mutation.
    """
    new_agents: List[HemisphericAgent] = []
    n = config.population_size
    n_parents = len(parents)

    while len(new_agents) < n:
        # Sample two parents with replacement
        i, j = rng.integers(0, n_parents, size=2)
        p1 = parents[int(i)]
        p2 = parents[int(j)]

        # Arithmetic recombination
        alpha = rng.uniform(0.0, 1.0)
        child_a_left = alpha * p1.a_left + (1.0 - alpha) * p2.a_left
        child_a_right = alpha * p1.a_right + (1.0 - alpha) * p2.a_right
        child_lambda = alpha * p1.lam + (1.0 - alpha) * p2.lam

        # Mutation
        child_a_left += rng.normal(0.0, config.mutation_sd_a)
        child_a_right += rng.normal(0.0, config.mutation_sd_a)
        child_lambda += rng.normal(0.0, config.mutation_sd_lambda)

        # Enforce parameter constraints
        child_a_left = max(config.min_a_value, child_a_left)
        child_a_right = max(config.min_a_value, child_a_right)

        new_agents.append(
            HemisphericAgent(
                a_left=child_a_left,
                a_right=child_a_right,
                lam=child_lambda,
            )
        )

    return new_agents


def run_lineage(
    config: SimulationConfig,
    env_type: str,
    lineage_seed: int,
) -> Dict[str, object]:
    """
    Run a single evolutionary lineage for a given environment type.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration parameters.
    env_type : {"spatial", "planar"}
        Type of environment to use.
    lineage_seed : int
        Random seed for this lineage.

    Returns
    -------
    results : dict
        Dictionary containing:
        - "summaries": list of GenerationSummary
        - "final_best_agent": HemisphericAgent
        - "env_type": str
        - "lineage_seed": int
    """
    rng = np.random.default_rng(lineage_seed)

    if env_type == "spatial":
        env = SpatialEnvironment(config)
    elif env_type == "planar":
        env = PlanarEnvironment(config)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    # Initialise population
    population = initialise_population(config, rng)

    summaries: List[GenerationSummary] = []

    for gen in range(config.n_generations):
        # For reproducibility, use a fresh RNG for evaluation but derived from the same lineage seed
        eval_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        fitnesses, mean_rts = evaluate_population(population, env, config, eval_rng)

        # Compute statistics
        best_idx = int(np.argmax(fitnesses))
        best_agent = population[best_idx]
        best_fitness = float(fitnesses[best_idx])
        best_mean_rt = float(mean_rts[best_idx])
        mean_fitness = float(fitnesses.mean())

        summary = GenerationSummary(
            generation=gen,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            best_specialization=best_agent.specialization,
            best_a_left=best_agent.a_left,
            best_a_right=best_agent.a_right,
            best_lambda=best_agent.lam,
            best_mean_rt=best_mean_rt,
        )
        summaries.append(summary)

        # Reproduce next generation
        parents = select_parents(population, fitnesses, config)
        population = reproduce(parents, config, rng)

    # Evaluate final best agent one more time with a fixed RNG for reporting
    final_rng = np.random.default_rng(lineage_seed + 12345)
    final_fitness, final_rt = best_agent.evaluate_environment(env, config, final_rng)

    results = {
        "summaries": summaries,
        "final_best_agent": best_agent,
        "env_type": env_type,
        "lineage_seed": lineage_seed,
        "final_fitness": float(final_fitness),
        "final_mean_rt": float(final_rt),
    }
    return results


def run_all_lineages(
    config: SimulationConfig,
    base_seed: int = 42,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Run all lineages for both spatial and planar environments.

    Returns
    -------
    spatial_results : list of dict
    planar_results : list of dict
    """
    spatial_results: List[Dict[str, object]] = []
    planar_results: List[Dict[str, object]] = []

    for lineage_idx in range(config.n_lineages):
        # Use the same seed for paired spatial/planar lineages
        seed = base_seed + lineage_idx

        spatial_results.append(run_lineage(config, env_type="spatial", lineage_seed=seed))
        planar_results.append(run_lineage(config, env_type="planar", lineage_seed=seed))

    return spatial_results, planar_results
