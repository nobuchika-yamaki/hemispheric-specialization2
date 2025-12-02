"""
Hemispheric processing model and fitness evaluation.

This module defines the HemisphericAgent class, which implements:
- Gaussian spatial integration in each hemisphere
- Mismatch computation
- Decision variable and reaction time
- Fitness as the negative mean reaction time
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from config import SimulationConfig
from envs import _gaussian_blur_2d


@dataclass
class HemisphericAgent:
    """
    Minimal two-hemisphere agent.

    Parameters
    ----------
    a_left : float
        Spatial integration width (Gaussian sigma) for the left hemisphere.
    a_right : float
        Spatial integration width (Gaussian sigma) for the right hemisphere.
    lam : float
        Gating parameter lambda in the decision variable.
    """
    a_left: float
    a_right: float
    lam: float

    def compute_mismatch(
        self, grid: NDArray[np.float64], a: float
    ) -> float:
        """
        Compute absolute-difference mismatch between original and filtered grid.

        mismatch = sum |original - filtered|
        """
        filtered = _gaussian_blur_2d(grid, sigma=a)
        mismatch = np.sum(np.abs(grid - filtered))
        return float(mismatch)

    def evaluate_trial(
        self,
        grid: NDArray[np.float64],
        config: SimulationConfig,
    ) -> Tuple[float, float]:
        """
        Evaluate the agent on a single trial.

        Returns
        -------
        reaction_time : float
            Reaction time for this trial.
        decision_variable : float
            Raw decision variable D.
        """
        # Apply developmental gains (symmetric here but kept for completeness)
        a_L = self.a_left * config.m_left
        a_R = self.a_right * config.m_right

        mismatch_L = self.compute_mismatch(grid, a_L)
        mismatch_R = self.compute_mismatch(grid, a_R)

        # Decision variable
        D = (mismatch_R - mismatch_L) + self.lam * (
            abs(mismatch_R) - abs(mismatch_L)
        )

        # Reaction time: inverse of absolute decision variable
        reaction_time = 1.0 / (abs(D) + config.epsilon)

        return float(reaction_time), float(D)

    def evaluate_environment(
        self,
        env,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        """
        Evaluate the agent over multiple trials in a given environment.

        Returns
        -------
        mean_fitness : float
            Negative mean reaction time across trials.
        mean_rt : float
            Mean reaction time across trials.
        """
        rts = []

        for _ in range(config.n_trials_per_generation):
            grid = env.generate_trial(rng)
            rt, _ = self.evaluate_trial(grid, config)
            rts.append(rt)

        rts = np.asarray(rts, dtype=float)
        mean_rt = float(rts.mean())
        fitness = -mean_rt

        return fitness, mean_rt

    @property
    def specialization(self) -> float:
        """Magnitude of hemispheric specialization |Δ| = |a_right - a_left|."""
        return float(abs(self.a_right - self.a_left))
