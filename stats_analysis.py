"""
Statistical analysis for hemispheric specialization simulations.

This script:
- Runs all lineages for spatial and planar environments
- Extracts specialization magnitude |Δ| and mean RT for each lineage
- Performs Wilcoxon signed-rank test on specialization
- Performs paired t-test on reaction times
- Prints summary statistics
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

import numpy as np
from scipy import stats

from config import SimulationConfig
from evolution import run_all_lineages, GenerationSummary


def extract_final_generation_summaries(results: List[dict]) -> List[GenerationSummary]:
    """Extract the last GenerationSummary from each lineage."""
    final_summaries: List[GenerationSummary] = []
    for lineage in results:
        summaries = lineage["summaries"]
        final_summaries.append(summaries[-1])
    return final_summaries


def main():
    config = SimulationConfig()

    print("Running evolutionary simulations for spatial and planar environments...")
    spatial_results, planar_results = run_all_lineages(config, base_seed=42)

    # Extract final generation summaries
    spatial_final = extract_final_generation_summaries(spatial_results)
    planar_final = extract_final_generation_summaries(planar_results)

    # Collect specialization |Δ| and reaction times
    spatial_delta = np.array([s.best_specialization for s in spatial_final], dtype=float)
    planar_delta = np.array([s.best_specialization for s in planar_final], dtype=float)

    spatial_rt = np.array([s.best_mean_rt for s in spatial_final], dtype=float)
    planar_rt = np.array([s.best_mean_rt for s in planar_final], dtype=float)

    # Summary statistics
    print("\n=== Specialization magnitude |Δ| ===")
    print(f"Spatial condition: median = {np.median(spatial_delta):.3f}, IQR = {np.percentile(spatial_delta, 25):.3f}–{np.percentile(spatial_delta, 75):.3f}")
    print(f"Planar condition : median = {np.median(planar_delta):.3f}, IQR = {np.percentile(planar_delta, 25):.3f}–{np.percentile(planar_delta, 75):.3f}")

    # Wilcoxon signed-rank test
    wilcoxon_res = stats.wilcoxon(spatial_delta, planar_delta, alternative="greater")
    print(f"\nWilcoxon signed-rank test on |Δ| (spatial > planar):")
    print(f"  statistic = {wilcoxon_res.statistic:.3f}, p-value = {wilcoxon_res.pvalue:.6f}")

    # Reaction times
    print("\n=== Best-individual reaction times (lower is better) ===")
    print(f"Spatial condition: mean = {np.mean(spatial_rt):.4f}, SD = {np.std(spatial_rt, ddof=1):.4f}")
    print(f"Planar condition : mean = {np.mean(planar_rt):.4f}, SD = {np.std(planar_rt, ddof=1):.4f}")

    # Paired t-test
    ttest_res = stats.ttest_rel(spatial_rt, planar_rt, alternative="less")
    print(f"\nPaired t-test on reaction times (spatial < planar):")
    print(f"  t = {ttest_res.statistic:.3f}, p-value = {ttest_res.pvalue:.6f}")

    # Optional: save results to disk for reproducibility
    np.savez(
        "simulation_results.npz",
        config=asdict(config),
        spatial_delta=spatial_delta,
        planar_delta=planar_delta,
        spatial_rt=spatial_rt,
        planar_rt=planar_rt,
    )
    print('\nSaved summary arrays to "simulation_results.npz".')


if __name__ == "__main__":
    main()
