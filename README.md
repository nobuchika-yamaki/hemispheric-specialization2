# Environmental Multiscale Structure as a Necessary Condition for the Evolution of Hemispheric Specialization  
### Reproducible Simulation Code

This repository contains the full Python implementation used for the analyses in:

**"Environmental multiscale structure is a necessary condition for the evolution of hemispheric specialization: a minimal computational model"**

The scripts reproduce all evolutionary simulations, statistical analyses, and reported numerical results in the manuscript.  
No proprietary libraries are required; all code runs with standard scientific Python packages.

---

## 1. Overview

This project investigates whether **hemispheric specialization** can emerge from **environmental structure alone**, without any built-in biological asymmetry. Two environments are compared under *identical* evolutionary conditions:

1. **Spatial Environment**  
   - Contains multiscale structure (global gradients + local anomaly).  
   - Expected to induce functional hemispheric differentiation.

2. **Planar Environment**  
   - Structure-free (i.i.d. noise, normalized).  
   - Serves as a control; no specialization should evolve.

Each hemisphere evolves two parameters:

- **a_left**, **a_right** : spatial integration scales  
- **λ** : gating parameter

Evolution proceeds for 20 generations × 20 individuals, repeated across 10 lineages per condition.

This repository enables full reproduction of:

- Specialization magnitude |Δ| = |a_R – a_L|
- Reaction time distributions
- Wilcoxon signed-rank test (|Δ| difference)
- Paired t-test (RT difference)
- All summary statistics reported in the manuscript

---

## 2. Directory Structure

### File descriptions

| File | Purpose |
|------|---------|
| **config.py** | Defines all simulation parameters (population size, mutation SD, grid size, seeds). |
| **envs.py** | Implements the Spatial and Planar environments and a custom NumPy-based Gaussian blur. |
| **model.py** | Defines the hemispheric agent (integration, mismatch, decision variable, reaction time). |
| **evolution.py** | Full evolutionary algorithm (initialization, evaluation, selection, mutation). |
| **stats_analysis.py** | Runs all lineages and performs Wilcoxon / t-tests; saves results. |
| **run_simulations.py** | Main entry point. Executes the full simulation + statistical analysis pipeline. |

---

## 3. Installation

The code requires only **NumPy** and **SciPy**.

### Install dependencies

```bash
pip install numpy scipy


