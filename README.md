# ITP Spiraling Intelligence Simulations

This repository contains reference simulations for:

**Atalebe, S.**  
*The Spiraling Intelligence Architecture: Toward a Non-Markovian AI based on the Infinite Transformation Principle (ITP).*

The code implements:

- **Markov Agent** – a conventional Hebbian learner with convergence.
- **Spiral Agent** – a non-Markovian learner with:
  - Rotational Hebbian updates (complex-phase drift),
  - Autopoietic sleep cycles (pruning + replay),
  - Stress-driven modulation of learning rate and rotation speed.

## Files

- `spiral_vs_markov_demo.py`  
  Simple 2D demonstration of weight trajectories on the unit circle.

- `two_task_sim.py`  
  Two-task continual learning scenario:
  - Task A and Task B with shifting patterns.
  - Compares catastrophic forgetting in Markov vs Spiral agents.

- `long_run_sim.py`  
  Long-horizon run showing:
  - Spiraling representational drift,
  - Sleep-driven consolidation,
  - Phase-based oscillations in alignment.

Generated figures (if committed):

- `fig_spiral_vs_markov.png` – weight trajectories.
- `fig_two_task_alignment.png` – alignment across tasks.
- `fig_long_run_alignment.png` – long-term behaviour.

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
