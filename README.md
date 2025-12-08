# ITP Spiraling Intelligence Simulations

This repository contains minimal Python simulations used in the paper:

**“The Spiraling Intelligence Architecture: Toward a Non-Markovian AI Based on the Infinite Transformation Principle” (Atalebe, 2025)**

## Contents

- `spiral_core_demo.py`  
  Simple 2D spiral demo showing weight evolution and wake/sleep phases.

- `two_task_sim.py`  
  Two-task experiment (Task A then Task B) comparing:
  - a Markov (static) agent, and
  - a Spiral (ITP) agent with rotational drift and sleep.

- `long_run_sim.py`  
  Long-horizon experiment with:
  - Phase A: only Task A,
  - Phase B: only Task B,
  - Phase C: mixed A/B.
  Used to illustrate non-convergent, cyclic behaviour of the Spiral agent.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`

Install with:

```bash
pip install numpy matplotlib
