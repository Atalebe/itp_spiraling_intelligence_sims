#!/usr/bin/env python3
"""
cost_benchmark_sim_v2.py

Constrained compute-cost benchmark for Markov vs Spiral (ITP/SIA-style).

Key idea:
- Cost claims must be conditioned on *maintaining retention*.
- We measure:
  (1) recovery steps to get back above A_target after Task B shock,
  (2) total compute cost over the run,
  (3) min alignment with Task A during mixed phase (retention floor),
  (4) failure rate (violates retention floor).

Compute model (simple, explicit):
- wake update cost = 1.0
- sleep step cost = sleep_cost (default 0.25)
- Markov "retrain A" event cost = retrain_cost_per_step * retrain_steps
  (models a heavy replay / fine-tune episode)

This is not "GPU dollars". It is a clean, reproducible *compute unit* proxy.
"""

import time
import numpy as np
import csv
from dataclasses import dataclass


# -----------------------------
# Helpers
# -----------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        return v
    return v / n

def cosine_alignment(w: np.ndarray, v: np.ndarray) -> float:
    wn = np.linalg.norm(w)
    vn = np.linalg.norm(v)
    if wn == 0 or vn == 0:
        return 0.0
    return float(np.dot(w, v) / (wn * vn))


# -----------------------------
# Agents
# -----------------------------

class MarkovAgent:
    """
    Simple Markovian online learner (perceptron-ish).
    """
    def __init__(self, dim: int, lr: float = 0.15, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = normalize(rng.standard_normal(dim))
        self.lr = lr

    def update(self, x: np.ndarray, target_sign: float = 1.0):
        y_raw = float(np.dot(self.w, x))
        y = 1.0 if y_raw > 0 else -1.0
        error = target_sign - y
        self.w = normalize(self.w + self.lr * error * x)


class SpiralAgent:
    """
    Spiral agent with:
      - internal rotation (drift),
      - consolidation C(t) reducing plasticity,
      - sleep with pruning + REPLAY of stored prototypes (cheap rehearsal).
    """
    def __init__(
        self,
        dim: int,
        base_lr: float = 0.10,
        base_omega: float = 0.03,
        kappa: float = 0.004,
        replay_strength: float = 0.06,
        seed: int = 0
    ):
        rng = np.random.default_rng(seed)
        self.w = normalize(rng.standard_normal(dim))
        self.base_lr = base_lr
        self.base_omega = base_omega
        self.kappa = kappa
        self.replay_strength = replay_strength
        self.C = 0.0
        self.dim = dim

        # "memory" prototypes for replay
        self.proto_A = None
        self.proto_B = None

    def _omega_eff(self) -> float:
        # more consolidation -> more stable drift (or interpret as matured cycle)
        return self.base_omega * (0.2 + 0.8 * self.C)

    def _rotate(self):
        # 2D-only rotation; for dim>2, rotate first 2 coords.
        theta = self._omega_eff()
        c, s = np.cos(theta), np.sin(theta)

        if self.dim == 2:
            R = np.array([[c, -s],
                          [s,  c]], dtype=float)
            self.w = R @ self.w
            return

        # dim>2: rotate a chosen plane (0,1), leave others unchanged
        w = self.w.copy()
        w0, w1 = w[0], w[1]
        w[0] = c * w0 - s * w1
        w[1] = s * w0 + c * w1
        self.w = w

    def update(self, x: np.ndarray, target_sign: float = 1.0, label: str = ""):
        # internal drift
        self._rotate()

        # plastic update with consolidation
        y_raw = float(np.dot(self.w, x))
        y = 1.0 if y_raw > 0 else -1.0
        error = target_sign - y

        plasticity = (1.0 - self.C)
        lr_eff = self.base_lr * (0.6 + 0.4 * plasticity)

        self.w = normalize(self.w + lr_eff * plasticity * error * x)

        # consolidation grows when error is small (stability ratchet)
        self.C += self.kappa * max(0.0, 1.0 - abs(error))
        self.C = min(self.C, 1.0)

        # update prototypes
        if label == "A":
            self.proto_A = x.copy()
        elif label == "B":
            self.proto_B = x.copy()

    def sleep_step(self):
        # gentle drift
        self._rotate()

        # pruning tiny components
        thresh = 1e-3
        self.w[np.abs(self.w) < thresh] = 0.0
        self.w = normalize(self.w)

        # cheap replay: nudge toward stored prototypes (if available)
        # this is the key difference vs your earlier cost run.
        for proto in (self.proto_A, self.proto_B):
            if proto is None:
                continue
            proto = normalize(proto)
            y = float(np.dot(self.w, proto))
            # Oja-like stabilizing nudge, scaled down
            dw = self.replay_strength * (proto - y * self.w)
            self.w = normalize(self.w + dw)

        # tiny consolidation during sleep
        self.C = min(1.0, self.C + 0.25 * self.kappa)


# -----------------------------
# Benchmark definition
# -----------------------------

@dataclass
class Params:
    dim: int = 2
    # Phases
    T_A: int = 200
    T_B: int = 200
    T_mix: int = 1200

    # Targets
    A_target: float = 0.80     # recovery threshold
    A_floor: float = 0.00      # retention constraint during mixed phase

    # Costs (compute units)
    wake_cost: float = 1.0
    sleep_cost: float = 0.25

    # Markov retrain model
    retrain_trigger: float = 0.35   # if align_A falls below this, do a retrain burst
    retrain_steps: int = 80
    retrain_cost_per_step: float = 3.0  # retrain is "heavier" than online updates


def run_one(seed: int, K_sleep: int, P: Params):
    """
    K_sleep: sleep cadence. If K_sleep=0 => no sleep.
    """

    rng = np.random.default_rng(seed)

    # Define tasks
    # Keep them orthogonal for clarity
    A = normalize(np.array([1.0, 0.0], dtype=float))
    B = normalize(np.array([0.0, 1.0], dtype=float))

    # Agents
    M = MarkovAgent(dim=P.dim, lr=0.18, seed=seed)
    S = SpiralAgent(dim=P.dim, base_lr=0.11, base_omega=0.03, kappa=0.004,
                    replay_strength=0.06, seed=seed)

    cost_M = 0.0
    cost_S = 0.0

    # Phase A learn
    for t in range(P.T_A):
        M.update(A, 1.0)
        cost_M += P.wake_cost

        S.update(A, 1.0, label="A")
        cost_S += P.wake_cost

        if K_sleep > 0 and (t + 1) % K_sleep == 0:
            S.sleep_step()
            cost_S += P.sleep_cost

    # Phase B learn (shock to A)
    # Recovery time measured from start of B.
    M_recovery = None
    S_recovery = None

    for t in range(P.T_B):
        M.update(B, 1.0)
        cost_M += P.wake_cost

        S.update(B, 1.0, label="B")
        cost_S += P.wake_cost

        if K_sleep > 0 and (P.T_A + t + 1) % K_sleep == 0:
            S.sleep_step()
            cost_S += P.sleep_cost

        # track first time alignment back above A_target during B
        if M_recovery is None and cosine_alignment(M.w, A) >= P.A_target:
            M_recovery = (t + 1)

        if S_recovery is None and cosine_alignment(S.w, A) >= P.A_target:
            S_recovery = (t + 1)

        # Markov retrain model: if it forgets too much, it pays a heavy retrain burst
        if cosine_alignment(M.w, A) < P.retrain_trigger:
            for _ in range(P.retrain_steps):
                M.update(A, 1.0)
            cost_M += P.retrain_steps * P.retrain_cost_per_step

    # Mixed future phase
    minA_M =  1e9
    minA_S =  1e9
    failed_M = False
    failed_S = False

    for t in range(P.T_mix):
        x = A if (rng.random() < 0.5) else B

        # Markov
        M.update(x, 1.0)
        cost_M += P.wake_cost

        # Spiral
        S.update(x, 1.0, label=("A" if x is A else "B"))
        cost_S += P.wake_cost

        if K_sleep > 0 and (P.T_A + P.T_B + t + 1) % K_sleep == 0:
            S.sleep_step()
            cost_S += P.sleep_cost

        aM = cosine_alignment(M.w, A)
        aS = cosine_alignment(S.w, A)
        minA_M = min(minA_M, aM)
        minA_S = min(minA_S, aS)

        if aM < P.A_floor:
            failed_M = True
        if aS < P.A_floor:
            failed_S = True

    # If never recovered during B, mark as T_B+ (failed recovery)
    if M_recovery is None:
        M_recovery = P.T_B + 1
    if S_recovery is None:
        S_recovery = P.T_B + 1

    return {
        "seed": seed,
        "K": K_sleep,
        "M_recovery_steps": float(M_recovery),
        "S_recovery_steps": float(S_recovery),
        "M_total_cost": float(cost_M),
        "S_total_cost": float(cost_S),
        "minAlignA_M": float(minA_M),
        "minAlignA_S": float(minA_S),
        "failed_M": int(failed_M),
        "failed_S": int(failed_S),
    }


def median_or_nan(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def main():
    t0 = time.perf_counter()

    P = Params()

    # Sleep cadences to test (K=0 means no sleep)
    K_list = [0, 10, 20, 30, 40, 60]

    seeds = list(range(25))  # increase if you want tighter intervals

    rows = []
    for K in K_list:
        for seed in seeds:
            rows.append(run_one(seed=seed, K_sleep=K, P=P))

    # Write raw rows
    out_csv = "cost_benchmark_results_v2.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summaries
    print("\n=== CONSTRAINED COST BENCHMARK SUMMARY (medians over seeds) ===")
    print(f"A_target={P.A_target:.2f}  A_floor={P.A_floor:.2f}  sleep_cost={P.sleep_cost:.2f}  retrain_cost/step={P.retrain_cost_per_step:.2f}")
    print("K | M_rec | S_rec | M_total | S_total | fail_M | fail_S | minA_M | minA_S")
    print("-" * 92)

    for K in K_list:
        sub = [r for r in rows if r["K"] == K]
        M_rec = median_or_nan([r["M_recovery_steps"] for r in sub])
        S_rec = median_or_nan([r["S_recovery_steps"] for r in sub])
        M_tot = median_or_nan([r["M_total_cost"] for r in sub])
        S_tot = median_or_nan([r["S_total_cost"] for r in sub])
        fail_M = sum(r["failed_M"] for r in sub)
        fail_S = sum(r["failed_S"] for r in sub)
        minA_M = median_or_nan([r["minAlignA_M"] for r in sub])
        minA_S = median_or_nan([r["minAlignA_S"] for r in sub])

        print(f"{K:2d} | {M_rec:5.1f} | {S_rec:5.1f} | {M_tot:7.1f} | {S_tot:7.1f} |"
              f" {fail_M:6d} | {fail_S:6d} | {minA_M:6.3f} | {minA_S:6.3f}")

    t1 = time.perf_counter()
    print(f"\nWrote: {out_csv}")
    print(f"Wall time: {t1 - t0:.3f} s")


if __name__ == "__main__":
    main()
