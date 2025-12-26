import numpy as np
import csv
import time
from dataclasses import dataclass, asdict


# ============================================================
# Helpers
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v.copy()
    return v / n


def cosine_alignment(w: np.ndarray, v: np.ndarray) -> float:
    wn = np.linalg.norm(w)
    vn = np.linalg.norm(v)
    if wn == 0 or vn == 0:
        return 0.0
    return float(np.dot(w, v) / (wn * vn))


# ============================================================
# Agents
# ============================================================

class MarkovAgent:
    """
    Baseline continual learner.
    Cost proxy: every update counts as 1 wake-op.
    """
    def __init__(self, dim: int, lr: float = 0.2, seed: int = 0):
        rng = np.random.default_rng(seed)
        w0 = rng.standard_normal(dim)
        self.w = normalize(w0)
        self.lr = float(lr)

        # cost counters
        self.wake_ops = 0
        self.recovery_ops = 0

    def update(self, x: np.ndarray, target_sign: float):
        y_raw = np.dot(self.w, x)
        y = 1.0 if y_raw > 0.0 else -1.0
        error = target_sign - y
        self.w = self.w + self.lr * error * x
        self.w = normalize(self.w)

        self.wake_ops += 1

    def recover_until(self, x: np.ndarray, target_sign: float, pattern: np.ndarray,
                      align_threshold: float, max_steps: int):
        """
        Recovery burst: count extra ops required to restore alignment >= threshold.
        """
        steps = 0
        while steps < max_steps:
            if cosine_alignment(self.w, pattern) >= align_threshold:
                break
            self.update(x, target_sign)
            steps += 1
        self.recovery_ops += steps
        return steps


class SpiralAgent:
    """
    Spiral (ITP-style) agent with consolidation + periodic sleep pruning.
    We count wake_ops and sleep_ops separately.
    """
    def __init__(self, dim: int, base_lr: float = 0.08, base_omega: float = 0.04, kappa: float = 0.005, seed: int = 0):
        rng = np.random.default_rng(seed)
        w0 = rng.standard_normal(dim)
        self.w = normalize(w0)

        self.base_lr = float(base_lr)
        self.base_omega = float(base_omega)
        self.kappa = float(kappa)
        self.C = 0.0

        self.wake_ops = 0
        self.sleep_ops = 0
        self.recovery_ops = 0

    def _effective_omega(self) -> float:
        # ramps with consolidation
        return self.base_omega * (0.2 + 0.8 * self.C)

    def _rotate(self):
        theta = self._effective_omega()
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s],
                      [s,  c]], dtype=float)
        self.w = R @ self.w

    def update(self, x: np.ndarray, target_sign: float):
        self._rotate()

        y_raw = np.dot(self.w, x)
        y = 1.0 if y_raw > 0.0 else -1.0
        error = target_sign - y

        plasticity = 1.0 - self.C
        lr_eff = self.base_lr * (0.5 + 0.5 * (1.0 - self.C))

        self.w = self.w + lr_eff * plasticity * error * x
        self.w = normalize(self.w)

        # consolidate when error small
        self.C += self.kappa * max(0.0, 1.0 - abs(error))
        self.C = min(self.C, 1.0)

        self.wake_ops += 1

    def sleep_step(self):
        self._rotate()

        # prune tiny components
        thresh = 1e-3
        self.w[np.abs(self.w) < thresh] = 0.0
        self.w = normalize(self.w)

        self.sleep_ops += 1

    def recover_until(self, x: np.ndarray, target_sign: float, pattern: np.ndarray,
                      align_threshold: float, max_steps: int, sleep_every: int):
        """
        Recovery burst: wake updates until alignment >= threshold.
        Spiral can still do its periodic sleep while recovering.
        """
        steps = 0
        while steps < max_steps:
            if cosine_alignment(self.w, pattern) >= align_threshold:
                break
            self.update(x, target_sign)
            steps += 1
            if sleep_every > 0 and (steps % sleep_every == 0):
                self.sleep_step()
        self.recovery_ops += steps
        return steps


# ============================================================
# Experiment definition
# ============================================================

@dataclass
class RunResult:
    seed: int
    sleep_period_K: int
    horizon_T: int
    align_threshold: float

    # measured outcomes
    markov_final_align_A: float
    spiral_final_align_A: float
    markov_min_align_A: float
    spiral_min_align_A: float

    markov_wake_ops: int
    spiral_wake_ops: int
    spiral_sleep_ops: int

    markov_recovery_ops: int
    spiral_recovery_ops: int

    # total "cost" proxies (you can pick weighting in paper)
    # Here: total_ops_1to1 = wake + sleep
    markov_total_ops: int
    spiral_total_ops_1to1: int


def run_cost_benchmark(
    seed: int = 0,
    sleep_period_K: int = 30,
    dim: int = 2,
    lr_markov: float = 0.2,
    base_lr_spiral: float = 0.08,
    base_omega: float = 0.04,
    kappa: float = 0.005,
    horizon_T: int = 1200,
    align_threshold: float = 0.85,
    eval_every: int = 10,
    recovery_max_steps: int = 60,
):
    rng = np.random.default_rng(seed)

    # Two orthogonal tasks (you can make them non-orthogonal if you want)
    pattern_A = normalize(np.array([1.0, 0.0], dtype=float))
    pattern_B = normalize(np.array([0.0, 1.0], dtype=float))

    # Inputs used during learning (noisy around task pattern)
    def sample_x(pattern: np.ndarray, noise: float = 0.05):
        return normalize(pattern + rng.normal(0.0, noise, size=dim))

    M = MarkovAgent(dim=dim, lr=lr_markov, seed=seed + 11)
    S = SpiralAgent(dim=dim, base_lr=base_lr_spiral, base_omega=base_omega, kappa=kappa, seed=seed + 29)

    # Task schedule: A then B then mixed, but with shocks
    T_A = 200
    T_B = 200
    T_mix = max(0, horizon_T - (T_A + T_B))

    alignA_M = []
    alignA_S = []

    def step_agents(current_task: str):
        if current_task == "A":
            x = sample_x(pattern_A)
            M.update(x, +1.0)
            S.update(x, +1.0)
        elif current_task == "B":
            x = sample_x(pattern_B)
            M.update(x, +1.0)
            S.update(x, +1.0)
        else:
            # mixed
            if rng.random() < 0.5:
                x = sample_x(pattern_A)
                M.update(x, +1.0)
                S.update(x, +1.0)
            else:
                x = sample_x(pattern_B)
                M.update(x, +1.0)
                S.update(x, +1.0)

    # Periodic sleep during the whole run (for Spiral)
    def maybe_sleep(t: int):
        if sleep_period_K > 0 and ((t + 1) % sleep_period_K == 0):
            S.sleep_step()

    # Event-triggered recovery rule:
    # If alignment with A drops below threshold at evaluation time,
    # do recovery bursts on A for each agent until restored (or cap).
    def maybe_recover(t: int):
        if ((t + 1) % eval_every) != 0:
            return
        aM = cosine_alignment(M.w, pattern_A)
        aS = cosine_alignment(S.w, pattern_A)

        if aM < align_threshold:
            xA = sample_x(pattern_A, noise=0.02)
            M.recover_until(xA, +1.0, pattern_A, align_threshold, recovery_max_steps)

        if aS < align_threshold:
            xA = sample_x(pattern_A, noise=0.02)
            # during recovery, spiral still sleeps occasionally inside the loop
            S.recover_until(xA, +1.0, pattern_A, align_threshold, recovery_max_steps, sleep_every=max(1, sleep_period_K))

    # Run phases
    t = 0
    for _ in range(T_A):
        step_agents("A")
        maybe_sleep(t)
        maybe_recover(t)
        alignA_M.append(cosine_alignment(M.w, pattern_A))
        alignA_S.append(cosine_alignment(S.w, pattern_A))
        t += 1

    for _ in range(T_B):
        step_agents("B")
        maybe_sleep(t)
        maybe_recover(t)
        alignA_M.append(cosine_alignment(M.w, pattern_A))
        alignA_S.append(cosine_alignment(S.w, pattern_A))
        t += 1

    # Mixed with occasional shocks (distribution shift)
    for _ in range(T_mix):
        step_agents("mix")

        # shock: occasionally increase noise, simulating environment drift
        if rng.random() < 0.02:
            # do a few high-noise mixed steps
            for __ in range(5):
                # alternate randomly
                patt = pattern_A if rng.random() < 0.5 else pattern_B
                x = sample_x(patt, noise=0.25)
                M.update(x, +1.0)
                S.update(x, +1.0)

        maybe_sleep(t)
        maybe_recover(t)
        alignA_M.append(cosine_alignment(M.w, pattern_A))
        alignA_S.append(cosine_alignment(S.w, pattern_A))
        t += 1

    alignA_M = np.array(alignA_M, dtype=float)
    alignA_S = np.array(alignA_S, dtype=float)

    res = RunResult(
        seed=seed,
        sleep_period_K=int(sleep_period_K),
        horizon_T=int(horizon_T),
        align_threshold=float(align_threshold),

        markov_final_align_A=float(alignA_M[-1]),
        spiral_final_align_A=float(alignA_S[-1]),
        markov_min_align_A=float(np.min(alignA_M)),
        spiral_min_align_A=float(np.min(alignA_S)),

        markov_wake_ops=int(M.wake_ops),
        spiral_wake_ops=int(S.wake_ops),
        spiral_sleep_ops=int(S.sleep_ops),

        markov_recovery_ops=int(M.recovery_ops),
        spiral_recovery_ops=int(S.recovery_ops),

        markov_total_ops=int(M.wake_ops),
        spiral_total_ops_1to1=int(S.wake_ops + S.sleep_ops),
    )

    return res


# ============================================================
# Main: sweep K and seeds, write CSV
# ============================================================

def main():
    t0 = time.perf_counter()

    # Sweep sleep cadence K (every K steps)
    K_list = [0, 10, 20, 30, 40, 60]
    seeds = list(range(0, 30))

    out_csv = "cost_benchmark_results.csv"

    all_rows = []
    for K in K_list:
        for seed in seeds:
            r = run_cost_benchmark(
                seed=seed,
                sleep_period_K=K,
                horizon_T=1200,
                align_threshold=0.85,
                eval_every=10,
                recovery_max_steps=60,
            )
            all_rows.append(r)

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for r in all_rows:
            writer.writerow(asdict(r))

    # Print a compact summary (medians per K)
    print("\n=== COST BENCHMARK SUMMARY (medians over seeds) ===")
    print("K | M_recovery | S_recovery | M_total | S_total(1:1) | minAlignA(M) | minAlignA(S)")
    print("-" * 86)

    for K in K_list:
        rows = [r for r in all_rows if r.sleep_period_K == K]
        med = lambda xs: float(np.median(np.array(xs, dtype=float)))

        m_rec = med([r.markov_recovery_ops for r in rows])
        s_rec = med([r.spiral_recovery_ops for r in rows])
        m_tot = med([r.markov_total_ops for r in rows])
        s_tot = med([r.spiral_total_ops_1to1 for r in rows])
        m_min = med([r.markov_min_align_A for r in rows])
        s_min = med([r.spiral_min_align_A for r in rows])

        print(f"{K:2d} | {m_rec:9.1f} | {s_rec:9.1f} | {m_tot:7.1f} | {s_tot:12.1f} | {m_min:11.3f} | {s_min:11.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\nWrote: {out_csv}")
    print(f"Wall time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
