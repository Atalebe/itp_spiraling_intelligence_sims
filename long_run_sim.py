import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helper functions
# -----------------------------

def cosine_alignment(w, v):
    """Cosine similarity between weight vector w and pattern v."""
    wn = np.linalg.norm(w)
    vn = np.linalg.norm(v)
    if wn == 0 or vn == 0:
        return 0.0
    return float(np.dot(w, v) / (wn * vn))


def compute_phase_stats(series, idx_start):
    """
    Compute min/mean/max over a given phase of the time series.
    series: list or np.array of alignment values over time
    idx_start: index where the phase begins
    """
    arr = np.array(series[idx_start:], dtype=float)
    return float(arr.min()), float(arr.mean()), float(arr.max())


# -----------------------------
# Agents
# -----------------------------

class MarkovAgent:
    """
    Simple Markovian agent: no rotation, no sleep, just gradient-like updates.
    """
    def __init__(self, dim, lr=0.1):
        self.w = np.zeros(dim, dtype=float)
        self.lr = lr

    def update(self, x, target_sign):
        """
        Perceptron-style update toward target_sign in {+1, -1}.
        """
        y_raw = np.dot(self.w, x)

        # Treat zero as WRONG so it learns from the start
        y = 1.0 if y_raw > 0.0 else -1.0

        error = target_sign - y
        self.w += self.lr * error * x

        # Normalize to keep bounded
        norm = np.linalg.norm(self.w)
        if norm > 0:
            self.w /= norm


class SpiralAgent:
    """
    Spiral (ITP-style) agent with:
      - rotational drift (Omega_eff, ramped by consolidation)
      - consolidation C(t) (plasticity ratchet)
      - sleep step (pruning & gentle consolidation)
    """
    def __init__(self, dim, base_lr=0.08, base_omega=0.04, kappa=0.005):
        # Start with a small random direction so it can respond immediately
        w0 = np.random.randn(dim)
        w0 = w0 / np.linalg.norm(w0)
        self.w = w0

        self.base_lr = base_lr
        self.base_omega = base_omega
        self.kappa = kappa

        self.C = 0.0   # consolidation (0 = fresh, 1 = hardened)

    def _effective_omega(self):
        """
        Rotation speed ramps with consolidation:
        - when C ~ 0: almost no rotation (learn first)
        - when C ~ 1: full base_omega (mature spiral)
        """
        return self.base_omega * (0.2 + 0.8 * self.C)

    def _rotate(self):
        """Apply a 2D rotation by angle omega_eff to self.w."""
        theta = self._effective_omega()
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s],
                      [s,  c]])
        self.w = R @ self.w

    def update(self, x, target_sign):
        """
        Wake update: rotation + plastic update with consolidation ratchet.
        """
        # First, internal drift
        self._rotate()

        # Then perceptron-like update
        y_raw = np.dot(self.w, x)
        y = 1.0 if y_raw > 0.0 else -1.0

        error = target_sign - y

        # Plasticity reduced by consolidation C
        plasticity = 1.0 - self.C
        lr_eff = self.base_lr * (0.5 + 0.5 * (1.0 - self.C))  # slightly higher when fresh

        self.w += lr_eff * plasticity * error * x

        # Normalize
        norm = np.linalg.norm(self.w)
        if norm > 0:
            self.w /= norm

        # Consolidation increases when error is small (we've learned)
        self.C += self.kappa * max(0.0, 1.0 - abs(error))
        self.C = min(self.C, 1.0)

    def sleep_step(self):
        """
        Sleep-phase step: gentle rotation + pruning of tiny components.
        """
        self._rotate()

        # Prune tiny components (entropic waste removal)
        thresh = 1e-3
        self.w[np.abs(self.w) < thresh] = 0.0

        # Normalize
        norm = np.linalg.norm(self.w)
        if norm > 0:
            self.w /= norm


# -----------------------------
# Simulation setup
# -----------------------------

def main():
    np.random.seed(0)

    dim = 2

    # Task A and B patterns (orthogonal)
    pattern_A = np.array([1.0, 0.0])
    pattern_B = np.array([0.0, 1.0])

    pattern_A /= np.linalg.norm(pattern_A)
    pattern_B /= np.linalg.norm(pattern_B)

    # Agents
    markov = MarkovAgent(dim=dim, lr=0.2)
    spiral = SpiralAgent(dim=dim, base_lr=0.08, base_omega=0.04, kappa=0.005)

    # Time structure
    T_A = 150   # Phase A: only Task A
    T_B = 150   # Phase B: only Task B
    T_C = 700   # Phase C: long mixed future
    T_total = T_A + T_B + T_C

    align_A_markov = []
    align_A_spiral = []

    # -------------------------
    # Phase A: only Task A
    # -------------------------
    for t in range(T_A):
        x = pattern_A
        target_sign = 1.0

        markov.update(x, target_sign)
        spiral.update(x, target_sign)

        align_A_markov.append(cosine_alignment(markov.w, pattern_A))
        align_A_spiral.append(cosine_alignment(spiral.w, pattern_A))

        # Occasional sleep for spiral
        if (t + 1) % 30 == 0:
            spiral.sleep_step()

    # -------------------------
    # Phase B: only Task B
    # -------------------------
    for t in range(T_B):
        x = pattern_B
        target_sign = 1.0

        markov.update(x, target_sign)
        spiral.update(x, target_sign)

        align_A_markov.append(cosine_alignment(markov.w, pattern_A))
        align_A_spiral.append(cosine_alignment(spiral.w, pattern_A))

        if (T_A + t + 1) % 30 == 0:
            spiral.sleep_step()

    # -------------------------
    # Phase C: mixed future
    # -------------------------
    for t in range(T_C):
        if np.random.rand() < 0.5:
            x = pattern_A
        else:
            x = pattern_B
        target_sign = 1.0

        markov.update(x, target_sign)
        spiral.update(x, target_sign)

        align_A_markov.append(cosine_alignment(markov.w, pattern_A))
        align_A_spiral.append(cosine_alignment(spiral.w, pattern_A))

        if (T_A + T_B + t + 1) % 30 == 0:
            spiral.sleep_step()

    # Convert to arrays
    align_A_markov = np.array(align_A_markov, dtype=float)
    align_A_spiral = np.array(align_A_spiral, dtype=float)

    # -----------------------------
    # Print summary statistics
    # -----------------------------

    print("\nFinal alignment with Task A:")
    print(f"  Markov: {align_A_markov[-1]:6.3f}")
    print(f"  Spiral: {align_A_spiral[-1]:6.3f}")

    phaseC_start = T_A + T_B
    m_min, m_mean, m_max = compute_phase_stats(align_A_markov, phaseC_start)
    s_min, s_mean, s_max = compute_phase_stats(align_A_spiral, phaseC_start)

    print("\nPhase C (long-term) alignment with Task A:")
    print(f"  Markov -> min: {m_min:6.3f}, mean: {m_mean:6.3f}, max: {m_max:6.3f}")
    print(f"  Spiral -> min: {s_min:6.3f}, mean: {s_mean:6.3f}, max: {s_max:6.3f}")

    # -----------------------------
    # Plotting
    # -----------------------------

    t_axis = np.arange(T_total)

    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, align_A_markov, label="Markov (Task A alignment)")
    plt.plot(t_axis, align_A_spiral, label="Spiral (Task A alignment)")

    plt.axvspan(0, T_A, color="green", alpha=0.05, label="Phase A")
    plt.axvspan(T_A, T_A + T_B, color="orange", alpha=0.05, label="Phase B")
    plt.axvspan(T_A + T_B, T_total, color="blue", alpha=0.03, label="Phase C")

    plt.xlabel("Time step")
    plt.ylabel("Alignment with Task A (cosine)")
    plt.title("Long-run alignment with Task A: Markov vs Spiral Agent")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
