import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
# ============================================================
# Helper functions
# ============================================================

def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# ============================================================
# Markov (baseline) agent
# ============================================================

class MarkovAgent:
    def __init__(self):
        self.w = normalize(np.random.randn(2))

    def update(self, task_vector, lr=0.05):
        y = float(np.dot(self.w, task_vector))
        dw = lr * y * (task_vector - y * self.w)
        self.w = normalize(self.w + dw)


# ============================================================
# Spiral (SIA) Agent with Rotation + Sleep
# ============================================================

class SpiralAgent:
    def __init__(self, omega=0.05, lr=0.05):
        self.w = normalize(np.random.randn(2))
        self.omega = omega
        self.lr = lr

    def rotate(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return R @ self.w

    def wake_update(self, task_vector):
        # 1. Hebbian term
        y = float(np.dot(self.w, task_vector))
        dw_hebb = self.lr * y * (task_vector - y * self.w)

        # 2. Rotation
        self.w = normalize(self.rotate(self.omega) + dw_hebb)

    def sleep_cycle(self, steps=10):
        for _ in range(steps):
            # Slower drift
            self.w = self.rotate(self.omega * 0.2)
            self.w = normalize(self.w)


# ============================================================
# Simulation
# ============================================================

def simulate_two_tasks():
    # Define two non-orthogonal tasks
    A = normalize(np.array([1.0, 1.0]))
    B = normalize(np.array([1.0, -0.2]))

    # Agents
    M = MarkovAgent()
    S = SpiralAgent()

    # History storage for plots
    M_hist, S_hist = [], []
    M_align_A, M_align_B = [], []
    S_align_A, S_align_B = [], []

    # ------------------------------
    # Phase 1: Learn Task A
    # ------------------------------
    for t in range(150):
        M.update(A, lr=0.05)
        S.wake_update(A)

        M_hist.append(M.w.copy())
        S_hist.append(S.w.copy())
        M_align_A.append(np.dot(M.w, A))
        M_align_B.append(np.dot(M.w, B))
        S_align_A.append(np.dot(S.w, A))
        S_align_B.append(np.dot(S.w, B))

    # ------------------------------
    # Sleep Spiral
    # ------------------------------
    for _ in range(40):
        S.sleep_cycle(steps=1)
        S_hist.append(S.w.copy())
        S_align_A.append(np.dot(S.w, A))
        S_align_B.append(np.dot(S.w, B))

    # ------------------------------
    # Phase 2: Learn Task B
    # ------------------------------
    for t in range(150):
        M.update(B, lr=0.05)
        S.wake_update(B)

        M_hist.append(M.w.copy())
        S_hist.append(S.w.copy())
        M_align_A.append(np.dot(M.w, A))
        M_align_B.append(np.dot(M.w, B))
        S_align_A.append(np.dot(S.w, A))
        S_align_B.append(np.dot(S.w, B))

    # ============================================================
    # Compute key alignments
    # ============================================================
    print("\n=== Alignment with Task A at key checkpoints ===")
    print("Markov:  after A-phase (t≈150): ", round(M_align_A[149], 3))
    print("Markov:  after B-phase (t≈300): ", round(M_align_A[-1], 3))
    print("Spiral:  after A-phase (t≈150): ", round(S_align_A[149], 3))
    print("Spiral:  after B-phase (t≈300): ", round(S_align_A[-1], 3))
    print("===============================================\n")

    # ============================================================
    # FIGURE (Three Panels)
    # ============================================================

    M_hist = np.array(M_hist)
    S_hist = np.array(S_hist)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ------------------------------------------------------------
    # Left panel: Weight trajectories in 2D
    # ------------------------------------------------------------
    axes[0].plot(M_hist[:,0], M_hist[:,1], 'b-', alpha=0.7, label="Markov")
    axes[0].plot(S_hist[:,0], S_hist[:,1], 'r-', alpha=0.7, label="Spiral (SIA)")
    axes[0].set_title("Weight Trajectories")
    axes[0].set_xlabel("w1")
    axes[0].set_ylabel("w2")
    axes[0].legend()
    axes[0].axis('equal')

    # ------------------------------------------------------------
    # Middle panel: Alignment with A
    # ------------------------------------------------------------
    axes[1].plot(M_align_A, 'b-', label="Markov")
    axes[1].plot(S_align_A, 'r--', label="Spiral")
    axes[1].set_title("Alignment with Task A")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Dot Product")
    axes[1].legend()

    # ------------------------------------------------------------
    # Right panel: Alignment with B
    # ------------------------------------------------------------
    axes[2].plot(M_align_B, 'b-', label="Markov")
    axes[2].plot(S_align_B, 'r--', label="Spiral")
    axes[2].set_title("Alignment with Task B")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Dot Product")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("two_task_sim_summary.png", dpi=300)
    plt.show()


# ============================================================
# Run the simulation
# ============================================================

if __name__ == "__main__":
    simulate_two_tasks()
