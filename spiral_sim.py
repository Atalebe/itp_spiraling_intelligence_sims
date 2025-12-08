import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# SpiralAgent with Sleep + Stress
# -----------------------------

class SpiralAgent:
    def __init__(self,
                 base_learning_rate=0.05,
                 base_omega=0.12,
                 kappa=0.01,
                 beta_stress=0.95,
                 beta_frustration=0.98,
                 stress_threshold=0.5):
        """
        2D Spiral Agent implementing:
          - Wake: rotation + Hebbian update with consolidation
          - Sleep: rotation + pruning + tiny consolidation
          - Meta-layer: stress & frustration modulate eta and omega
        """

        # 2D weight vector
        self.w = np.random.randn(2)
        self.w = self.w / (np.linalg.norm(self.w) + 1e-8)

        # "true concept" direction (for error/stress)
        self.base = np.array([1.0, 1.0])
        self.base = self.base / (np.linalg.norm(self.base) + 1e-8)

        # consolidation state in [0,1]
        self.C = 0.0

        # hyperparameter baselines
        self.base_eta = base_learning_rate
        self.base_omega = base_omega
        self.kappa = kappa

        # current effective hyperparameters (will be updated)
        self.eta = base_learning_rate
        self.omega = base_omega

        # meta-cognition: stress & frustration
        self.stress = 0.0
        self.frustration = 0.0
        self.beta_stress = beta_stress
        self.beta_frustration = beta_frustration
        self.stress_threshold = stress_threshold

        # history for plotting
        self.traj = []          # weight vectors
        self.phase_labels = []  # "wake" or "sleep"
        self.stress_hist = []   # stress over time
        self.frustration_hist = []  # frustration over time
        self.eta_hist = []      # learning rate over time
        self.omega_hist = []    # rotation speed over time

    # ---------- UTILITIES ----------

    def rotation_matrix(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s],
                         [s,  c]])

    def _record(self, phase):
        self.traj.append(self.w.copy())
        self.phase_labels.append(phase)
        self.stress_hist.append(self.stress)
        self.frustration_hist.append(self.frustration)
        self.eta_hist.append(self.eta)
        self.omega_hist.append(self.omega)

    # ---------- META-CONTROL (STRESS) ----------

    def update_stress_and_frustration(self, error):
        """
        error: scalar in [0, 2] roughly
        """
        # leaky integrator for stress
        self.stress = (
            self.beta_stress * self.stress
            + (1.0 - self.beta_stress) * error
        )

        # frustration: leaky integrator of "excess" stress
        excess = max(0.0, self.stress - self.stress_threshold)
        self.frustration = (
            self.beta_frustration * self.frustration
            + (1.0 - self.beta_frustration) * excess
        )

        # now update eta and omega based on mood:
        # - higher stress => higher learning rate (panic learning)
        # - higher frustration => higher rotation (agitated search)

        # cap factors to keep things sane
        lr_factor = 1.0 + 5.0 * (self.stress ** 2)      # up to ~6x
        omega_factor = 1.0 + 3.0 * self.frustration     # can grow but slowly

        self.eta = self.base_eta * lr_factor
        self.omega = self.base_omega * omega_factor

    # ---------- WAKE PHASE ----------

    def wake_step(self, x):
        """
        One wake step:
          - measure error vs base concept
          - update stress/frustration -> eta, omega
          - rotate w by omega
          - Hebbian update with (1 - C_t)
          - consolidation update
        """
        # normalize input
        x = x / (np.linalg.norm(x) + 1e-8)

        # "output" as projection
        y = np.dot(self.w, x)

        # define error relative to base direction:
        # if w perfectly aligned with base => error ~0
        # if w opposite => error ~2
        alignment = np.dot(self.w, self.base)  # in [-1, 1]
        error = 1.0 - alignment                # in [0, 2]

        # update meta-state (stress/frustration) and hyperparams
        self.update_stress_and_frustration(error)

        # intrinsic rotation with current omega
        R = self.rotation_matrix(self.omega)
        w_rot = R @ self.w

        # Hebbian increment with current eta and (1 - C)
        delta_w = self.eta * (1.0 - self.C) * y * x

        # update weights and renormalize
        self.w = w_rot + delta_w
        self.w = self.w / (np.linalg.norm(self.w) + 1e-8)

        # consolidation update (hardens over time)
        self.C = min(1.0, self.C + self.kappa * abs(y))

        self._record("wake")

    def wake_phase(self, steps=50, noise_level=0.1):
        """
        Several wake steps with noisy versions of the same concept.
        """
        base = self.base.copy()
        for _ in range(steps):
            noise = np.random.normal(0.0, noise_level, size=2)
            x_t = base + noise
            self.wake_step(x_t)

    # ---------- SLEEP PHASE ----------

    def sleep_step(self, pruning_strength=0.1):
        """
        One sleep step:
          - no external input
          - slow rotation
          - dream input based on current w
          - tiny consolidation
          - pruning of small components
        """
        # during sleep, we don't define supervised error,
        # but we can still let stress slowly relax if you like.
        # here, we just let stress/frustration decay passively:
        self.stress *= self.beta_stress
        self.frustration *= self.beta_frustration

        # keep effective hyperparameters updated after decay
        self.update_stress_and_frustration(error=0.0)

        # slower rotation
        theta = self.omega * 0.5
        R = self.rotation_matrix(theta)
        self.w = R @ self.w

        # "dream input" = current direction
        dream_input = self.w / (np.linalg.norm(self.w) + 1e-8)
        y = np.dot(self.w, dream_input)

        # tiny consolidation during sleep
        self.C = min(1.0, self.C + 0.5 * self.kappa * abs(y))

        # pruning: shrink each component slightly toward 0
        self.w -= pruning_strength * np.sign(self.w) * 0.01

        # renormalize
        self.w = self.w / (np.linalg.norm(self.w) + 1e-8)

        self._record("sleep")

    def sleep_phase(self, steps=20, pruning_strength=0.1):
        for _ in range(steps):
            self.sleep_step(pruning_strength=pruning_strength)

    # ---------- DRIVER ----------

    def run_days(self, n_days=3,
                 wake_steps=50,
                 sleep_steps=20,
                 noise_level=0.1,
                 pruning_strength=0.1):
        for _ in range(n_days):
            self.wake_phase(steps=wake_steps, noise_level=noise_level)
            self.sleep_phase(steps=sleep_steps, pruning_strength=pruning_strength)

    # ---------- PLOTTING ----------

    def plot_trajectory(self):
        traj = np.array(self.traj)
        wake_idxs = [i for i, p in enumerate(self.phase_labels) if p == "wake"]
        sleep_idxs = [i for i, p in enumerate(self.phase_labels) if p == "sleep"]

        plt.figure()

        if wake_idxs:
            plt.plot(traj[wake_idxs, 0],
                     traj[wake_idxs, 1],
                     '.', label="wake", alpha=0.8)
        if sleep_idxs:
            plt.plot(traj[sleep_idxs, 0],
                     traj[sleep_idxs, 1],
                     'x', label="sleep", alpha=0.7)

        plt.scatter(traj[0, 0], traj[0, 1], label="start", s=60)
        plt.scatter(traj[-1, 0], traj[-1, 1], label="end", s=60)

        plt.axhline(0, linewidth=0.5)
        plt.axvline(0, linewidth=0.5)
        plt.xlabel("w1")
        plt.ylabel("w2")
        plt.title("SpiralAgent with Wake + Sleep + Stress")
        plt.legend()
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.show()

    def plot_stress(self):
        """
        Plot stress, frustration, learning rate, and omega over time.
        """
        steps = np.arange(len(self.stress_hist))

        fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

        axes[0].plot(steps, self.stress_hist)
        axes[0].set_ylabel("Stress")

        axes[1].plot(steps, self.frustration_hist)
        axes[1].set_ylabel("Frustration")

        axes[2].plot(steps, self.eta_hist)
        axes[2].set_ylabel("eta (LR)")

        axes[3].plot(steps, self.omega_hist)
        axes[3].set_ylabel("omega")
        axes[3].set_xlabel("Step")

        plt.tight_layout()
        plt.show()


def main():
    agent = SpiralAgent(
        base_learning_rate=0.03,
        base_omega=0.10,
        kappa=0.01,
        beta_stress=0.95,
        beta_frustration=0.98,
        stress_threshold=0.5
    )

    agent.run_days(
        n_days=5,
        wake_steps=60,
        sleep_steps=30,
        noise_level=0.15,
        pruning_strength=0.08
    )

    agent.plot_trajectory()
    agent.plot_stress()


if __name__ == "__main__":
    main()
