import numpy as np
from typing import Callable, Tuple, Optional

class BlindSearch:
    """
    Simple blind random search.

    - func: objective (accepts (d,) or (n,d))
    - bounds: (lb, ub) arrays
    - NP: samples per generation
    - g_max: number of generations
    - seed: RNG seed
    """
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray],
                 NP=50, g_max=100, seed: Optional[int]=None):
        self.func = func
        self.lb = np.asarray(bounds[0])
        self.ub = np.asarray(bounds[1])
        assert self.lb.shape == self.ub.shape
        self.d = self.lb.size
        self.NP = int(NP)
        self.g_max = int(g_max)
        self.rng = np.random.default_rng(seed)

    def random_solution(self, n=1):
        return self.rng.uniform(self.lb, self.ub, size=(n, self.d))

    def run(self, record_history=True):
        x_b = self.random_solution(1)[0]
        f_b = float(self.func(x_b))
        history = {"best_x": [x_b.copy()], "best_f": [f_b], "sampled": []}

        for g in range(1, self.g_max + 1):
            samples = self.random_solution(self.NP)
            vals = np.asarray(self.func(samples))
            idx = np.argmin(vals)
            x_s = samples[idx].copy()
            f_s = float(vals[idx])

            if record_history:
                history["sampled"].append(samples.copy())

            if f_s < f_b:
                x_b = x_s.copy()
                f_b = f_s

            if record_history:
                history["best_x"].append(x_b.copy())
                history["best_f"].append(f_b)

        if record_history:
            return x_b, f_b, history
        return x_b, f_b
    
class HillClimbing:
    """
    Hill Climbing algorithm (minimization) using multiple normal neighbors per generation.

    Flowchart / behavior:
      - Start: sample baseline x_b uniformly in bounds.
      - For g = 1..g_max:
          - Generate NP neighbors ~ N(mean = x_b, cov = sigma^2 * I)
          - Clip neighbors to bounds
          - Evaluate neighbors and pick best x_s
          - If f(x_s) < f(x_b): x_b = x_s
      - Return x_b (best found)

    Parameters:
      - func: objective (accepts (d,) or (n,d))
      - bounds: (lb, ub) arrays
      - NP: neighbors per generation (must be >=1)
      - sigma: standard deviation for normal neighbor generation (scalar or array-like length d)
      - g_max: number generations
      - seed: RNG seed (int or None)
    """
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray],
                 NP: int = 50, sigma: float = 0.1, g_max: int = 100, seed: Optional[int] = None):
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        assert self.lb.shape == self.ub.shape, "Bounds must match in shape"
        self.d = self.lb.size
        assert int(NP) > 0, "NP must be positive integer"
        self.NP = int(NP)
        self.g_max = int(g_max)
        # Accept scalar or vector sigma
        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.size == 1:
            self.sigma = np.full(self.d, float(sigma))
        assert self.sigma.shape[0] == self.d, "sigma must be scalar or of shape (d,)"
        assert np.all(self.sigma >= 0), "sigma must be non-negative"
        self.rng = np.random.default_rng(seed)

    def random_solution(self, n=1):
        """Uniform random solution(s) in bounds. Returns (n,d)."""
        return self.rng.uniform(self.lb, self.ub, size=(n, self.d))

    def _sample_neighbors(self, center: np.ndarray):
        """
        Sample NP neighbors from multivariate normal N(center, diag(sigma^2)).
        Clip to bounds and return shape (NP, d).
        """
        # Draw from standard normal (NP, d) and scale by sigma
        z = self.rng.standard_normal(size=(self.NP, self.d))
        samples = center.reshape(1, -1) + z * self.sigma.reshape(1, -1)
        # Clip to bounds
        np.clip(samples, self.lb, self.ub, out=samples)
        return samples

    def run(self, record_history=True):
        """
        Execute Hill Climbing and return best_x, best_f, history (if requested).
        history fields:
          - best_x: list of best-so-far vectors (length g_max+1)
          - best_f: list of best-so-far objective values (length g_max+1)
          - sampled: list of arrays (NP,d) for each generation (length g_max)
        """
        # initialize baseline
        x_b = self.random_solution(1)[0]
        f_b = float(self.func(x_b))
        history = {"best_x": [x_b.copy()], "best_f": [f_b], "sampled": []}

        for g in range(1, self.g_max + 1):
            # sample neighbors around current best
            samples = self._sample_neighbors(x_b)
            vals = np.asarray(self.func(samples))
            idx = np.argmin(vals)
            x_s = samples[idx].copy()
            f_s = float(vals[idx])

            if record_history:
                history["sampled"].append(samples.copy())

            # acceptance: greedy minimization
            if f_s < f_b:
                x_b = x_s.copy()
                f_b = f_s

            if record_history:
                history["best_x"].append(x_b.copy())
                history["best_f"].append(f_b)

        if record_history:
            return x_b, f_b, history
        return x_b, f_b