import numpy as np
from typing import Callable, Tuple, Optional

class BlindSearch:
    """
    Blind random search.

    - func: objective (accepts (d,) or (n,d))
    - bounds: (lb, ub) arrays
    - NP: samples per generation
    - g_max: number of generations
    - seed: RNG seed
    """
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray],
                 NP=50, g_max=100, seed: Optional[int]=None):
        # store objective and bounds as numpy arrays for vectorized operations
        self.func = func
        self.lb = np.asarray(bounds[0])
        self.ub = np.asarray(bounds[1])
        # dimensionality inferred from bounds
        self.d = self.lb.size
        # number of samples per generation, and maximum generations
        self.NP = int(NP)
        self.g_max = int(g_max)
        # initialize numpy random generator with provided seed for reproducibility
        self.rng = np.random.default_rng(seed)

    def random_solution(self, n=1):
        # draw n uniform samples in the box [lb, ub], shape (n, d)
        return self.rng.uniform(self.lb, self.ub, size=(n, self.d))

    def run(self, record_history=True):
        # initialize with a single random baseline solution
        x_b = self.random_solution(1)[0]
        # evaluate baseline objective
        f_b = float(self.func(x_b))
        # prepare history structure if requested
        history = {"best_x": [x_b.copy()], "best_f": [f_b], "sampled": []}

        # main loop: independent random sampling each generation
        for g in range(1, self.g_max + 1):
            # sample NP candidate solutions uniformly in bounds
            samples = self.random_solution(self.NP)
            # evaluate all samples; func may accept (n,d) to return vectorized values
            vals = np.asarray(self.func(samples))
            # pick index of best (minimum) evaluated value
            idx = np.argmin(vals)
            x_s = samples[idx].copy()
            f_s = float(vals[idx])

            # record the raw samples of this generation
            if record_history:
                history["sampled"].append(samples.copy())

            # if a sampled solution is better, update baseline
            if f_s < f_b:
                x_b = x_s.copy()
                f_b = f_s

            # record best-so-far after this generation
            if record_history:
                history["best_x"].append(x_b.copy())
                history["best_f"].append(f_b)

        # return best solution and optional history
        if record_history:
            return x_b, f_b, history
        return x_b, f_b
    
class HillClimbing:
    """
    Hill Climbing algorithm (minimization) using multiple normal neighbors per generation.
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
        # store objective and bounds as float arrays
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        # dimensionality of problem
        self.d = self.lb.size
        # number of neighbors per generation
        assert int(NP) > 0, "NP must be positive integer"
        self.NP = int(NP)
        self.g_max = int(g_max)
        # accept scalar or per-dimension sigma and normalize to length-d array
        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.size == 1:
            # broadcast scalar sigma to all dimensions
            self.sigma = np.full(self.d, float(sigma))
        # RNG for reproducible neighbor draws
        self.rng = np.random.default_rng(seed)

    def random_solution(self, n=1):
        """Random solution in bounds. Returns (n,d)."""
        # draw n uniform samples in the bounds box
        return self.rng.uniform(self.lb, self.ub, size=(n, self.d))

    def _sample_neighbors(self, center: np.ndarray):
        """
        Sample NP neighbors from multivariate normal N(center, diag(sigma^2)).
        Clip to bounds and return shape (NP, d).
        """
        # draw standard normal variates (NP, d)
        z = self.rng.standard_normal(size=(self.NP, self.d))
        # scale by sigma and shift to be centered at `center`
        samples = center.reshape(1, -1) + z * self.sigma.reshape(1, -1)
        # clip samples to respect bounds (in-place)
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
        # initialize baseline solution uniformly at random
        x_b = self.random_solution(1)[0]
        # evaluate baseline
        f_b = float(self.func(x_b))
        # prepare history container
        history = {"best_x": [x_b.copy()], "best_f": [f_b], "sampled": []}

        # iterate generations, sampling neighbors around current best
        for g in range(1, self.g_max + 1):
            # generate NP Gaussian neighbors centered at current best
            samples = self._sample_neighbors(x_b)
            # evaluate neighbors (vectorized if func supports it)
            vals = np.asarray(self.func(samples))
            # pick best neighbor index
            idx = np.argmin(vals)
            x_s = samples[idx].copy()
            f_s = float(vals[idx])

            # record sampled neighbors if requested
            if record_history:
                history["sampled"].append(samples.copy())

            # replace baseline with better neighbor
            if f_s < f_b:
                x_b = x_s.copy()
                f_b = f_s

            # append best-so-far after this generation
            if record_history:
                history["best_x"].append(x_b.copy())
                history["best_f"].append(f_b)

        # return best solution and optional history
        if record_history:
            return x_b, f_b, history
        return x_b, f_b
    
# --- SimulatedAnnealing class (copy into bio_opt/algorithms.py) ---

import numpy as np
from typing import Callable, Tuple, Optional

class SimulatedAnnealing:
    """Simulated Annealing for minimization."""

    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray], T0: float = 100.0,
                 Tmin: float = 0.5, alpha: float = 0.95, sigma: float = 0.1, max_iters: int = 100000, seed: Optional[int] = None):
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.d = self.lb.size
        self.T0 = float(T0)
        self.Tmin = float(Tmin)
        self.alpha = float(alpha)

        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.size == 1:
            self.sigma = np.full(self.d, float(sigma))

        self.max_iters = int(max_iters)
        self.rng = np.random.default_rng(seed)

    def random_solution(self, n=1):
        """Uniform random solution(s) in bounds: returns array shape (n,d)."""
        return self.rng.uniform(self.lb, self.ub, size=(n, self.d))

    def _neighbor(self, x: np.ndarray) -> np.ndarray:
        """Generate a single neighbor around x using Normal(center=x, sigma^2 I) and clip to bounds."""
        z = self.rng.standard_normal(size=(self.d,))
        x1 = x + z * self.sigma
        # Clip to bounds
        np.clip(x1, self.lb, self.ub, out=x1)
        return x1

    def run(self, record_history: bool = True):
        """Execute Simulated Annealing."""
        T = self.T0

        # initial solution
        x = self.random_solution(1)[0]
        f_x = float(self.func(x))

        # best-ever
        x_best = x.copy()
        f_best = f_x

        history = {"x": [x.copy()], "f": [f_x], "best_x": [x_best.copy()], "best_f": [f_best], "accepted": []} if record_history else None
 
        iters = 0
        # follow pseudocode: while T > Tmin
        while T > self.Tmin and iters < self.max_iters:
            x1 = self._neighbor(x)
            f_x1 = float(self.func(x1))

            accepted = False
            if f_x1 < f_x:
                # better -> accept
                x = x1
                f_x = f_x1
                accepted = True
            else:
                # worse -> accept with prob exp(-(f_x1 - f_x)/T)
                delta = f_x1 - f_x  # >= 0
                # if T==0 then prob=0 (can't accept worse)
                prob = np.exp(-delta / T) if T > 0 else 0.0
                r = self.rng.random()
                if r < prob:
                    x = x1
                    f_x = f_x1
                    accepted = True

            # update best-ever if improved
            if f_x < f_best:
                f_best = f_x
                x_best = x.copy()

            # record iteration
            if record_history:
                history["x"].append(x.copy())
                history["f"].append(f_x)
                history["best_x"].append(x_best.copy())
                history["best_f"].append(f_best)
                history["accepted"].append(bool(accepted))

            # cool down
            T = T * self.alpha
            iters += 1

        if record_history:
            return x_best, f_best, history
        return x_best, f_best
