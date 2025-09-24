import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

@dataclass
class Function:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: Tuple[np.ndarray, np.ndarray]
    global_minimum: Optional[np.ndarray] = None
    global_minimum_value: Optional[float] = None

    def evaluate(self, x: np.ndarray):
        return self.func(x)

    def is_2d(self) -> bool:
        lb, ub = self.bounds
        return lb.size == 2 and ub.size == 2


# --- internal helpers ---
def _ensure_2d(x: np.ndarray):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1), True
    return x, False


# --- functions ---
def sphere(x: np.ndarray):
    x2, w1 = _ensure_2d(x)
    res = np.sum(x2**2, axis=1)
    return float(res[0]) if w1 else res

def ackley(x: np.ndarray, a=20, b=0.2, c=2*np.pi):
    x2, w1 = _ensure_2d(x)
    d = x2.shape[1]
    sum_sq = np.sum(x2**2, axis=1)
    sum_cos = np.sum(np.cos(c * x2), axis=1)
    res = -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e
    return float(res[0]) if w1 else res

def rastrigin(x: np.ndarray, A=10.0):
    x2, w1 = _ensure_2d(x)
    d = x2.shape[1]
    res = A*d + np.sum(x2**2 - A * np.cos(2*np.pi*x2), axis=1)
    return float(res[0]) if w1 else res

def rosenbrock(x: np.ndarray):
    x2, w1 = _ensure_2d(x)
    res = np.sum(100.0 * (x2[:, 1:] - x2[:, :-1]**2)**2 + (x2[:, :-1] - 1.0)**2, axis=1)
    return float(res[0]) if w1 else res

def griewank(x: np.ndarray):
    x2, w1 = _ensure_2d(x)
    d = x2.shape[1]
    sum_term = np.sum(x2**2 / 4000.0, axis=1)
    idx = np.arange(1, d+1)
    prod_term = np.prod(np.cos(x2 / np.sqrt(idx)), axis=1)
    res = sum_term - prod_term + 1.0
    return float(res[0]) if w1 else res

def schwefel(x: np.ndarray):
    x2, w1 = _ensure_2d(x)
    d = x2.shape[1]
    res = 418.9829 * d - np.sum(x2 * np.sin(np.sqrt(np.abs(x2))), axis=1)
    return float(res[0]) if w1 else res

def levy(x: np.ndarray):
    x2, w1 = _ensure_2d(x)
    w = 1.0 + (x2 - 1.0) / 4.0
    term1 = np.sin(np.pi * w[:, 0])**2
    term3 = (w[:, -1] - 1.0)**2 * (1 + np.sin(2*np.pi*w[:, -1])**2)
    sum_term = np.sum((w[:, :-1] - 1.0)**2 * (1 + 10*np.sin(np.pi*w[:, :-1] + 1)**2), axis=1)
    res = term1 + sum_term + term3
    return float(res[0]) if w1 else res

def michalewicz(x: np.ndarray, m=10):
    x2, w1 = _ensure_2d(x)
    i = np.arange(1, x2.shape[1] + 1)
    term = np.sin(x2) * (np.sin(i * x2**2 / np.pi) ** (2*m))
    res = -np.sum(term, axis=1)
    return float(res[0]) if w1 else res

def zakharov(x: np.ndarray):
    x2, w1 = _ensure_2d(x)
    d = x2.shape[1]
    sum1 = np.sum(x2**2, axis=1)
    i = np.arange(1, d+1)
    sum2 = np.sum(0.5 * i * x2, axis=1)
    res = sum1 + sum2**2 + sum2**4
    return float(res[0]) if w1 else res


# --- registry ---
def _make_bounds(d, lower, upper):
    return np.full(d, lower), np.full(d, upper)

registry = {
    "Sphere": Function("Sphere", sphere, _make_bounds(2, -5.12, 5.12), global_minimum=np.zeros(2), global_minimum_value=0.0),
    "Ackley": Function("Ackley", ackley, _make_bounds(2, -32.768, 32.768), global_minimum=np.zeros(2), global_minimum_value=0.0),
    "Rastrigin": Function("Rastrigin", rastrigin, _make_bounds(2, -5.12, 5.12), global_minimum=np.zeros(2), global_minimum_value=0.0),
    "Rosenbrock": Function("Rosenbrock", rosenbrock, _make_bounds(2, -5.0, 10.0), global_minimum=np.ones(2), global_minimum_value=0.0),
    "Griewank": Function("Griewank", griewank, _make_bounds(2, -600.0, 600.0), global_minimum=np.zeros(2), global_minimum_value=0.0),
    "Schwefel": Function("Schwefel", schwefel, _make_bounds(2, -500.0, 500.0), global_minimum=np.full(2, 418.9829), global_minimum_value=0.0),
    "Levy": Function("Levy", levy, _make_bounds(2, -10.0, 10.0), global_minimum=np.ones(2), global_minimum_value=0.0),
    "Michalewicz": Function("Michalewicz", michalewicz, _make_bounds(2, 0.0, np.pi)),
    "Zakharov": Function("Zakharov", zakharov, _make_bounds(2, -5.0, 10.0), global_minimum=np.zeros(2), global_minimum_value=0.0),
}