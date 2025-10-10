import numpy as np
from typing import Callable, Tuple, Optional
from functions import Cities

class BlindSearch:
    """Blind random search."""
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray],
                 NP=50, g_max=100, seed: Optional[int]=None):
        """Initialize Blind Search optimizer."""
        self.func = func # objective function
        self.lb = np.asarray(bounds[0])
        self.ub = np.asarray(bounds[1])
        self.d = self.lb.size # dimensionality inferred from bounds
        self.NP = int(NP) # number of samples per generation
        self.g_max = int(g_max) # maximum generations
        self.rng = np.random.default_rng(seed) # RNG for reproducible sampling

    def random_solution(self, n=1):
        """Generate n random solutions uniformly in bounds."""
        return self.rng.uniform(self.lb, self.ub, size=(n, self.d))

    def run(self, record_history=True):
        """Execute Blind Search and return best_x, best_f, history (if requested)."""
        # initialize with a single random baseline solution
        x_b = self.random_solution(1)[0]
        f_b = float(self.func(x_b))
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
    """Hill Climbing algorithm (minimization) using multiple normal neighbors per generation."""
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray],
                 NP: int = 50, sigma: float = 0.1, g_max: int = 100, seed: Optional[int] = None):
        """Initialize Hill Climbing optimizer."""
        self.func = func # objective function
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.d = self.lb.size # dimensionality of problem
        self.NP = int(NP) # number of neighbors per generation
        self.g_max = int(g_max) # maximum generations
        self.sigma = np.asarray(sigma, dtype=float) # accept scalar or per-dimension sigma and normalize to length-d array
        if self.sigma.size == 1:
            self.sigma = np.full(self.d, float(sigma)) # broadcast scalar sigma to all dimensions
        self.rng = np.random.default_rng(seed) # RNG for reproducible neighbor draws

    def random_solution(self, n=1):
        """Random solution in bounds"""
        # draw n uniform samples in the bounds box
        return self.rng.uniform(self.lb, self.ub, size=(n, self.d))

    def _sample_neighbors(self, center: np.ndarray):
        """Sample NP neighbors from multivariate normal N(center, diag(sigma^2)) within bounds."""
        # draw standard normal variates (NP, d)
        z = self.rng.standard_normal(size=(self.NP, self.d))
        # scale by sigma and shift to be centered at `center`
        samples = center.reshape(1, -1) + z * self.sigma.reshape(1, -1)
        # clip samples to respect bounds (in-place)
        np.clip(samples, self.lb, self.ub, out=samples)
        return samples

    def run(self, record_history=True):
        """Execute Hill Climbing and return best_x, best_f, history (if requested)."""
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

class SimulatedAnnealing:
    """Simulated Annealing for minimization."""
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray], T0: float = 100.0,
                 Tmin: float = 0.5, alpha: float = 0.95, sigma: float = 0.1, max_iters: int = 100000, seed: Optional[int] = None):
        """Initialize Simulated Annealing optimizer."""
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
        # initial solution
        T = self.T0
        x = self.random_solution(1)[0]
        f_x = float(self.func(x))
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
                delta = f_x1 - f_x
                # if T==0 then prob=0
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

class GeneticAlgorithm:
    """Genetic Algorithm for solving the TSP (Traveling Salesman Problem) using permutation encoding."""
    def __init__(self, cities: Cities, NP: int = 50, G: int = 200, mutation_prob: float = 0.5, max_mutation_strength: float = 0.1, seed: Optional[int] = None):
        """Initialize Genetic Algorithm for TSP."""
        self.cities = cities # Cities instance with coordinates and distance methods
        self.n_cities = cities.n # number of cities
        cities.distance_matrix() # Precompute distance matrix for efficiency
        self.NP = max(int(NP), 2) # population size, at least 2
        self.G = int(G) # number of generations
        self.mutation_prob = float(mutation_prob)
        self.max_mutation = min(max(int(max_mutation_strength * self.n_cities), 2), self.n_cities) # Maximum number of cities to mutate in a swap
        self.rng = np.random.default_rng(seed) # Random number generator

    def _random_population(self):
        """Generate initial population of random tours (permutations)."""
        pop = np.empty((self.NP, self.n_cities), dtype=int)
        for i in range(self.NP):
            # Each individual is a random permutation of city indices
            pop[i] = self.rng.permutation(self.n_cities)
        return pop

    def _evaluate_population(self, population):
        """Evaluate the total tour length for each individual in the population."""
        lengths = np.empty(population.shape[0], dtype=float)
        for i, tour in enumerate(population):
            lengths[i] = self.cities.total_distance(tour)
        return lengths

    def _order_crossover(self, parent_a, parent_b):
        """Perform order crossover (OX) between two parent tours."""
        n = self.n_cities
        child = -np.ones(n, dtype=int)
        # Randomly select a subsequence from parent_a
        i, j = sorted(self.rng.choice(n, size=2, replace=False))
        child[i:j+1] = parent_a[i:j+1]
        # Fill remaining positions with cities from parent_b in order, skipping duplicates
        fill_pos = (j + 1) % n
        b_idx = (j + 1) % n
        while np.any(child == -1):
            city = parent_b[b_idx]
            if city not in child:
                child[fill_pos] = city
                fill_pos = (fill_pos + 1) % n
            b_idx = (b_idx + 1) % n
        return child

    def _mutate_swap(self, tour):
        """Mutate a tour by randomly swapping a subset of cities."""
        # Decide whether to mutate based on mutation_prob
        if self.rng.random() >= self.mutation_prob:
            return tour.copy()
        # Choose number of cities to mutate: between 2 and max_mutation
        n_mut = self.rng.integers(2, self.max_mutation + 1)
        idx = self.rng.choice(self.n_cities, size=n_mut, replace=False)
        # Shuffle the selected indices
        shuffled = idx.copy()
        self.rng.shuffle(shuffled)
        t = tour.copy()
        # Assign shuffled cities to the selected indices
        t[idx] = t[shuffled]
        return t

    def run(self, record_history=True):
        """Run the genetic algorithm for G generations."""
        # Initialize population and evaluate
        population = self._random_population()
        lengths = self._evaluate_population(population)
        best_idx = int(np.argmin(lengths)) 
        best_tour = population[best_idx].copy()
        best_len = float(lengths[best_idx])
        # History records best tour and population best per generation
        history = {"best_tour": [best_tour.copy()], "best_f": [best_len], "pop_best": [best_len]}
        for gen in range(1, self.G+1):
            # Copy current population and fitness
            new_pop = population.copy()
            new_lengths = lengths.copy()
            for j in range(self.NP):
                # Select parent A (current individual)
                parent_A = population[j]
                # Select parent B (different individual)
                k = j
                while k == j:
                    k = int(self.rng.integers(0, self.NP))
                parent_B = population[k]
                # Generate offspring by crossover and mutation
                offspring = self._order_crossover(parent_A, parent_B)
                offspring = self._mutate_swap(offspring)
                off_length = self.cities.total_distance(offspring)
                # Replace parent if offspring is better or equal
                if off_length <= new_lengths[j]:
                    new_pop[j] = offspring
                    new_lengths[j] = off_length
            # Update population and fitness
            population = new_pop
            lengths = new_lengths
            # Find best in current generation
            gen_best_idx = int(np.argmin(lengths))
            gen_best_len = float(lengths[gen_best_idx])
            gen_best_tour = population[gen_best_idx].copy()
            # Update overall best if improved
            if gen_best_len < best_len:
                best_len = gen_best_len
                best_tour = gen_best_tour.copy()
            # Record history if requested
            if record_history:
                history["best_tour"].append(best_tour.copy())
                history["best_f"].append(best_len)
                history["pop_best"].append(gen_best_len)
        if record_history:
            return best_tour, best_len, history
        return best_tour, best_len

class DifferentialEvolution:
    """Differential Evolution (DE) for minimization."""
    def __init__(self, func: Callable, bounds: Tuple, NP: int = 50, F: float = 0.8,
                 CR: float = 0.9, g_max: int = 200, seed: Optional[int] = None):
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.d = self.lb.size
        self.NP = max(int(NP), 4) # number of population members
        self.F = float(F) # differential weight
        self.CR = float(CR) # crossover probability
        self.g_max = int(g_max) # max generations
        self.rng = np.random.default_rng(seed)

    def _rand_population(self):
        """Initial population uniformly sampled in the box [lb, ub]."""
        return self.rng.uniform(self.lb, self.ub, size=(self.NP, self.d))

    def run(self, record_history = True):
        """Execute DE and return best solution, value, and optional history dict."""
        pop = self._rand_population() # initial population
        fitness = np.asarray(self.func(pop)) # evaluate population (batch)
        best_idx = int(np.argmin(fitness))
        best = pop[best_idx].copy()
        best_f = float(fitness[best_idx])
        history = {"best_x":[best.copy()], "best_f":[best_f], "pop":[pop.copy()]} if record_history else None

        # main loop
        for g in range(1, self.g_max+1):
            new_pop = pop.copy()
            new_fit = fitness.copy()

            for i in range(self.NP):
                # select r1, r2, r3 distinct and not equal to current index i
                idxs = list(range(self.NP))
                idxs.remove(i)
                r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)

                x_r1 = pop[r1]
                x_r2 = pop[r2]
                x_r3 = pop[r3]

                # mutation (v = x_r3 + F*(x_r1 - x_r2))
                v = x_r3 + self.F * (x_r1 - x_r2)
                # clip mutated vector to bounds
                v = np.clip(v, self.lb, self.ub)

                # binomial crossover
                jrand = int(self.rng.integers(0, self.d))
                trial = np.empty(self.d, dtype=float)
                for j in range(self.d):
                    if self.rng.random() < self.CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i, j]

                # selection
                f_trial = float(self.func(trial))
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = f_trial
                    # update global best if improved
                    if f_trial < best_f:
                        best_f = f_trial
                        best = trial.copy()

            # replace population
            pop = new_pop
            fitness = new_fit

            if record_history:
                history["best_x"].append(best.copy())
                history["best_f"].append(best_f)
                history["pop"].append(pop.copy())

        return (best, best_f, history) if record_history else (best, best_f)