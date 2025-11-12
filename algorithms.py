import numpy as np
from typing import Callable, Tuple, Optional, List, Any, Dict
from functions import Cities

class BlindSearch:
    """Blind random search."""
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray],
                 NP=50, g_max=100, seed: Optional[int]=None):
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
    
class ParticleSwarmOptimizer:
    """Particle Swarm Optimization (global-best PSO) for minimization."""
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray],
                 pop_size: int = 50, w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 v_max: Optional[float] = None, max_iter: int = 200, seed: Optional[int] = None,
                 w_decay: bool = False):
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        if self.lb.shape != self.ub.shape:
            raise ValueError("lb and ub must have same shape")
        self.d = self.lb.size
        self.pop_size = int(pop_size)

        self.w = w # inertia weight
        self.w_decay = bool(w_decay) # whether to decay inertia weight over iterations
        if self.w_decay and (not hasattr(self.w, "__iter__")):
            self.w = (0.9, 0.4) # default decay if user set w_decay True but gave single w

        self.v_max = 0.2 * (self.ub - self.lb) if v_max is None else np.asarray(v_max, dtype=float)
        self.c1 = float(c1) # cognitive coefficient
        self.c2 = float(c2) # social coefficient
        self.max_iter = int(max_iter) # max iterations
        self.rng = np.random.default_rng(seed) # RNG for reproducibility

    def _init_swarm(self):
        """Initialize positions and velocities."""
        X = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.d))
        V = self.rng.uniform(-1.0, 1.0, size=(self.pop_size, self.d)) * self.v_max.reshape(1, -1)
        return X, V

    def run(self, record_history: bool = True):
        """Execute PSO. Returns best position, best value, and history (if requested)."""
        X, V = self._init_swarm()
        # evaluate initial positions
        fitness = np.asarray(self.func(X))
        pbest = X.copy()
        pbest_f = fitness.copy()
        gbest_idx = int(np.argmin(pbest_f))
        gbest = pbest[gbest_idx].copy()
        gbest_f = float(pbest_f[gbest_idx])

        history = {"best_x":[gbest.copy()], "best_f":[gbest_f]} if record_history else None

        for t in range(self.max_iter):
            # calculate current inertia weight
            if self.w_decay or hasattr(self.w, "__iter__"):
                w0, wf = self.w
                w = w0 + (wf - w0) * (t / max(1, self.max_iter-1))
            else:
                w = float(self.w)   

            r1 = self.rng.random(size=(self.pop_size, self.d))
            r2 = self.rng.random(size=(self.pop_size, self.d))

            # velocity update
            cognitive = self.c1 * r1 * (pbest - X)
            social = self.c2 * r2 * (gbest.reshape(1, -1) - X)
            V = w * V + cognitive + social

            # clamp velocity
            V = np.clip(V, -self.v_max.reshape(1, -1), self.v_max.reshape(1, -1))

            # position update + bounds enforcement
            X = X + V
            X = np.clip(X, self.lb, self.ub)

            # evaluate
            fitness = np.asarray(self.func(X))

            # update personal bests
            better = fitness < pbest_f
            if np.any(better):
                pbest[better] = X[better]
                pbest_f[better] = fitness[better]

            # update global best
            cur_best_idx = int(np.argmin(pbest_f))
            if pbest_f[cur_best_idx] < gbest_f:
                gbest_f = float(pbest_f[cur_best_idx])
                gbest = pbest[cur_best_idx].copy()

            if record_history:
                history["best_x"].append(gbest.copy())
                history["best_f"].append(gbest_f)

        return (gbest, gbest_f, history) if record_history else (gbest, gbest_f)

class SOMA:
    """SOMA (All-to-One variant) for minimization."""
    def __init__(self, func: Callable[[np.ndarray], float], bounds: Tuple[np.ndarray, np.ndarray], NP: int = 50, step: float = 0.11,
                 PRT: float = 0.1, path_length: float = 3.0, migrations: int = 200, seed: Optional[int] = None):
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.d = self.lb.size
        self.NP = int(NP)
        self.step = float(step)
        self.PRT = float(PRT)
        self.path_length = float(path_length)
        self.migrations = int(migrations)
        self.rng = np.random.default_rng(seed)

    def _init_population(self):
        """Uniform initial population in bounds: shape (NP, d)."""
        return self.rng.uniform(self.lb, self.ub, size=(self.NP, self.d))

    def _ensure_prt_mask(self):
        """Return a boolean mask of length d with PRT probability. Guarantee at least one (dim) True ."""
        mask = self.rng.random(self.d) < self.PRT
        if not np.any(mask):
            mask[self.rng.integers(0, self.d)] = True
        return mask

    def run(self, record_history: bool = True):
        """Execute SOMA All-to-One."""
        # initialize
        pop = self._init_population()  # (NP, d)
        fitness = np.asarray(self.func(pop))  # (NP,)
        best_idx = int(np.argmin(fitness))
        best_x = pop[best_idx].copy()
        best_f = float(fitness[best_idx])
        history = {"best_x": [best_x.copy()], "best_f": [best_f], "pop": [pop.copy()]} if record_history else None

        steps = np.arange(self.step, self.path_length + 1e-12, self.step)  # e.g., [0.11, 0.22, ...]

        for mig in range(1, self.migrations + 1):
            # Leader (best-so-far)
            leader_idx = int(np.argmin(fitness))
            leader = pop[leader_idx].copy()

            # For each individual (all-to-one) attempt migration toward leader
            for i in range(self.NP):
                if i == leader_idx:
                    continue  # leader does not move

                x = pop[i].copy()
                # build the perturbation mask for this migrant
                prt_mask = self._ensure_prt_mask()  # boolean array length d

                # generate candidate points along the path for this individual
                direction = (leader - x) * prt_mask  # movement only in masked dims

                # Create array of candidate points: (n_steps, d)
                candidates = x.reshape(1, -1) + np.outer(steps, direction)  # steps[:,None] * direction[None,:]
                candidates = np.clip(candidates, self.lb, self.ub)  # Clip to bounds

                # Find best candidate along path
                vals = np.asarray(self.func(candidates))  # shape (n_steps,)
                idx_best = int(np.argmin(vals))
                best_candidate = candidates[idx_best]
                best_candidate_val = float(vals[idx_best])

                # Greedy replacement for the individual if candidate better than current
                if best_candidate_val <= fitness[i]:
                    pop[i] = best_candidate
                    fitness[i] = best_candidate_val
                    # possibly update overall best
                    if best_candidate_val < best_f:
                        best_f = best_candidate_val
                        best_x = best_candidate.copy()

            # Optionally update history
            if record_history:
                history["best_x"].append(best_x.copy())
                history["best_f"].append(best_f)
                history["pop"].append(pop.copy())

        if record_history:
            return best_x, best_f, history
        return best_x, best_f

class ACO:
    """Ant Colony Optimization for TSP."""
    def __init__(self, cities, n_ants: Optional[int] = None, n_iterations: int = 200,
                 alpha: float = 1.0, beta: float = 5.0, rho: float = 0.5, Q: float = 1.0, seed: Optional[int] = None):
        self.cities = cities
        self.n = cities.n
        self.dist = cities.distance_matrix()
        eps = 1e-12
        self.eta = np.zeros_like(self.dist) # distance information (1/distance)
        nz = self.dist > 0
        self.eta[nz] = 1.0 / (self.dist[nz] + eps)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.Q = float(Q)
        self.n_ants = int(n_ants) if n_ants is not None else self.n
        self.n_iterations = int(n_iterations)
        self.rng = np.random.default_rng(seed)
        # pheromone matrix (symmetric) - initialize to small positive constant
        self.tau0 = 1.0
        self.tau = np.full((self.n, self.n), self.tau0, dtype=float) # pheromone levels
        np.fill_diagonal(self.tau, 0.0)

    def _construct_solution(self, start: int):
        """Construct a tour for a single ant using probabilistic selection."""
        tour = [start]
        visited = np.zeros(self.n, dtype=bool)
        visited[start] = True
        for _ in range(self.n - 1):
            current = tour[-1]
            candidates = np.where(~visited)[0]
            # probabilities
            tau_vals = self.tau[current, candidates] ** self.alpha # pheromone influence
            eta_vals = self.eta[current, candidates] ** self.beta # distance influence
            weight = tau_vals * eta_vals
            s = weight.sum()
            probs = weight / s if s > 0 else np.full_like(weight, 1.0 / weight.size)
            # random selection
            next_city = int(self.rng.choice(candidates, p=probs))
            tour.append(next_city)
            visited[next_city] = True
        return tour

    def run(self, record_history: bool = True):
        """Run ACO."""
        best_tour = None
        best_len = float('inf')
        history = {"best_tour": [], "best_len": [], "pop_tours": []} if record_history else None

        for it in range(self.n_iterations):
            # 1) construct tours for all ants (start positions may be random or cyclical)
            tours = []
            lengths = np.zeros(self.n_ants, dtype=float)
            starts = self.rng.integers(0, self.n, size=self.n_ants)
            for k in range(self.n_ants):
                tour = np.asarray(self._construct_solution(int(starts[k])), dtype=int)
                L = self.cities.total_distance(tour)
                tours.append(tour)
                lengths[k] = L
                if L < best_len:
                    best_len = float(L)
                    best_tour = tour.copy()

            # 2) pheromone evaporation
            self.tau *= (1.0 - self.rho)

            # 3) pheromone deposit: for each ant, add Q / length on edges visited
            for tour, L in zip(tours, lengths):
                deposit = self.Q / (L + 1e-12)
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    self.tau[a, b] += deposit
                    self.tau[b, a] += deposit

            if record_history:
                history["best_tour"].append(best_tour.copy())
                history["best_len"].append(best_len)
                history["pop_tours"].append(tours)

        return (best_tour, best_len, history) if record_history else (best_tour, best_len)
    
class FireflyAlgorithm:
    """Firefly Algorithm for minimization."""
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], bounds: Tuple[np.ndarray, np.ndarray], n: int = 20, 
                 alpha: float = 0.3, beta0: float = 1.0, gamma: float = 1.0, max_iter: int = 100, seed: Optional[int] = None):
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.d = self.lb.size
        self.n = int(n)
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.gamma = float(gamma)
        self.max_iter = int(max_iter)
        self.rng = np.random.default_rng(seed)

    def _init_fireflies(self):
        return self.rng.uniform(self.lb, self.ub, size=(self.n, self.d))

    def _attractiveness(self, dist):
        return self.beta0 * np.exp(-self.gamma * dist)

    def run(self, record_history: bool = True):
        # initialize population and evaluate
        X = self._init_fireflies()
        fitness = np.asarray(self.func(X))
        best_idx = int(np.argmin(fitness))
        best_x = X[best_idx].copy()
        best_f = float(fitness[best_idx])
        history = {"best_x":[best_x.copy()], "best_f":[best_f], "pop":[X.copy()]} if record_history else None

        for t in range(1, self.max_iter+1):
            best_idx = int(np.argmin(fitness))
            # best firefly random trial
            x_best = X[best_idx].copy()
            trial = x_best + self.alpha * self.rng.normal(size=self.d)
            trial = np.clip(trial, self.lb, self.ub)
            f_trial = float(self.func(trial))
            if f_trial < fitness[best_idx]:
                X[best_idx] = trial
                fitness[best_idx] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

            # move each firefly i toward all brighter fireflies j
            for i in range(self.n):
                for j in range(self.n):
                    if fitness[j] < fitness[i]:
                        dist = np.sum((X[i] - X[j])**2)
                        beta = self._attractiveness(dist)
                        X[i] = X[i] + beta * (X[j] - X[i]) + self.alpha * self.rng.normal(size=self.d) #* (self.ub - self.lb)
                        X[i] = np.clip(X[i], self.lb, self.ub)
                        # evaluate new position
                        fitness[i] = float(self.func(X[i]))
                        if fitness[i] < best_f:
                            best_f = fitness[i]
                            best_x = X[i].copy()

            if record_history:
                history["best_x"].append(best_x.copy())
                history["best_f"].append(best_f)
                history["pop"].append(X.copy())

        return (best_x, best_f, history) if record_history else (best_x, best_f)
    
class TLBO:
    """Teaching-Learning Based Optimization (TLBO) for minimization."""
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], bounds: Tuple[np.ndarray, np.ndarray],
                 NP: int = 30, Max_OFE: int = 3000, seed: Optional[int] = None):
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        if self.lb.shape != self.ub.shape:
            raise ValueError("Bounds (lb, ub) must have same shape")
        self.d = self.lb.size
        self.NP = int(NP)
        self.Max_OFE = int(Max_OFE)
        self.rng = np.random.default_rng(seed)

    def _init_population(self):
        return self.rng.uniform(self.lb, self.ub, size=(self.NP, self.d))

    def run(self, record_history: bool = True):
        """Execute TLBO."""
        # Initialize population and evaluate (vectorized)
        X = self._init_population()                              # (NP, d)
        fitness = np.asarray(self.func(X))                       # (NP,)
        evals = self.NP                                          # initial evaluations
        best_idx = int(np.argmin(fitness))
        best_x = X[best_idx].copy()
        best_f = float(fitness[best_idx])

        history = {"best_x":[best_x.copy()], "best_f":[best_f], "evals":[evals]} if record_history else None

        # Main loop: alternate Teacher and Learner phases until budget exhausted
        while evals < self.Max_OFE:
            # --- Teacher phase ---
            # compute population mean M (vector)
            M = np.mean(X, axis=0)
            teacher_idx = int(np.argmin(fitness))
            X_teacher = X[teacher_idx].copy()

            # For each learner i: try teacher-updated solution
            for i in range(self.NP):
                if evals >= self.Max_OFE:
                    break
                T = int(self.rng.integers(1, 3))   # 1 or 2
                r = float(self.rng.random())       # uniform [0,1)
                diff = r * (X_teacher - T * M)    # Difference term from teacher phase
                X_new = X[i] + diff
                # enforce bounds
                X_new = np.clip(X_new, self.lb, self.ub)
                # evaluate candidate
                f_new = float(self.func(X_new))
                evals += 1
                # accept if better
                if f_new < fitness[i]:
                    X[i] = X_new
                    fitness[i] = f_new
                    if f_new < best_f:
                        best_f = f_new
                        best_x = X_new.copy()

            # --- Learner phase ---
            # Each learner i interacts with a randomly selected student j != i
            for i in range(self.NP):
                if evals >= self.Max_OFE:
                    break
                # pick random j != i (uniform among indices except i)
                choices = np.delete(np.arange(self.NP), i)
                j = int(self.rng.choice(choices))
                r = float(self.rng.random())
                if fitness[j] < fitness[i]:
                    # move toward better peer
                    X_new = X[i] + r * (X[j] - X[i])
                else:
                    # move away (exploit other direction)
                    X_new = X[i] + r * (X[i] - X[j])
                X_new = np.clip(X_new, self.lb, self.ub)
                f_new = float(self.func(X_new))
                evals += 1
                if f_new < fitness[i]:
                    X[i] = X_new
                    fitness[i] = f_new
                    if f_new < best_f:
                        best_f = f_new
                        best_x = X_new.copy()

            # record snapshot after full generation
            if record_history:
                history["best_x"].append(best_x.copy())
                history["best_f"].append(best_f)
                history["evals"].append(evals)

        return (best_x, best_f, history) if record_history else (best_x, best_f)
    
# --- NSGA2 Helpers: dominance, non-dominated sorting, crowding distance ---
def dominates(a_obj: np.ndarray, b_obj: np.ndarray) -> bool:
    """Return True if a dominates b for minimization problems."""
    return np.all(a_obj <= b_obj) and np.any(a_obj < b_obj)

def fast_non_dominated_sort(objs: np.ndarray) -> List[List[int]]:
    """ Fast non-dominated sorting, takes objs shape (N-individuals, M-objectives), outputs list of fronts (each front is a list of indices)."""
    N = objs.shape[0]
    S = [[] for _ in range(N)]    # S[p] = set of solutions dominated by p
    n = np.zeros(N, dtype=int)   # n[p] = number of solutions that dominate p
    fronts: List[List[int]] = []

    # Build S and n
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts.append([p])

    # Extract subsequent fronts
    i = 0
    while i < len(fronts):
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts

def crowding_distance(objs: np.ndarray, front: List[int]) -> np.ndarray:
    """Crowding distance for solutions in 'front' using their objective vectors in 'objs', returns array with distance for each solution in the same order as 'front'."""
    if len(front) == 0:
        return np.array([], dtype=float)
    M = objs.shape[1]
    distances = np.zeros(len(front), dtype=float)
    for m in range(M):
        vals = objs[front, m]
        order = np.argsort(vals)
        sorted_vals = vals[order]
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        denom = sorted_vals[-1] - sorted_vals[0]
        if denom == 0:
            continue
        for k in range(1, len(front)-1):
            distances[order[k]] += (sorted_vals[k+1] - sorted_vals[k-1]) / denom
    return distances

# --- NSGA2 Genetic operators: SBX crossover and polynomial mutation ---
def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta_c: float = 20.0, rng: Optional[np.random.Generator] = None):
    """Simulated Binary Crossover (SBX) for real-coded parents p1, p2, returns two children."""
    if rng is None:
        rng = np.random.default_rng()
    d = p1.size
    c1 = p1.copy()
    c2 = p2.copy()
    for i in range(d):
        if rng.random() <= 0.5:
            if abs(p1[i] - p2[i]) > 1e-14:
                u = rng.random()
                if u <= 0.5:
                    beta = (2*u)**(1.0/(eta_c+1))
                else:
                    beta = (1.0/(2*(1-u)))**(1.0/(eta_c+1))
                # children
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
            else:
                c1[i] = p1[i]; c2[i] = p2[i]
        else:
            c1[i] = p1[i]; c2[i] = p2[i]
    return c1, c2

def polynomial_mutation(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, eta_m: float = 20.0,
                       pm: Optional[float] = None, rng: Optional[np.random.Generator] = None):
    """Polynomial mutation (Deb and Goyal). pm is per-gene mutation probability (default 1/d)."""
    if rng is None:
        rng = np.random.default_rng()
    y = x.copy()
    d = x.size
    if pm is None:
        pm = 1.0 / d
    for i in range(d):
        if rng.random() < pm:
            xi = x[i]; yl = lb[i]; yu = ub[i]
            if yu <= yl:
                continue
            delta1 = (xi - yl) / (yu - yl)
            delta2 = (yu - xi) / (yu - yl)
            r = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)
            if r < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * r + (1.0 - 2.0*r) * (xy**(eta_m+1))
                delta_q = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0*(1.0 - r) + 2.0*(r - 0.5) * (xy**(eta_m+1))
                delta_q = 1.0 - val**mut_pow
            y[i] = xi + delta_q * (yu - yl)
            # clip safety
            if y[i] < yl: y[i] = yl
            if y[i] > yu: y[i] = yu
    return y

class NSGA2:
    """NSGA-II implementation for continuous multiobjective minimization."""
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], bounds: Tuple[np.ndarray, np.ndarray], pop_size: int = 100, generations: int = 250,
                 eta_c: float = 20.0, eta_m: float = 20.0, crossover_prob: float = 0.9, mutation_prob: Optional[float] = None, seed: Optional[int] = None):
        self.func = func
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.d = self.lb.size
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.pc = float(crossover_prob)
        self.pm = mutation_prob if mutation_prob is not None else 1.0/self.d
        self.rng = np.random.default_rng(seed)

    def _init_pop(self) -> np.ndarray:
        """Initialize population uniformly in bounds."""
        return self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.d))

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate population X using objective function (vectorized)."""
        out = np.asarray(self.func(X))
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out

    def _create_offspring(self, pop_vars: np.ndarray, pop_objs: np.ndarray) -> np.ndarray:
        """Create offspring population using tournament selection, SBX crossover, and polynomial mutation."""
        N = pop_vars.shape[0]
        children: List[np.ndarray] = []

        # compute ranks & crowding distances for tournament selection
        fronts = fast_non_dominated_sort(pop_objs)
        rank = np.empty(N, dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        crowd = np.zeros(N, dtype=float)
        for front in fronts:
            cd = crowding_distance(pop_objs, front)
            crowd[np.array(front)] = cd

        def tournament():
            a, b = self.rng.integers(0, N, size=2)
            if rank[a] < rank[b]:
                return a
            elif rank[b] < rank[a]:
                return b
            else:
                return a if crowd[a] > crowd[b] else b

        # produce N children
        while len(children) < N:
            p1 = pop_vars[tournament()].copy()
            p2 = pop_vars[tournament()].copy()
            if self.rng.random() < self.pc:
                c1, c2 = sbx_crossover(p1, p2, eta_c=self.eta_c, rng=self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = polynomial_mutation(c1, self.lb, self.ub, eta_m=self.eta_m, pm=self.pm, rng=self.rng)
            c2 = polynomial_mutation(c2, self.lb, self.ub, eta_m=self.eta_m, pm=self.pm, rng=self.rng)
            children.append(np.clip(c1, self.lb, self.ub))
            if len(children) < N:
                children.append(np.clip(c2, self.lb, self.ub))

        return np.array(children[:N])

    def run(self, return_history: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        """Run NSGA-II. Returns final population variables, objectives, and optional history dict."""
        pop = self._init_pop()
        pop_objs = self._evaluate(pop)
        history = {"pop": [], "objs": []} if return_history else None

        for gen in range(self.generations):
            offspring = self._create_offspring(pop, pop_objs)
            off_objs = self._evaluate(offspring)
            combined_vars = np.vstack([pop, offspring])
            combined_objs = np.vstack([pop_objs, off_objs])
            fronts = fast_non_dominated_sort(combined_objs)

            new_pop_vars: List[np.ndarray] = []
            new_pop_objs: List[np.ndarray] = []
            for front in fronts:
                if len(new_pop_vars) + len(front) <= self.pop_size:
                    new_pop_vars.extend(combined_vars[front].tolist())
                    new_pop_objs.extend(combined_objs[front].tolist())
                else:
                    remaining = self.pop_size - len(new_pop_vars)
                    cd = crowding_distance(combined_objs, front)
                    order = np.argsort(-cd)  # descending crowding distance
                    chosen = [front[i] for i in order[:remaining]]
                    new_pop_vars.extend(combined_vars[chosen].tolist())
                    new_pop_objs.extend(combined_objs[chosen].tolist())
                    break

            pop = np.array(new_pop_vars)
            pop_objs = np.array(new_pop_objs)
            if return_history:
                history["pop"].append(pop.copy())
                history["objs"].append(pop_objs.copy())

        return pop, pop_objs, history