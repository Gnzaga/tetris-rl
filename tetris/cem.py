"""Noisy cross-entropy method over the 8 BCTS feature weights (PLAN.md §7).

The cross-entropy method (CEM) maintains a diagonal Gaussian ``N(mu, diag(var))``
over the 8-dimensional weight vector of a :class:`~tetris.agents.LinearAgent`.
Each generation:

  1. sample a ``population`` of weight vectors from the current Gaussian;
  2. score each by *fitness* — mean raw lines cleared over ``fitness_games``
     games, all played on the **same** per-generation seeds (common random
     numbers), each capped at ``fitness_max_pieces`` (CEM ignores shaping);
  3. keep the top ``elites`` and refit ``mu``/``var`` to them;
  4. **noisy CEM** (Szita & Lőrincz 2006): add ``max(0, 5 - gen/2)`` to every
     variance component after the refit, delaying variance collapse so the
     search keeps exploring in early generations.

Fitness evaluation is embarrassingly parallel; the heavy lifting runs in a
``multiprocessing`` pool via the module-level, spawn-safe worker
:func:`_play_worker` (weights + seed passed by value, nothing captured). The
same worker powers the parallel greedy evaluations used for periodic monitoring
and the final 30-game report.

This module is pure algorithm + parallel plumbing; ``scripts/train_cem.py`` owns
the run directory, logging, the rich table, and CLI. Keeping the workers here
(and importing only light deps: numpy, engine, agents) means spawned children
never load torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import LinearAgent
from .evaluation import EvalResult, play_game

WEIGHT_DIM = 8

# Noisy-CEM variance floor schedule: variance component added after each refit
# is ``max(0, NOISE_BASE - gen / NOISE_DECAY)`` (PLAN.md §7 -> max(0, 5 - gen/2)).
NOISE_BASE = 5.0
NOISE_DECAY = 2.0

# --- adaptive fitness cap -------------------------------------------------
# A 10-wide board and 4-cell pieces bound lines at 0.4 * pieces, so a fitness
# capped at P pieces saturates at 0.4 * P lines. Once every decent candidate
# hits that ceiling, elite selection is arbitrary among cap-hitters and CEM
# stops learning (observed: best fitness pinned at 319 of a 320 max with the
# spec's 800-piece cap). Remedy: whenever the elite mean fitness reaches
# CAP_SATURATION_FRACTION of the ceiling, multiply the cap by CAP_GROWTH_FACTOR
# for subsequent generations (deterministic, recorded per generation).
LINES_PER_PIECE_MAX = 0.4
CAP_SATURATION_FRACTION = 0.9
CAP_GROWTH_FACTOR = 4


def noise_variance(gen: int) -> float:
    """Extra variance added to every component after the gen-``gen`` refit."""
    return max(0.0, NOISE_BASE - gen / NOISE_DECAY)


def max_attainable_lines(cap_pieces: int) -> float:
    """Upper bound on lines clearable within a ``cap_pieces``-piece game."""
    return LINES_PER_PIECE_MAX * cap_pieces


def cap_saturated(elite_mean_fitness: float, cap_pieces: int) -> bool:
    """True when the elite mean is within 10% of the cap's line ceiling."""
    return elite_mean_fitness >= CAP_SATURATION_FRACTION * max_attainable_lines(cap_pieces)


def grow_cap(cap_pieces: int, cap_max: int) -> int:
    """Next fitness cap after a saturated generation (clamped to ``cap_max``)."""
    return min(cap_pieces * CAP_GROWTH_FACTOR, cap_max)


# Ceiling-convergence early stop: once the cap is pinned at its maximum and the
# elites saturate its line ceiling for several consecutive generations, further
# generations are signal-free (elites indistinguishable at the measurement
# horizon; exploration noise anneals to 0 by gen 10), so the loop terminates.
CEILING_STOP_MIN_GEN = 4
CEILING_STOP_WINDOW = 3
CEILING_STOP_FRACTION = 0.99


def converged_at_ceiling(
    gen: int, cap_pieces: int, cap_max: int, recent_elite_means: list[float]
) -> bool:
    """True when the last ``CEILING_STOP_WINDOW`` consecutive generations *at the
    maximum cap* all had elite mean fitness >= ``CEILING_STOP_FRACTION`` of the
    max attainable lines, and ``gen >= CEILING_STOP_MIN_GEN``.

    ``recent_elite_means`` must contain only elite means of consecutive
    max-cap generations ending at ``gen`` (the caller resets it whenever a
    generation runs below the max cap).
    """
    if gen < CEILING_STOP_MIN_GEN or cap_pieces < cap_max:
        return False
    if len(recent_elite_means) < CEILING_STOP_WINDOW:
        return False
    threshold = CEILING_STOP_FRACTION * max_attainable_lines(cap_pieces)
    return all(m >= threshold for m in recent_elite_means[-CEILING_STOP_WINDOW:])


def should_stop_early(
    gen: int, cap_pieces: int, cap_max: int, saturated: bool, var: np.ndarray,
    std_threshold: float = 0.5,
) -> bool:
    """Authorized early-stop test, evaluated before starting generation ``gen``.

    All of: the cap has grown to its maximum, fitness is NOT saturated at that
    cap (selection pressure is real), the noisy-CEM exploration phase is over
    (``noise_variance(gen) == 0``), and the sampling distribution has collapsed
    (every component std below ``std_threshold``, tiny next to weights of order
    ``sigma_init``) — at which point further generations resample essentially
    the same weight vector and cannot make progress.
    """
    return (
        cap_pieces >= cap_max
        and not saturated
        and noise_variance(gen) == 0.0
        and float(np.sqrt(var.max())) < std_threshold
    )


def fitness_seeds(seed_base: int, gen: int, games: int) -> list[int]:
    """Common random seeds shared by the whole population in generation ``gen``.

    Distinct, non-overlapping across generations so a lucky/unlucky seed does not
    bias the same generation twice, while every individual *within* a generation
    is scored on the identical seed set (common random numbers — the variance
    reduction that makes elite selection meaningful).
    """
    start = seed_base + gen * games
    return [start + i for i in range(games)]


def select_elites(fitnesses: np.ndarray, n_elites: int) -> np.ndarray:
    """Indices of the ``n_elites`` highest fitnesses, best first (ties: lower index)."""
    order = np.argsort(fitnesses, kind="stable")[::-1]
    return order[:n_elites]


def refit(elite_weights: np.ndarray, gen: int) -> tuple[np.ndarray, np.ndarray]:
    """New ``(mu, var)`` from the elite set, with the noisy-CEM variance floor."""
    mu = elite_weights.mean(axis=0)
    var = elite_weights.var(axis=0) + noise_variance(gen)
    return mu, var


# --- multiprocessing workers (spawn-safe: module-level, no captured state) ----


def _play_worker(args):
    """Play one game and return ``(lines, pieces, moves)``.

    ``args = (weights, seed, max_pieces, record)``. Weights are passed by value
    (a plain list/ndarray, picklable) and a fresh ``LinearAgent`` is built in the
    child, so the worker holds no shared state across calls.
    """
    weights, seed, max_pieces, record = args
    agent = LinearAgent(weights)
    return play_game(agent, int(seed), int(max_pieces), record=record)


def _map(pool, tasks):
    """``pool.map`` when a pool is given, else a serial fallback (tests/smoke)."""
    if pool is None:
        return [_play_worker(t) for t in tasks]
    return pool.map(_play_worker, tasks)


def evaluate_population(
    pool, population: np.ndarray, seeds: list[int], max_pieces: int
) -> tuple[np.ndarray, int]:
    """Fitness (mean lines over ``seeds``) for every individual + total pieces.

    Flattened game-level parallelism (``population * len(seeds)`` tasks) load
    balances better than one task per individual. Returns
    ``(fitnesses[pop], pieces_simulated)`` where ``pieces_simulated`` is the true
    number of pieces played this call (used for the cumulative ``pieces_trained``
    step counter).
    """
    games = len(seeds)
    tasks = [
        (w, s, max_pieces, False)
        for w in population
        for s in seeds
    ]
    results = _map(pool, tasks)

    n = len(population)
    fitnesses = np.empty(n, dtype=np.float64)
    pieces_total = 0
    for i in range(n):
        lines_sum = 0
        for j in range(games):
            lines, pieces, _ = results[i * games + j]
            lines_sum += lines
            pieces_total += pieces
        fitnesses[i] = lines_sum / games
    return fitnesses, pieces_total


def parallel_eval(
    pool, weights, seeds: list[int], max_pieces: int, record: bool = False
) -> EvalResult:
    """Greedy evaluation of a single weight vector across ``seeds``, in parallel.

    Game-level parallelism (one task per seed). Returns the same
    :class:`~tetris.evaluation.EvalResult` the serial ``evaluate`` helper does, so
    downstream code (median/mean/p10, best_index, replay of best game) is shared.
    """
    tasks = [(weights, s, max_pieces, record) for s in seeds]
    results = _map(pool, tasks)
    lines = [int(r[0]) for r in results]
    pieces = [int(r[1]) for r in results]
    moves = [r[2] for r in results]
    return EvalResult(list(seeds), lines, pieces, moves)


# --- the CEM state machine ----------------------------------------------------


@dataclass
class GenerationResult:
    """Everything ``scripts/train_cem.py`` needs to log one generation."""

    gen: int
    mu: np.ndarray
    var: np.ndarray
    population: np.ndarray
    fitnesses: np.ndarray
    elite_indices: np.ndarray
    fit_seeds: list[int]
    pieces_this_gen: int
    fitness_cap: int  # piece cap used to score THIS generation
    saturated: bool  # elite mean hit the cap's line ceiling
    next_fitness_cap: int  # cap the NEXT generation will use

    @property
    def best_fitness(self) -> float:
        return float(self.fitnesses.max())

    @property
    def mean_fitness(self) -> float:
        return float(self.fitnesses.mean())

    @property
    def elite_mean_fitness(self) -> float:
        return float(self.fitnesses[self.elite_indices].mean())


class CEM:
    """Diagonal-Gaussian noisy CEM over the 8 BCTS weights.

    Sampling is driven by a single seeded ``numpy.random.Generator`` so an entire
    run (population draws included) is reproducible from ``seed`` alone. Variance
    is stored (not std); the Gaussian samples with ``std = sqrt(var)``. Initial
    state is ``mu = 0``, ``var = sigma_init**2`` per dimension.
    """

    def __init__(
        self,
        seed: int,
        population: int,
        elites: int,
        sigma_init: float,
        fitness_games: int,
        fitness_max_pieces: int,
        fitness_seed_base: int,
        fitness_cap_max: int | None = None,
        dim: int = WEIGHT_DIM,
    ):
        self.rng = np.random.default_rng(seed)
        self.population = population
        self.elites = elites
        self.dim = dim
        self.fitness_games = fitness_games
        self.fitness_seed_base = fitness_seed_base

        # Adaptive fitness cap: starts at the spec's fitness_max_pieces and
        # grows x4 whenever a generation saturates it (see module docstring).
        self.fitness_cap = int(fitness_max_pieces)
        self.fitness_cap_max = int(
            fitness_cap_max if fitness_cap_max is not None else fitness_max_pieces
        )

        self.mu = np.zeros(dim, dtype=np.float64)
        self.var = np.full(dim, float(sigma_init) ** 2, dtype=np.float64)

    def sample(self) -> np.ndarray:
        """Draw ``population`` weight vectors from the current Gaussian."""
        return self.rng.normal(
            self.mu, np.sqrt(self.var), size=(self.population, self.dim)
        )

    def run_generation(self, gen: int, pool=None) -> GenerationResult:
        """Sample, score on common seeds, select elites, refit ``mu``/``var``.

        Scores this generation at the current adaptive ``fitness_cap``, then —
        if the elite mean saturated the cap's line ceiling — grows the cap
        (x``CAP_GROWTH_FACTOR``, clamped to ``fitness_cap_max``) for subsequent
        generations.
        """
        cap = self.fitness_cap
        seeds = fitness_seeds(self.fitness_seed_base, gen, self.fitness_games)
        population = self.sample()
        fitnesses, pieces = evaluate_population(pool, population, seeds, cap)
        elite_idx = select_elites(fitnesses, self.elites)
        self.mu, self.var = refit(population[elite_idx], gen)

        elite_mean = float(fitnesses[elite_idx].mean())
        saturated = cap_saturated(elite_mean, cap)
        if saturated:
            self.fitness_cap = grow_cap(cap, self.fitness_cap_max)

        return GenerationResult(
            gen=gen,
            mu=self.mu.copy(),
            var=self.var.copy(),
            population=population,
            fitnesses=fitnesses,
            elite_indices=elite_idx,
            fit_seeds=seeds,
            pieces_this_gen=int(pieces),
            fitness_cap=cap,
            saturated=saturated,
            next_fitness_cap=self.fitness_cap,
        )
