"""Fast unit tests for CEM mechanics (PLAN.md §7).

These cover the deterministic algorithm surface — elite selection, the noisy-CEM
variance schedule, common-seeds-within-a-generation, and seeded reproducibility
of a single tiny generation — without any long training run. Fitness evaluation
here runs serially (``pool=None``) on a tiny piece budget.
"""

import numpy as np

from tetris.cem import (
    CEM,
    WEIGHT_DIM,
    cap_saturated,
    converged_at_ceiling,
    fitness_seeds,
    grow_cap,
    max_attainable_lines,
    noise_variance,
    refit,
    select_elites,
    should_stop_early,
)


def test_noise_variance_schedule():
    # max(0, 5 - gen/2): 5, 4.5, 4, ... hits 0 at gen 10 and stays there.
    assert noise_variance(0) == 5.0
    assert noise_variance(1) == 4.5
    assert noise_variance(2) == 4.0
    assert noise_variance(8) == 1.0
    assert noise_variance(10) == 0.0
    assert noise_variance(20) == 0.0
    assert noise_variance(39) == 0.0


def test_select_elites_picks_top_k_best_first():
    fitnesses = np.array([3.0, 10.0, 1.0, 7.0, 5.0])
    elite = select_elites(fitnesses, 3)
    # best-first: 10.0 (idx1), 7.0 (idx3), 5.0 (idx4)
    assert list(elite) == [1, 3, 4]


def test_select_elites_ties_and_excludes_worst():
    fitnesses = np.array([5.0, 5.0, 5.0, 1.0])
    elite = select_elites(fitnesses, 2)
    # Order among equal fitnesses is unspecified, but the elite set must be two
    # of the three tied maxima and never the strictly-worst individual (idx 3).
    assert len(elite) == 2
    assert set(elite).issubset({0, 1, 2})


def test_refit_adds_noise_floor_to_variance():
    # Two elite vectors; per-component mean and (population) variance are exact.
    elites = np.array([[0.0, 2.0], [4.0, 6.0]])
    mu, var = refit(elites, gen=0)
    assert np.allclose(mu, [2.0, 4.0])
    # var of {0,4} = 4, of {2,6} = 4; plus noise_variance(0)=5 -> 9.
    assert np.allclose(var, [9.0, 9.0])
    # At gen>=10 the floor is 0 so variance is the raw elite variance.
    _, var10 = refit(elites, gen=10)
    assert np.allclose(var10, [4.0, 4.0])


def test_fitness_seeds_common_within_generation_distinct_across():
    base, games = 10_000, 3
    g0 = fitness_seeds(base, 0, games)
    g1 = fitness_seeds(base, 1, games)
    assert g0 == [10_000, 10_001, 10_002]
    assert g1 == [10_003, 10_004, 10_005]
    # Non-overlapping across generations.
    assert set(g0).isdisjoint(g1)
    # The property that matters: within a generation there is exactly one seed
    # set, so every individual is scored on identical seeds (common random nums).
    assert len(g0) == games


def test_common_seeds_used_for_whole_population(monkeypatch):
    """The seeds passed to fitness eval must be identical for all individuals."""
    import tetris.cem as cem_mod

    captured = {}

    def fake_eval(pool, population, seeds, max_pieces):
        captured["seeds"] = list(seeds)
        captured["pop_size"] = len(population)
        return np.arange(len(population), dtype=float), 123

    monkeypatch.setattr(cem_mod, "evaluate_population", fake_eval)
    trainer = CEM(
        seed=0,
        population=6,
        elites=2,
        sigma_init=10.0,
        fitness_games=3,
        fitness_max_pieces=40,
        fitness_seed_base=10_000,
    )
    gr = trainer.run_generation(2, pool=None)
    # One seed set of size fitness_games, shared by the whole population.
    assert captured["seeds"] == fitness_seeds(10_000, 2, 3)
    assert captured["pop_size"] == 6
    assert gr.pieces_this_gen == 123


def test_initial_gaussian_state():
    trainer = CEM(
        seed=0,
        population=4,
        elites=2,
        sigma_init=10.0,
        fitness_games=1,
        fitness_max_pieces=10,
        fitness_seed_base=0,
    )
    assert np.allclose(trainer.mu, np.zeros(WEIGHT_DIM))
    assert np.allclose(trainer.var, np.full(WEIGHT_DIM, 100.0))  # sigma_init**2


def _tiny_trainer(seed):
    return CEM(
        seed=seed,
        population=6,
        elites=2,
        sigma_init=10.0,
        fitness_games=1,
        fitness_max_pieces=30,
        fitness_seed_base=0,
    )


def test_seeded_reproducibility_of_one_generation():
    """Same seed -> identical population sample, fitnesses, and refit mu/var."""
    a = _tiny_trainer(42)
    b = _tiny_trainer(42)
    ga = a.run_generation(0, pool=None)
    gb = b.run_generation(0, pool=None)
    assert np.array_equal(ga.population, gb.population)
    assert np.array_equal(ga.fitnesses, gb.fitnesses)
    assert np.array_equal(ga.mu, gb.mu)
    assert np.array_equal(ga.var, gb.var)
    assert ga.pieces_this_gen == gb.pieces_this_gen


def test_different_seed_gives_different_sample():
    ga = _tiny_trainer(1).run_generation(0, pool=None)
    gb = _tiny_trainer(2).run_generation(0, pool=None)
    assert not np.array_equal(ga.population, gb.population)


def test_adaptive_cap_saturation_trigger():
    # 800-piece cap -> line ceiling 320 -> saturation threshold 0.9*320 = 288.
    assert max_attainable_lines(800) == 320.0
    assert cap_saturated(288.0, 800)
    assert cap_saturated(319.0, 800)
    assert not cap_saturated(287.9, 800)


def test_grow_cap_quadruples_and_clamps():
    assert grow_cap(800, 25_600) == 3_200
    assert grow_cap(3_200, 25_600) == 12_800
    assert grow_cap(12_800, 25_600) == 25_600  # 51,200 clamped
    assert grow_cap(25_600, 25_600) == 25_600


def test_run_generation_grows_cap_when_saturated(monkeypatch):
    """A saturated generation raises the cap used by the NEXT generation."""
    import tetris.cem as cem_mod

    caps_used = []

    def fake_eval(pool, population, seeds, max_pieces):
        caps_used.append(max_pieces)
        # Everyone at the current cap's line ceiling -> elites saturate it.
        ceiling = cem_mod.max_attainable_lines(max_pieces)
        return np.full(len(population), ceiling), 1

    monkeypatch.setattr(cem_mod, "evaluate_population", fake_eval)
    trainer = CEM(
        seed=0,
        population=4,
        elites=2,
        sigma_init=10.0,
        fitness_games=1,
        fitness_max_pieces=800,
        fitness_seed_base=0,
        fitness_cap_max=25_600,
    )
    results = [trainer.run_generation(g, pool=None) for g in range(4)]
    assert caps_used == [800, 3_200, 12_800, 25_600]
    assert [r.saturated for r in results] == [True] * 4
    assert results[0].fitness_cap == 800 and results[0].next_fitness_cap == 3_200
    assert results[3].next_fitness_cap == 25_600  # clamped at the max


def test_run_generation_keeps_cap_when_unsaturated(monkeypatch):
    import tetris.cem as cem_mod

    def fake_eval(pool, population, seeds, max_pieces):
        return np.full(len(population), 10.0), 1  # far below the ceiling

    monkeypatch.setattr(cem_mod, "evaluate_population", fake_eval)
    trainer = CEM(
        seed=0,
        population=4,
        elites=2,
        sigma_init=10.0,
        fitness_games=1,
        fitness_max_pieces=800,
        fitness_seed_base=0,
        fitness_cap_max=25_600,
    )
    gr = trainer.run_generation(0, pool=None)
    assert not gr.saturated
    assert gr.next_fitness_cap == 800
    assert trainer.fitness_cap == 800


def test_should_stop_early_requires_all_conditions():
    collapsed = np.full(WEIGHT_DIM, 0.01)  # std 0.1 < 0.5
    wide = np.full(WEIGHT_DIM, 4.0)  # std 2.0 >= 0.5
    ok = dict(gen=20, cap_pieces=25_600, cap_max=25_600, saturated=False, var=collapsed)
    assert should_stop_early(**ok)
    # Cap not yet at max.
    assert not should_stop_early(**{**ok, "cap_pieces": 12_800})
    # Still saturated at the max cap.
    assert not should_stop_early(**{**ok, "saturated": True})
    # Exploration noise still active (noise_variance > 0 before gen 10).
    assert not should_stop_early(**{**ok, "gen": 9})
    # Sampling distribution not collapsed.
    assert not should_stop_early(**{**ok, "var": wide})


def test_converged_at_ceiling_rule():
    cap = 25_600
    ceiling = 0.4 * cap  # 10,240
    hi = 0.995 * ceiling  # above the 99% threshold
    lo = 0.90 * ceiling  # below it
    # Fires: gen >= 4, cap at max, 3 consecutive max-cap gens >= 99% of ceiling.
    assert converged_at_ceiling(6, cap, cap, [hi, hi, hi])
    # Too early in the run.
    assert not converged_at_ceiling(3, cap, cap, [hi, hi, hi])
    # Cap not yet at its maximum.
    assert not converged_at_ceiling(6, 12_800, cap, [hi, hi, hi])
    # Not enough consecutive max-cap generations.
    assert not converged_at_ceiling(6, cap, cap, [hi, hi])
    # One of the last three dipped below the threshold.
    assert not converged_at_ceiling(6, cap, cap, [hi, lo, hi])
    # Only the LAST window counts: an old dip does not block the stop.
    assert converged_at_ceiling(8, cap, cap, [lo, hi, hi, hi])
    # Exactly at the threshold counts (>=).
    assert converged_at_ceiling(6, cap, cap, [0.99 * ceiling] * 3)


def test_pieces_this_gen_matches_actual_simulation():
    """pieces_this_gen equals the real pieces played, not the piece cap."""
    trainer = _tiny_trainer(7)
    gr = trainer.run_generation(0, pool=None)
    # 6 individuals * 1 game each, cap 30 -> at most 180 pieces, at least 6.
    assert 0 < gr.pieces_this_gen <= 6 * 30
