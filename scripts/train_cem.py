"""Cross-entropy-method trainer over the 8 BCTS weights (PLAN.md §6/§7).

Phase 3 status: this is a MINIMAL, correct-shaped CEM skeleton whose only job is
to exercise every ``tetris/runio.py`` contract surface end-to-end (config.json,
metrics.jsonl, tb/, checkpoints/, replays/, runs/index.json, the rich live
table, TensorBoard mirroring). The loop body — sampling, fitness, elite refit —
is intentionally simple and single-process. Phase 4 replaces the internals of
:func:`run_cem` (population 100, noisy CEM per Szita & Lőrincz, multiprocessing
fitness, 40 generations, final 30-game eval) while keeping the runio surface and
the ``--smoke`` code path unchanged.

Overwrite policy: a run reuses ``runs/<run_name>/`` and is overwritten fresh on
each invocation (RunWriter(overwrite=True)); pick a new --run-name to keep old
artifacts. ``--smoke`` forces run name ``cem_smoke`` and a tiny budget that
finishes in well under two minutes.
"""

import _pathshim  # noqa: F401
import argparse
import time

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table

from tetris.agents import LinearAgent
from tetris.evaluation import evaluate
from tetris.runio import RunWriter

WEIGHT_DIM = 8


def _build_config(args: argparse.Namespace) -> dict:
    """Resolve CLI args into a flat config dict (persisted to config.json)."""
    eval_seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.eval_games))
    return {
        "smoke": args.smoke,
        "seed": args.seed,
        "population": args.population,
        "elites": args.elites,
        "generations": args.generations,
        "sigma_init": args.sigma_init,
        "weight_dim": WEIGHT_DIM,
        "fitness_games": args.fitness_games,
        "fitness_max_pieces": args.fitness_max_pieces,
        "fitness_seed_base": args.fitness_seed_base,
        "eval_games": args.eval_games,
        "eval_max_pieces": args.eval_max_pieces,
        "eval_seeds": eval_seeds,
    }


def _fitness(weights: np.ndarray, seeds: list[int], max_pieces: int) -> float:
    """Mean lines of a LinearAgent(weights) over `seeds` (CEM ignores shaping)."""
    result = evaluate(lambda _s: LinearAgent(weights), seeds, max_pieces)
    return result.mean_lines


def _render_table(rows: list[dict]) -> Table:
    table = Table(title="CEM training (Phase 3 skeleton)")
    for col in ("gen", "best fitness", "eval median", "eval mean", "elapsed s", "ETA s"):
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(
            str(r["gen"]),
            f"{r['best_fitness']:.1f}",
            f"{r['eval_median']:.1f}",
            f"{r['eval_mean']:.1f}",
            f"{r['elapsed']:.1f}",
            "-" if r["eta"] is None else f"{r['eta']:.1f}",
        )
    return table


def run_cem(run: RunWriter, cfg: dict, console: Console) -> dict:
    """Placeholder CEM loop. Phase 4 replaces this body.

    Samples weight vectors from N(mu, sigma^2 I), scores them by mean lines on
    shared per-generation fitness seeds, refits mu/sigma to the elites, then
    greedily evaluates mu on the fixed eval seeds and records metrics + a replay
    of the best eval game. Returns the final resolved weights + eval stats.
    """
    rng = np.random.default_rng(cfg["seed"])
    mu = np.zeros(WEIGHT_DIM, dtype=np.float64)
    sigma = np.full(WEIGHT_DIM, float(cfg["sigma_init"]), dtype=np.float64)

    eval_seeds = cfg["eval_seeds"]
    table_rows: list[dict] = []
    gen_times: list[float] = []
    pieces_trained = 0
    last_eval = None

    with Live(_render_table(table_rows), console=console, refresh_per_second=4) as live:
        for gen in range(cfg["generations"]):
            gen_start = time.time()
            fit_seeds = [
                cfg["fitness_seed_base"] + gen * cfg["fitness_games"] + i
                for i in range(cfg["fitness_games"])
            ]

            population = rng.normal(
                mu, sigma, size=(cfg["population"], WEIGHT_DIM)
            )
            fitnesses = np.array(
                [_fitness(w, fit_seeds, cfg["fitness_max_pieces"]) for w in population]
            )
            pieces_trained += cfg["population"] * cfg["fitness_games"] * cfg[
                "fitness_max_pieces"
            ]

            elite_idx = np.argsort(fitnesses)[-cfg["elites"]:]
            elites = population[elite_idx]
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 1e-6  # Phase 4: add noisy-CEM variance floor

            # Greedy eval of the refit mean on fixed seeds; record best replay.
            ev = evaluate(
                lambda _s: LinearAgent(mu),
                eval_seeds,
                cfg["eval_max_pieces"],
                record=True,
            )
            last_eval = ev
            best = ev.best_index

            gen_dt = time.time() - gen_start
            gen_times.append(gen_dt)
            pps = (
                cfg["population"] * cfg["fitness_games"] * cfg["fitness_max_pieces"]
            ) / max(gen_dt, 1e-9)

            run.log(
                phase="cem",
                pieces_trained=pieces_trained,
                loss=float(-fitnesses.max()),  # negative best fitness as a "loss"
                eval_median_lines=ev.median_lines,
                eval_mean_lines=ev.mean_lines,
                eval_p10_lines=ev.p10_lines,
                eval_pieces_per_game=ev.mean_pieces,
                pps_train=pps,
            )
            run.save_json_checkpoint(
                f"cem_gen_{gen}",
                {
                    "generation": gen,
                    "weights": mu.tolist(),
                    "sigma": sigma.tolist(),
                    "best_fitness": float(fitnesses.max()),
                    "eval": {
                        "median_lines": ev.median_lines,
                        "mean_lines": ev.mean_lines,
                        "p10_lines": ev.p10_lines,
                    },
                },
            )
            run.save_replay(
                seed=ev.seeds[best],
                moves=ev.moves[best] or [],
                final={"lines": ev.lines[best], "pieces": ev.pieces[best]},
                pieces_trained=pieces_trained,
                median_lines=ev.median_lines,
            )

            remaining = cfg["generations"] - gen - 1
            eta = (sum(gen_times) / len(gen_times)) * remaining if remaining else None
            table_rows.append(
                {
                    "gen": gen,
                    "best_fitness": float(fitnesses.max()),
                    "eval_median": ev.median_lines,
                    "eval_mean": ev.mean_lines,
                    "elapsed": time.time() - run.start_time,
                    "eta": eta,
                }
            )
            live.update(_render_table(table_rows))

    return {
        "weights": mu.tolist(),
        "eval": None
        if last_eval is None
        else {
            "median_lines": last_eval.median_lines,
            "mean_lines": last_eval.mean_lines,
            "p10_lines": last_eval.p10_lines,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CEM trainer over the 8 BCTS weights")
    ap.add_argument("--run-name", default="cem_v1")
    ap.add_argument("--smoke", action="store_true", help="tiny budget, run name cem_smoke")
    ap.add_argument("--seed", type=int, default=0)
    # Full-run defaults follow PLAN.md §7; Phase 4 wires the real loop to them.
    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--elites", type=int, default=10)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--sigma-init", type=float, default=10.0)
    ap.add_argument("--fitness-games", type=int, default=3)
    ap.add_argument("--fitness-max-pieces", type=int, default=800)
    ap.add_argument("--fitness-seed-base", type=int, default=10_000)
    ap.add_argument("--eval-games", type=int, default=10)
    ap.add_argument("--eval-max-pieces", type=int, default=2_000)
    ap.add_argument("--eval-seed-base", type=int, default=100)
    args = ap.parse_args(argv)

    if args.smoke:
        args.run_name = "cem_smoke"
        args.population = 8
        args.elites = 3
        args.generations = 3
        args.fitness_games = 1
        args.fitness_max_pieces = 40
        args.eval_games = 2
        args.eval_max_pieces = 150
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _build_config(args)
    console = Console()
    with RunWriter(args.run_name, cfg, phase="cem") as run:
        final = run_cem(run, cfg, console)
        run.save_json_checkpoint("cem_final", final)
    console.print(f"[green]run complete[/green] -> {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
