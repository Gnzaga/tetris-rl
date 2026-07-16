"""Noisy-CEM trainer over the 8 BCTS feature weights (PLAN.md §7).

Cross-entropy method (Szita & Lőrincz noisy variant) tuning a
:class:`~tetris.agents.LinearAgent`. The algorithm and its spawn-safe
multiprocessing workers live in :mod:`tetris.cem`; this script owns the run
directory contract (``tetris.runio.RunWriter``), the rich live table, periodic
greedy evaluation for observability, and the final 30-game report.

Per PLAN.md §7 the full-run defaults are: population 100, elites 10, 40
generations, init ``mu = 0`` / ``sigma = 10`` per dim, fitness = mean raw lines
over 3 games on common per-generation seeds, noisy-CEM variance floor
``max(0, 5 - gen/2)``; then evaluate the final mean weights on 30 games
(cap 200,000 pieces) into ``checkpoints/cem_final.json``.

Two controller-approved deviations from the §7 *defaults* (the gates are
unchanged; §2 rules untouched):

* **Adaptive fitness cap.** The spec's fixed 800-piece fitness cap saturates at
  320 lines; a first full run showed best fitness pinned at ~319 from gen 1 —
  elite selection had become arbitrary among cap-hitters. The cap now starts at
  800 and multiplies x4 whenever the elite mean fitness reaches 90% of the
  cap's line ceiling (0.4 lines/piece), clamped to ``--fitness-cap-max``
  (default 25,600 so one generation stays ~<=10 min on 10+ cores). The cap used
  by each generation is recorded in its checkpoint and shown in the live table.
* **Convergence early-stops.** (a) Ceiling convergence: once the cap is pinned
  at its maximum and 3 consecutive generations' elite mean fitness reaches 99%
  of that cap's line ceiling, further generations are signal-free (elites
  indistinguishable at the measurement horizon; exploration noise anneals to 0
  by gen 10) and the loop terminates. (b) Distribution collapse: cap at max,
  fitness unsaturated there, noise phase over, and every component std < 0.5 —
  further generations resample the same vector. Stop reason is recorded in
  cem_final.json and printed.

After the final 30-game 200k-cap eval, the Dellacherie hand weights are scored
on the SAME seeds (gate 3). If BOTH agents cap all 30 games at 200k the
comparison is censored, so both are re-run on the same seeds with a
``--uncensored-max-pieces`` cap (default 2,000,000) and the uncensored means
decide ``beats_dellacherie``. Both comparisons are recorded in cem_final.json.

``pieces_trained`` logged to metrics.jsonl / TensorBoard is the **cumulative
number of pieces actually simulated during fitness evaluation** — the training
work done so far. It is strictly increasing (every generation simulates > 0
pieces), which TensorBoard requires of its step axis. Periodic greedy-eval
pieces are observability overhead and deliberately excluded from this counter.

Overwrite policy: a run reuses ``runs/<run_name>/`` and is overwritten fresh on
each invocation. ``--smoke`` forces run name ``cem_smoke`` and a tiny budget
that finishes well under two minutes.
"""

import _pathshim  # noqa: F401
import argparse
import multiprocessing as mp
import os
import time

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table

from tetris import cem
from tetris.agents import DELLACHERIE_WEIGHTS
from tetris.cem import CEM, parallel_eval
from tetris.runio import RunWriter

WEIGHT_DIM = cem.WEIGHT_DIM


def _build_config(args: argparse.Namespace) -> dict:
    """Resolve CLI args into a flat config dict (persisted to config.json)."""
    final_eval_seeds = list(
        range(args.final_eval_seed_base, args.final_eval_seed_base + args.final_eval_games)
    )
    eval_seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.eval_games))
    return {
        "algorithm": "noisy-cem",
        "smoke": args.smoke,
        "seed": args.seed,
        "num_workers": args.workers,
        "population": args.population,
        "elites": args.elites,
        "generations": args.generations,
        "sigma_init": args.sigma_init,
        "weight_dim": WEIGHT_DIM,
        "noise_base": cem.NOISE_BASE,
        "noise_decay": cem.NOISE_DECAY,
        "fitness_games": args.fitness_games,
        "fitness_max_pieces": args.fitness_max_pieces,
        "fitness_cap_max": args.fitness_cap_max,
        "cap_growth_factor": cem.CAP_GROWTH_FACTOR,
        "cap_saturation_fraction": cem.CAP_SATURATION_FRACTION,
        "lines_per_piece_max": cem.LINES_PER_PIECE_MAX,
        "early_stop_std_threshold": 0.5,
        "ceiling_stop_min_gen": cem.CEILING_STOP_MIN_GEN,
        "ceiling_stop_window": cem.CEILING_STOP_WINDOW,
        "ceiling_stop_fraction": cem.CEILING_STOP_FRACTION,
        "fitness_seed_base": args.fitness_seed_base,
        "eval_games": args.eval_games,
        "eval_max_pieces": args.eval_max_pieces,
        "eval_seed_base": args.eval_seed_base,
        "eval_seeds": eval_seeds,
        "final_eval_games": args.final_eval_games,
        "final_eval_max_pieces": args.final_eval_max_pieces,
        "final_eval_seed_base": args.final_eval_seed_base,
        "final_eval_seeds": final_eval_seeds,
        "uncensored_max_pieces": args.uncensored_max_pieces,
        # pieces_trained semantics (documented for downstream readers):
        "pieces_trained_meaning": "cumulative pieces simulated during fitness eval",
    }


def _render_table(rows: list[dict], title: str) -> Table:
    table = Table(title=title)
    for col in (
        "gen",
        "cap",
        "best fit",
        "elite fit",
        "eval median",
        "eval mean",
        "pieces (M)",
        "elapsed s",
        "ETA s",
    ):
        table.add_column(col, justify="right")
    for r in rows[-15:]:
        table.add_row(
            str(r["gen"]),
            str(r["cap"]),
            f"{r['best_fitness']:.1f}",
            f"{r['elite_fitness']:.1f}",
            f"{r['eval_median']:.1f}",
            f"{r['eval_mean']:.1f}",
            f"{r['pieces'] / 1e6:.2f}",
            f"{r['elapsed']:.1f}",
            "-" if r["eta"] is None else f"{r['eta']:.0f}",
        )
    return table


def run_cem(run: RunWriter, cfg: dict, console: Console, pool) -> dict:
    """Drive the noisy-CEM loop; log + checkpoint + replay each generation.

    Returns the final resolved mean weights (used for the final eval + the
    ``cem_final`` checkpoint).
    """
    trainer = CEM(
        seed=cfg["seed"],
        population=cfg["population"],
        elites=cfg["elites"],
        sigma_init=cfg["sigma_init"],
        fitness_games=cfg["fitness_games"],
        fitness_max_pieces=cfg["fitness_max_pieces"],
        fitness_seed_base=cfg["fitness_seed_base"],
        fitness_cap_max=cfg["fitness_cap_max"],
    )

    eval_seeds = cfg["eval_seeds"]
    table_rows: list[dict] = []
    gen_times: list[float] = []
    pieces_trained = 0
    stopped_early_at: int | None = None
    stop_reason: str | None = None
    # Elite means of consecutive generations run at the maximum cap (reset
    # whenever a generation runs below it) — feeds converged_at_ceiling.
    max_cap_elite_means: list[float] = []

    with Live(
        _render_table(table_rows, "noisy-CEM training"),
        console=console,
        refresh_per_second=4,
    ) as live:
        for gen in range(cfg["generations"]):
            gen_start = time.time()

            gr = trainer.run_generation(gen, pool=pool)
            pieces_trained += gr.pieces_this_gen

            # Periodic greedy eval of the refit mean on fixed seeds (observability).
            ev = parallel_eval(
                pool, gr.mu, eval_seeds, cfg["eval_max_pieces"], record=True
            )
            best = ev.best_index

            gen_dt = time.time() - gen_start
            gen_times.append(gen_dt)
            pps = gr.pieces_this_gen / max(gen_dt, 1e-9)

            run.log(
                phase="cem",
                pieces_trained=pieces_trained,
                loss=float(-gr.best_fitness),  # negative best fitness stand-in
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
                    "weights": gr.mu.tolist(),
                    "variance": gr.var.tolist(),
                    "best_fitness": gr.best_fitness,
                    "mean_fitness": gr.mean_fitness,
                    "elite_mean_fitness": gr.elite_mean_fitness,
                    "fitness_cap": gr.fitness_cap,
                    "cap_saturated": gr.saturated,
                    "next_fitness_cap": gr.next_fitness_cap,
                    "fitness_seeds": gr.fit_seeds,
                    "eval": {
                        "median_lines": ev.median_lines,
                        "mean_lines": ev.mean_lines,
                        "p10_lines": ev.p10_lines,
                        "mean_pieces": ev.mean_pieces,
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
            # ETA from the most recent generations only: the adaptive cap makes
            # early (cheap) generations unrepresentative of late (big-cap) ones.
            recent = gen_times[-3:]
            eta = (sum(recent) / len(recent)) * remaining if remaining else None
            table_rows.append(
                {
                    "gen": gen,
                    "cap": gr.fitness_cap,
                    "best_fitness": gr.best_fitness,
                    "elite_fitness": gr.elite_mean_fitness,
                    "eval_median": ev.median_lines,
                    "eval_mean": ev.mean_lines,
                    "pieces": pieces_trained,
                    "elapsed": time.time() - run.start_time,
                    "eta": eta,
                }
            )
            live.update(_render_table(table_rows, "noisy-CEM training"))

            if gr.fitness_cap >= cfg["fitness_cap_max"]:
                max_cap_elite_means.append(gr.elite_mean_fitness)
            else:
                max_cap_elite_means = []

            if cem.converged_at_ceiling(
                gen=gen,
                cap_pieces=gr.fitness_cap,
                cap_max=cfg["fitness_cap_max"],
                recent_elite_means=max_cap_elite_means,
            ):
                stopped_early_at = gen
                stop_reason = (
                    f"converged at measurement ceiling, gen {gen} of "
                    f"{cfg['generations']}: last {cem.CEILING_STOP_WINDOW} "
                    "generations' elite mean fitness >= "
                    f"{cem.CEILING_STOP_FRACTION:.0%} of the "
                    f"{cfg['fitness_cap_max']}-piece cap's line ceiling"
                )
                break

            if cem.should_stop_early(
                gen=gen + 1,
                cap_pieces=trainer.fitness_cap,
                cap_max=cfg["fitness_cap_max"],
                saturated=gr.saturated,
                var=trainer.var,
                std_threshold=cfg["early_stop_std_threshold"],
            ):
                stopped_early_at = gen
                stop_reason = (
                    f"sampling distribution collapsed after gen {gen} "
                    f"(max std {float(np.sqrt(trainer.var.max())):.3f} < "
                    f"{cfg['early_stop_std_threshold']}), cap at max, unsaturated"
                )
                break

    if stop_reason is not None:
        console.print(f"[yellow]early stop[/yellow]: {stop_reason}")

    return {
        "weights": trainer.mu.tolist(),
        "pieces_trained": pieces_trained,
        "final_fitness_cap": trainer.fitness_cap,
        "stopped_early_at_gen": stopped_early_at,
        "early_stop_reason": stop_reason,
    }


def eval_weights_on_seeds(weights, seeds: list[int], max_pieces: int, pool) -> dict:
    """Parallel greedy eval of one weight vector -> a JSON-ready stats dict."""
    ev = parallel_eval(pool, np.asarray(weights), seeds, max_pieces, record=False)
    return {
        "games": len(seeds),
        "max_pieces": max_pieces,
        "seeds": list(seeds),
        "median_lines": ev.median_lines,
        "mean_lines": ev.mean_lines,
        "p10_lines": ev.p10_lines,
        "mean_pieces": ev.mean_pieces,
        "lines": ev.lines,
        "pieces": ev.pieces,
        "games_hit_cap": sum(1 for p in ev.pieces if p >= max_pieces),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Noisy-CEM trainer over the 8 BCTS weights")
    ap.add_argument("--run-name", default="cem_v1")
    ap.add_argument("--smoke", action="store_true", help="tiny budget, run name cem_smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    # Full-run defaults follow PLAN.md §7.
    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--elites", type=int, default=10)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--sigma-init", type=float, default=10.0)
    ap.add_argument("--fitness-games", type=int, default=3)
    ap.add_argument("--fitness-max-pieces", type=int, default=800)
    ap.add_argument("--fitness-seed-base", type=int, default=10_000)
    # Periodic greedy eval (observability only; modest budget).
    ap.add_argument("--eval-games", type=int, default=5)
    ap.add_argument("--eval-max-pieces", type=int, default=2_000)
    ap.add_argument("--eval-seed-base", type=int, default=900)
    # Adaptive fitness cap ceiling (see module docstring).
    ap.add_argument("--fitness-cap-max", type=int, default=25_600)
    # Final 30-game report (§7).
    ap.add_argument("--final-eval-games", type=int, default=30)
    ap.add_argument("--final-eval-max-pieces", type=int, default=200_000)
    ap.add_argument("--final-eval-seed-base", type=int, default=100)
    # Uncensored gate-3 rematch cap (used only when both agents cap all games).
    ap.add_argument("--uncensored-max-pieces", type=int, default=2_000_000)
    args = ap.parse_args(argv)

    if args.smoke:
        args.run_name = "cem_smoke"
        args.population = 8
        args.elites = 3
        args.generations = 3
        args.fitness_games = 1
        args.fitness_max_pieces = 40
        args.fitness_cap_max = 160
        args.eval_games = 2
        args.eval_max_pieces = 150
        args.final_eval_games = 3
        args.final_eval_max_pieces = 500
        args.uncensored_max_pieces = 2_000
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _build_config(args)
    console = Console()

    ctx = mp.get_context("spawn")
    workers = max(1, args.workers)
    with RunWriter(args.run_name, cfg, phase="cem") as run:
        with ctx.Pool(processes=workers) as pool:
            final_weights_info = run_cem(run, cfg, console, pool)
            weights = final_weights_info["weights"]
            seeds = cfg["final_eval_seeds"]

            console.print("[cyan]running final 30-game evaluation (gate 2)...[/cyan]")
            final_eval = eval_weights_on_seeds(
                weights, seeds, cfg["final_eval_max_pieces"], pool
            )
            console.print("[cyan]evaluating Dellacherie on the same seeds (gate 3)...[/cyan]")
            della = eval_weights_on_seeds(
                DELLACHERIE_WEIGHTS, seeds, cfg["final_eval_max_pieces"], pool
            )

            # Gate-3 measurement protocol: if BOTH agents cap every game, the
            # comparison is censored at the ceiling — rematch both, same seeds,
            # at the uncensored cap and let those means decide.
            n = len(seeds)
            censored = (
                final_eval["games_hit_cap"] == n and della["games_hit_cap"] == n
            )
            uncensored = None
            if censored:
                console.print(
                    "[yellow]both agents capped all games — running uncensored "
                    f"rematch at {cfg['uncensored_max_pieces']:,} pieces...[/yellow]"
                )
                uncensored = {
                    "max_pieces": cfg["uncensored_max_pieces"],
                    "cem": eval_weights_on_seeds(
                        weights, seeds, cfg["uncensored_max_pieces"], pool
                    ),
                    "dellacherie": eval_weights_on_seeds(
                        DELLACHERIE_WEIGHTS, seeds, cfg["uncensored_max_pieces"], pool
                    ),
                }

        if uncensored is not None:
            beats = (
                uncensored["cem"]["mean_lines"] > uncensored["dellacherie"]["mean_lines"]
            )
        else:
            beats = final_eval["mean_lines"] > della["mean_lines"]

        final_payload = {
            "algorithm": "noisy-cem",
            "seed": cfg["seed"],
            "weights": weights,
            "pieces_trained": final_weights_info["pieces_trained"],
            "final_fitness_cap": final_weights_info["final_fitness_cap"],
            "stopped_early_at_gen": final_weights_info["stopped_early_at_gen"],
            "early_stop_reason": final_weights_info["early_stop_reason"],
            "eval": final_eval,
            "dellacherie_baseline": della,
            "comparison_censored_at_final_cap": censored,
            "uncensored_comparison": uncensored,
            "beats_dellacherie": beats,
        }
        run.save_json_checkpoint("cem_final", final_payload)

    console.print(
        f"[green]run complete[/green] -> {run.run_dir}\n"
        f"final mean lines : {final_eval['mean_lines']:.1f} "
        f"(median {final_eval['median_lines']:.1f}, p10 {final_eval['p10_lines']:.1f}, "
        f"{final_eval['games_hit_cap']}/{final_eval['games']} at cap)\n"
        f"dellacherie mean : {della['mean_lines']:.1f} "
        f"(median {della['median_lines']:.1f}, "
        f"{della['games_hit_cap']}/{della['games']} at cap)"
    )
    if uncensored is not None:
        console.print(
            f"uncensored (cap {uncensored['max_pieces']:,}): "
            f"cem mean {uncensored['cem']['mean_lines']:.1f} vs "
            f"dellacherie mean {uncensored['dellacherie']['mean_lines']:.1f}"
        )
    console.print(f"beats dellacherie: {beats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
