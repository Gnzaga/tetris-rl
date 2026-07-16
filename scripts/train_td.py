"""TD(0) afterstate ValueNet trainer (PLAN.md §8).

Vectorized single-process self-play (``tetris.td.VecSelfPlay``) feeds a replay
buffer; the online ValueNet is trained by Huber TD(0) toward a hard-copied
target network. Optional ``--warm-start`` behavior-clones the CEM teacher (or the
built-in Dellacherie weights) before TD, which skips the random-flailing phase
and starts exploration at epsilon = 0.2 instead of 1.0.

This script owns the run-directory contract (``tetris.runio.RunWriter``), the
rich live table, periodic greedy eval, milestone checkpoints, and CLI; the
algorithm, replay buffer, schedules, and warm-start live in ``tetris.td``.

Milestone checkpoints (PLAN.md §8, for the demo): ``nn_step_0`` is the genuinely
untrained net (saved before any warm-start); the trainer then saves
``nn_step_<pieces>.pt`` at every eval and, at the end, records the checkpoints
nearest 0/10/30/60/100% of the budget into ``config.json``.

``--smoke`` forces run name ``td_smoke``, a tiny budget, and a small self-contained
Dellacherie warm-start so the whole pipeline (warm-start regression -> TD ->
eval -> checkpoints -> replays) is exercised in well under two minutes with a
finite, decreasing loss.
"""

import _pathshim  # noqa: F401
import argparse
import json
import os
import time

import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.table import Table

from tetris.agents import DELLACHERIE_WEIGHTS
from tetris.model import ValueNet, count_parameters
from tetris.rng import Mulberry32
from tetris.runio import RUNS_DIR, RunWriter
from tetris import td

MILESTONE_PCTS = (0, 10, 30, 60, 100)


def _build_config(args: argparse.Namespace) -> dict:
    eval_seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.eval_games))
    return {
        "algorithm": "td0-afterstate",
        "smoke": args.smoke,
        "seed": args.seed,
        "device": args.device,
        "threads": args.threads,
        "num_envs": args.num_envs,
        "gamma": args.gamma,
        "budget": args.budget,
        "buffer_capacity": args.buffer_capacity,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "target_sync": args.target_sync,
        "learning_starts": args.learning_starts,
        "updates_per_step": args.updates_per_step,
        "epsilon_end": 0.02,
        "epsilon_frac": 0.30,
        "epsilon_start_warm": 0.2,
        "epsilon_start_cold": 1.0,
        "beta_frac": 0.60,
        "warm_start": args.warm_start,
        "teacher_run": args.teacher_run,
        "warm_placements": args.warm_placements,
        "warm_epochs": args.warm_epochs,
        "warm_batch": args.warm_batch,
        "warm_lr": args.warm_lr,
        "eval_interval": args.eval_interval,
        "eval_games": args.eval_games,
        "eval_max_pieces": args.eval_max_pieces,
        "eval_seed_base": args.eval_seed_base,
        "eval_seeds": eval_seeds,
        "train_seed_base": args.train_seed_base,
        "teacher_seed_base": args.teacher_seed_base,
        "milestone_pcts": list(MILESTONE_PCTS),
        "pieces_trained_meaning": "cumulative placements simulated during self-play",
    }


def _load_teacher_weights(args: argparse.Namespace, console: Console) -> list[float]:
    """Teacher weights for warm-start: the CEM run's final weights when
    ``--teacher-run`` is given, else the built-in Dellacherie hand weights."""
    if args.teacher_run:
        path = RUNS_DIR / args.teacher_run / "checkpoints" / "cem_final.json"
        if not path.exists():
            raise SystemExit(f"teacher checkpoint not found: {path}")
        with open(path) as f:
            weights = json.load(f)["weights"]
        console.print(f"[cyan]warm-start teacher[/cyan]: {args.teacher_run} weights")
        return weights
    console.print("[cyan]warm-start teacher[/cyan]: built-in Dellacherie weights")
    return list(DELLACHERIE_WEIGHTS)


def _render_table(rows: list[dict], title: str) -> Table:
    table = Table(title=title)
    for col in ("pieces (K)", "%", "loss", "eps", "beta",
                "eval median", "eval mean", "pps", "elapsed s", "ETA s"):
        table.add_column(col, justify="right")
    for r in rows[-15:]:
        table.add_row(
            f"{r['pieces'] / 1e3:.0f}",
            f"{r['pct']:.0f}",
            "-" if r["loss"] is None else f"{r['loss']:.4f}",
            f"{r['eps']:.3f}",
            f"{r['beta']:.3f}",
            f"{r['eval_median']:.1f}",
            f"{r['eval_mean']:.1f}",
            f"{r['pps']:.0f}",
            f"{r['elapsed']:.1f}",
            "-" if r["eta"] is None else f"{r['eta']:.0f}",
        )
    return table


def _do_eval(model, cfg, pieces, loss, eps, beta, pps, run, console_rows, start_time):
    """Greedy eval + metrics log + replay + milestone-eligible checkpoint."""
    ev = td.evaluate_net(
        model,
        cfg["eval_seeds"],
        cfg["eval_max_pieces"],
        cfg["gamma"],
        device=cfg["device"],
        record=True,
    )
    run.log(
        phase="td",
        pieces_trained=pieces,
        loss=loss,
        epsilon=eps,
        beta=beta,
        eval_median_lines=ev.median_lines,
        eval_mean_lines=ev.mean_lines,
        eval_p10_lines=ev.p10_lines,
        eval_pieces_per_game=ev.mean_pieces,
        pps_train=pps,
    )
    ckpt = f"nn_step_{pieces}"
    run.save_torch_checkpoint(
        ckpt,
        {
            "model_state": model.state_dict(),
            "pieces_trained": pieces,
            "arch": "ValueNet",
        },
    )
    best = ev.best_index
    run.save_replay(
        seed=ev.seeds[best],
        moves=ev.moves[best] or [],
        final={"lines": ev.lines[best], "pieces": ev.pieces[best]},
        pieces_trained=pieces,
        median_lines=ev.median_lines,
    )
    return ev, ckpt


def _write_milestones(run_dir, budget, eval_points):
    """Record the checkpoint nearest each milestone % into config.json.

    ``eval_points`` is ``[(pieces, checkpoint_name), ...]`` for every saved eval
    checkpoint (including ``nn_step_0``). 0% maps to the untrained ``nn_step_0``.
    """
    milestones = {}
    for pct in MILESTONE_PCTS:
        target = (pct / 100.0) * budget
        pieces, name = min(eval_points, key=lambda p: abs(p[0] - target))
        milestones[str(pct)] = {"pieces_trained": pieces, "checkpoint": name}
    cfg_path = run_dir / "config.json"
    payload = json.loads(cfg_path.read_text())
    payload["milestones"] = milestones
    with open(cfg_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return milestones


def train(run: RunWriter, cfg: dict, args, console: Console) -> dict:
    device = cfg["device"]
    model = ValueNet().to(device)
    target = ValueNet().to(device)
    target.load_state_dict(model.state_dict())
    n_params = count_parameters(model)
    console.print(f"[cyan]ValueNet[/cyan]: {n_params:,} params")

    eval_points: list[tuple[int, str]] = []

    # nn_step_0 = the genuinely untrained net (before any warm-start).
    run.save_torch_checkpoint(
        "nn_step_0",
        {"model_state": model.state_dict(), "pieces_trained": 0, "arch": "ValueNet"},
    )
    eval_points.append((0, "nn_step_0"))

    table_rows: list[dict] = []
    with Live(_render_table(table_rows, "TD(0) ValueNet"), console=console,
              refresh_per_second=2) as live:
        # 0% eval: untrained model, so the learning curve starts at the floor.
        ev0, _ = _do_eval(model, cfg, 0, None, cfg["epsilon_start_cold"],
                          1.0, 0.0, run, table_rows, run.start_time)
        table_rows.append({
            "pieces": 0, "pct": 0.0, "loss": None,
            "eps": cfg["epsilon_start_cold"], "beta": 1.0,
            "eval_median": ev0.median_lines, "eval_mean": ev0.mean_lines,
            "pps": 0.0, "elapsed": time.time() - run.start_time, "eta": None,
        })
        live.update(_render_table(table_rows, "TD(0) ValueNet"))

        # -- warm-start (behavior cloning) ----------------------------------
        eps_start = cfg["epsilon_start_cold"]
        warm_losses = None
        if cfg["warm_start"]:
            teacher_weights = _load_teacher_weights(args, console)
            console.print(
                f"[cyan]collecting teacher data[/cyan]: {cfg['warm_placements']:,} placements"
            )
            data = td.collect_teacher_data(
                teacher_weights, cfg["warm_placements"], cfg["teacher_seed_base"]
            )
            console.print(
                f"[cyan]pretraining[/cyan]: {len(data.boards):,} candidate boards, "
                f"{cfg['warm_epochs']} epochs"
            )
            warm_losses = td.pretrain_regression(
                model, data, cfg["warm_epochs"], cfg["warm_batch"],
                cfg["warm_lr"], device, seed=cfg["seed"],
            )
            console.print(f"[cyan]warm-start MSE per epoch[/cyan]: "
                          + ", ".join(f"{x:.4f}" for x in warm_losses))
            target.load_state_dict(model.state_dict())
            eps_start = cfg["epsilon_start_warm"]
            run.save_json_checkpoint(
                "warmstart",
                {"epoch_losses": warm_losses, "num_boards": int(len(data.boards)),
                 "placements": cfg["warm_placements"]},
            )
            del data

        # -- TD(0) self-play loop -------------------------------------------
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        buffer = td.ReplayBuffer(cfg["buffer_capacity"], seed=cfg["seed"])
        selfplay = td.VecSelfPlay(cfg["num_envs"], cfg["train_seed_base"])
        eps_rng = Mulberry32(cfg["seed"] ^ 0x9E3779B9)

        budget = cfg["budget"]
        gamma = cfg["gamma"]
        batch = cfg["batch_size"]
        pieces = 0
        updates = 0
        next_eval = cfg["eval_interval"]
        loss_sum = 0.0
        loss_count = 0
        last_eval_pieces = 0
        last_eval_time = time.time()

        while pieces < budget:
            eps = td.epsilon_at(pieces, budget, eps_start)
            beta = td.beta_at(pieces, budget)

            s, r, ns, d, num = selfplay.step(model, beta, eps, gamma, eps_rng, device)
            buffer.add_many(s, r, ns, d)
            pieces += num

            if buffer.size >= max(batch, cfg["learning_starts"]):
                for _ in range(cfg["updates_per_step"]):
                    bs, br, bns, bd = buffer.sample(batch)
                    model.train()
                    v_s = model(td.boards_to_tensor(bs, device))
                    with torch.no_grad():
                        v_ns = target(td.boards_to_tensor(bns, device))
                        tgt = torch.from_numpy(br).to(device) + gamma * v_ns * (
                            1.0 - torch.from_numpy(bd).to(device)
                        )
                    loss = torch.nn.functional.smooth_l1_loss(v_s, tgt)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.item())
                    loss_count += 1
                    updates += 1
                    if updates % cfg["target_sync"] == 0:
                        target.load_state_dict(model.state_dict())

            if pieces >= next_eval or pieces >= budget:
                now = time.time()
                pps = (pieces - last_eval_pieces) / max(now - last_eval_time, 1e-9)
                mean_loss = loss_sum / loss_count if loss_count else None
                ev, ckpt = _do_eval(model, cfg, pieces, mean_loss, eps, beta, pps,
                                    run, table_rows, run.start_time)
                eval_points.append((pieces, ckpt))
                pct = 100.0 * pieces / budget
                remaining = budget - pieces
                elapsed = now - run.start_time
                eta = (elapsed * remaining / pieces) if pieces else None
                table_rows.append({
                    "pieces": pieces, "pct": pct, "loss": mean_loss,
                    "eps": eps, "beta": beta,
                    "eval_median": ev.median_lines, "eval_mean": ev.mean_lines,
                    "pps": pps, "elapsed": elapsed, "eta": eta,
                })
                live.update(_render_table(table_rows, "TD(0) ValueNet"))
                loss_sum = 0.0
                loss_count = 0
                last_eval_pieces = pieces
                last_eval_time = now
                while next_eval <= pieces:
                    next_eval += cfg["eval_interval"]

    milestones = _write_milestones(run.run_dir, budget, eval_points)
    return {
        "pieces_trained": pieces,
        "updates": updates,
        "num_params": n_params,
        "warm_losses": warm_losses,
        "milestones": milestones,
        "eval_points": eval_points,
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TD(0) afterstate ValueNet trainer")
    ap.add_argument("--run-name", default="td_v1")
    ap.add_argument("--smoke", action="store_true", help="tiny budget, run name td_smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    # Full-run defaults follow PLAN.md §8.
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--budget", type=int, default=2_000_000)
    ap.add_argument("--buffer-capacity", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-sync", type=int, default=2_500)
    ap.add_argument("--learning-starts", type=int, default=5_000)
    ap.add_argument("--updates-per-step", type=int, default=1)
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--teacher-run", default=None, help="CEM run whose cem_final weights seed warm-start")
    ap.add_argument("--warm-placements", type=int, default=200_000)
    ap.add_argument("--warm-epochs", type=int, default=3)
    ap.add_argument("--warm-batch", type=int, default=512)
    ap.add_argument("--warm-lr", type=float, default=1e-3)
    ap.add_argument("--eval-interval", type=int, default=50_000)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-max-pieces", type=int, default=10_000)
    ap.add_argument("--eval-seed-base", type=int, default=900_000)
    ap.add_argument("--train-seed-base", type=int, default=100_000)
    ap.add_argument("--teacher-seed-base", type=int, default=700_000)
    args = ap.parse_args(argv)

    if args.smoke:
        args.run_name = "td_smoke"
        args.num_envs = 8
        args.budget = 6_000
        args.buffer_capacity = 8_000
        args.batch_size = 64
        args.target_sync = 100
        args.learning_starts = 200
        args.warm_start = True
        args.teacher_run = None  # self-contained: built-in Dellacherie teacher
        args.warm_placements = 300
        args.warm_epochs = 3
        args.warm_batch = 64
        args.eval_interval = 2_000
        args.eval_games = 3
        args.eval_max_pieces = 300
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    cfg = _build_config(args)
    console = Console()

    with RunWriter(args.run_name, cfg, phase="td") as run:
        info = train(run, cfg, args, console)

    console.print(
        f"[green]run complete[/green] -> {run.run_dir}\n"
        f"pieces trained : {info['pieces_trained']:,}\n"
        f"updates        : {info['updates']:,}\n"
        f"params         : {info['num_params']:,}\n"
        f"milestones     : "
        + ", ".join(f"{k}%->{v['checkpoint']}" for k, v in info["milestones"].items())
    )
    if info["warm_losses"]:
        console.print(
            "warm-start MSE: "
            + " -> ".join(f"{x:.4f}" for x in info["warm_losses"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
