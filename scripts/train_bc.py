"""BC + DAgger driver for the pixel-input keypress policy (PLAN2.md §6, Phase D).

Trains :class:`tetris.policy_model.PolicyNet` by class-weighted behavioral
cloning on the keypress-expert corpus (``runs/bc_data_v1``), then optionally runs
DAgger: roll out the student, relabel every visited decision frame with the
expert's CURRENT-POSE action, aggregate, and continue training. Closed-loop
greedy eval (20 real-time games, seeds 950000+, 10k-piece cap) runs every
quarter-epoch and at every milestone; :mod:`tetris.bc` owns the training loop,
the eval, and the DAgger relabel/rollout so Phase E's PPO reuses the same eval.

Observability: the run-dir contract (``tetris.runio.RunWriter``) — metrics.jsonl
(frozen schema; `pieces_trained` carries the OPTIMIZER-STEP index for this
trainer), TB scalars, milestone checkpoints at 0/25/50/100% of optimizer steps,
and one best-game replay per eval. Per-class accuracy is NOT a metrics-schema
field, so it is stored in each checkpoint's payload metadata and printed.

Device: tries MPS, verifies logits parity vs CPU (< 1e-3 on a batch) and falls
back to CPU if unavailable or divergent; the choice is printed and recorded.

``--smoke`` (< 2 min): tiny synthetic CEM dataset, a handful of optimizer steps,
2-game eval, one tiny DAgger iter, MPS-vs-CPU parity check — exercises the whole
pipeline and asserts the run dir is well-formed.
"""

import _pathshim  # noqa: F401
import argparse
import json
import tempfile
import time

import numpy as np
import torch
from rich.console import Console

from tetris import bc
from tetris.frame_env import ACTIONS
from tetris.keypress_expert import make_teacher
from tetris.policy_model import PolicyNet, count_parameters
from tetris.runio import RunWriter

MILESTONE_PCTS = (0, 25, 50, 100)


def _build_config(args, device, total_steps, n_frames) -> dict:
    return {
        "algorithm": "bc+dagger",
        "smoke": args.smoke,
        "device": device,
        "requested_device": args.device,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "n_frames": n_frames,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "total_optimizer_steps": total_steps,
        "dagger_iters": args.dagger_iters,
        "dagger_frames": args.dagger_frames,
        "dagger_retrain_epochs": 1,
        "dagger_aggregation": "base + all dagger shards, reshuffled 1 epoch, "
                              "continues optimizer state",
        "teacher": args.teacher,
        "eval_games": args.eval_games,
        "eval_max_pieces": args.eval_max_pieces,
        "eval_seed_base": bc.EVAL_SEED_BASE,
        "rollout_max_pieces": args.rollout_max_pieces,
        "milestone_pcts": list(MILESTONE_PCTS),
        "pieces_trained_meaning": "optimizer-step index (BC has no self-play pieces)",
        "per_class_accuracy_location": "checkpoint payload metadata (schema frozen)",
    }


def _save_ckpt(run, name, model, step, phase, per_class, median):
    run.save_torch_checkpoint(name, {
        "model_state": model.state_dict(),
        "arch": "PolicyNet",
        "optimizer_step": step,
        "phase": phase,
        "per_class_accuracy": per_class,
        "eval_median_lines": median,
    })


def _do_eval(run, model, cfg, step, phase, loss, pps, dataset, rng, console):
    seeds = list(range(bc.EVAL_SEED_BASE, bc.EVAL_SEED_BASE + cfg["eval_games"]))
    ev = bc.evaluate_policy(model, seeds, cfg["eval_max_pieces"], cfg["device"])
    per_class = bc.per_class_accuracy(model, dataset, cfg["device"],
                                      cfg["pca_samples"], rng)
    run.log(phase=phase, pieces_trained=step, loss=loss,
            eval_median_lines=ev.median_lines, eval_mean_lines=ev.mean_lines,
            eval_p10_lines=ev.p10_lines, eval_pieces_per_game=ev.mean_pieces,
            pps_train=pps)
    b = ev.best_index
    run.save_replay(seed=ev.seeds[b], moves=ev.moves[b] or [],
                    final={"lines": ev.lines[b], "pieces": ev.pieces[b]},
                    pieces_trained=step, median_lines=ev.median_lines)
    pca = {k: (round(v, 3) if v is not None else None) for k, v in per_class.items()}
    console.print(f"[cyan]{phase}[/cyan] step={step} loss="
                  f"{'-' if loss is None else f'{loss:.4f}'} "
                  f"median={ev.median_lines:.1f} mean={ev.mean_lines:.1f} "
                  f"p10={ev.p10_lines:.1f} pcs={ev.mean_pieces:.0f} | acc={pca}")
    return ev, per_class


def train(run, cfg, args, console) -> dict:
    device = cfg["device"]
    rng = np.random.default_rng(args.seed)
    model = PolicyNet().to(device)
    console.print(f"[cyan]PolicyNet[/cyan]: {count_parameters(model):,} params, device={device}")

    dataset = bc.BCDataset(args.data_dir)
    weights = bc.inverse_freq_weights(bc.class_histogram(np.asarray(dataset.actions)))
    wt = bc.weight_tensor(weights, device)
    total = bc.num_optimizer_steps(len(dataset), cfg["batch_size"], cfg["epochs"])
    steps_per_epoch = bc.num_optimizer_steps(len(dataset), cfg["batch_size"], 1)
    eval_every = max(1, steps_per_epoch // 4)
    milestone_steps = {p: (round(p / 100 * total)) for p in MILESTONE_PCTS}
    console.print(f"[cyan]BC[/cyan]: {len(dataset):,} frames, {cfg['epochs']} epochs, "
                  f"{total:,} steps, eval every {eval_every}, milestones {milestone_steps}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    eval_medians = {}      # optimizer_step -> median (for milestone lookup)
    ckpt_names = {}        # optimizer_step -> checkpoint name

    # 0% milestone: genuinely untrained net.
    ev0, pc0 = _do_eval(run, model, cfg, 0, "bc", None, 0.0, dataset, rng, console)
    _save_ckpt(run, "nn_step_0", model, 0, "bc", pc0, ev0.median_lines)
    eval_medians[0] = ev0.median_lines
    ckpt_names[0] = "nn_step_0"

    step = 0
    loss_sum = loss_cnt = 0
    t_last = time.perf_counter()
    seen_last = 0
    for _ep, stacks, labels in bc.iter_minibatches(dataset, cfg["batch_size"],
                                                   cfg["epochs"], rng):
        loss = bc.train_step(model, optimizer, stacks, labels, wt, device)
        step += 1
        loss_sum += loss
        loss_cnt += 1
        seen_last += len(labels)
        is_milestone = step in milestone_steps.values()
        if step % eval_every == 0 or step == total or is_milestone:
            now = time.perf_counter()
            pps = seen_last / max(now - t_last, 1e-9)
            mloss = loss_sum / max(loss_cnt, 1)
            ev, pc = _do_eval(run, model, cfg, step, "bc", mloss, pps, dataset, rng, console)
            eval_medians[step] = ev.median_lines
            if is_milestone or step == total:
                name = f"nn_step_{step}"
                _save_ckpt(run, name, model, step, "bc", pc, ev.median_lines)
                ckpt_names[step] = name
            loss_sum = loss_cnt = seen_last = 0
            t_last = now

    bc_only_median = eval_medians.get(total, ev.median_lines)
    ckpt25_step = milestone_steps[25]
    ckpt25_median = eval_medians.get(ckpt25_step)
    if ckpt25_median is None and eval_medians:  # nearest recorded eval to 25%
        ckpt25_step = min(eval_medians, key=lambda s: abs(s - milestone_steps[25]))
        ckpt25_median = eval_medians[ckpt25_step]

    # -- DAgger iterations --------------------------------------------------
    teacher = make_teacher(args.teacher, args.checkpoint, "cpu")
    shards = [dataset]
    dagger_medians = []
    dagger_dir = run.run_dir / "dagger"
    for it in range(cfg["dagger_iters"]):
        console.print(f"[magenta]DAgger iter {it}[/magenta]: rolling out student "
                      f"for {cfg['dagger_frames']:,} frames")
        roll = bc.dagger_rollout(model, teacher, cfg["dagger_frames"],
                                 base_seed=args.dagger_seed_base + it * 100000,
                                 device=device, max_pieces=cfg["rollout_max_pieces"],
                                 progress=args.progress)
        shard_dir = dagger_dir / f"iter_{it}"
        bc.write_dagger_dataset(shard_dir, roll)
        shards.append(bc.BCDataset(shard_dir))
        combined = bc.MultiBCDataset(shards)
        weights = bc.combined_class_weights(combined)
        wt = bc.weight_tensor(weights, device)
        console.print(f"  relabeled {roll['n_frames']:,} frames in {roll['elapsed_s']}s; "
                      f"combined = {len(combined):,} frames; retrain 1 epoch")
        for _ep, stacks, labels in bc.iter_minibatches(combined, cfg["batch_size"], 1, rng):
            loss = bc.train_step(model, optimizer, stacks, labels, wt, device)
            step += 1
        ev, pc = _do_eval(run, model, cfg, step, "dagger", loss, 0.0, combined, rng, console)
        _save_ckpt(run, f"nn_dagger_{it}", model, step, "dagger", pc, ev.median_lines)
        dagger_medians.append(ev.median_lines)

    dagger_final_median = dagger_medians[-1] if dagger_medians else None

    # -- summary into config.json ------------------------------------------
    milestones = {str(p): {"optimizer_step": milestone_steps[p],
                           "checkpoint": ckpt_names.get(milestone_steps[p])}
                  for p in MILESTONE_PCTS}
    cfg_path = run.run_dir / "config.json"
    payload = json.loads(cfg_path.read_text())
    payload["milestones"] = milestones
    payload["results"] = {
        "bc_only_median_lines": bc_only_median,
        "checkpoint25_median_lines": ckpt25_median,
        "dagger_final_median_lines": dagger_final_median,
        "dagger_medians": dagger_medians,
        "gate_bc_ge_100": bc_only_median >= 100,
        "gate_monotonic_ish": (dagger_final_median is None
                               or (dagger_final_median >= bc_only_median
                                   >= (ckpt25_median or 0))),
    }
    with open(cfg_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return payload["results"] | {"milestones": milestones, "total_steps": total}


def _make_smoke_dataset(console) -> str:
    """Tiny synthetic CEM-expert dataset for the smoke gate (a few short games)."""
    d = tempfile.mkdtemp(prefix="bc_smoke_data_")
    console.print(f"[cyan]smoke[/cyan]: generating tiny dataset in {d}")
    bc.generate_dataset(out_dir=d, total_pieces=12, max_game_pieces=6,
                        base_seed=444000, teacher_kind="cem", progress=False)
    return d


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="BC + DAgger trainer (PLAN2.md §6)")
    ap.add_argument("--run-name", default="bc_v2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps", help="preferred device (mps|cpu)")
    ap.add_argument("--data-dir", default=str(bc.DEFAULT_DATA_DIR))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dagger-iters", type=int, default=2)
    ap.add_argument("--dagger-frames", type=int, default=300_000)
    ap.add_argument("--dagger-seed-base", type=int, default=600_000)
    ap.add_argument("--rollout-max-pieces", type=int, default=10_000)
    ap.add_argument("--teacher", choices=["td", "cem"], default="td")
    ap.add_argument("--checkpoint", default=None, help="override teacher checkpoint")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-max-pieces", type=int, default=10_000)
    ap.add_argument("--pca-samples", type=int, default=5000)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.run_name = "bc_smoke"
        args.epochs = 1
        args.batch_size = 64
        args.dagger_iters = 1
        args.dagger_frames = 400
        args.rollout_max_pieces = 60
        args.teacher = "cem"
        args.eval_games = 2
        args.eval_max_pieces = 60
        args.pca_samples = 512
        args.progress = True
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    console = Console()

    device = bc.select_device(args.device)
    # MPS numeric parity gate: verify vs CPU, fall back to CPU on divergence.
    if device == "mps":
        probe = PolicyNet().to("cpu")
        diff = bc.mps_cpu_max_logit_diff(probe, "mps", seed=args.seed)
        if diff < 1e-3:
            console.print(f"[green]MPS parity OK[/green]: max logit diff {diff:.2e} < 1e-3")
        else:
            console.print(f"[yellow]MPS parity FAILED[/yellow]: {diff:.2e} >= 1e-3 -> CPU")
            device = "cpu"
    else:
        console.print(f"[cyan]device[/cyan]: {device} (MPS not requested/available)")

    if args.smoke:
        args.data_dir = _make_smoke_dataset(console)

    ds_n = bc.BCDataset(args.data_dir)
    n_frames = len(ds_n)
    total_steps = bc.num_optimizer_steps(n_frames, args.batch_size, args.epochs)
    cfg = _build_config(args, device, total_steps, n_frames)
    cfg.update({"device": device, "batch_size": args.batch_size,
                "epochs": args.epochs, "lr": args.lr, "eval_games": args.eval_games,
                "eval_max_pieces": args.eval_max_pieces,
                "dagger_iters": args.dagger_iters, "dagger_frames": args.dagger_frames,
                "rollout_max_pieces": args.rollout_max_pieces,
                "pca_samples": args.pca_samples})

    t0 = time.perf_counter()
    with RunWriter(args.run_name, cfg, phase="bc") as run:
        results = train(run, cfg, args, console)
    elapsed = time.perf_counter() - t0

    console.print(
        f"[green]run complete[/green] -> {run.run_dir}  ({elapsed:.1f}s)\n"
        f"BC-only median      : {results['bc_only_median_lines']}\n"
        f"25% ckpt median     : {results['checkpoint25_median_lines']}\n"
        f"DAgger final median : {results['dagger_final_median_lines']}\n"
        f"gate BC>=100        : {results['gate_bc_ge_100']}\n"
        f"gate monotonic-ish  : {results['gate_monotonic_ish']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
