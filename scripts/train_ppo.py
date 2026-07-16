"""Pure-RL PPO comparison arm driver (PLAN2.md §7, Phase E).

Trains :class:`tetris.policy_model.PolicyNet` **from scratch** with the minimal
PPO in :mod:`tetris.ppo` — the honest contrast to the BC/DAgger agent. There is
NO performance gate: the run is time-boxed and its result (expected: near-zero
lines) is reported as-is.

Hard time-box (PLAN2.md §7): training stops at ``--max-hours`` OR ``--max-frames``
decision frames, whichever comes first; SIGTERM stops it the same way. In every
case the driver writes a final checkpoint (``nn_final.pt``) and runs a final eval
before exiting.

Eval reuses the Phase D protocol exactly (``tetris.bc.evaluate_policy``, 20
greedy real-time games, seeds 950000+, 10k-piece cap) every ``--eval-every-frames``
plus a final eval. Observability is the frozen ``tetris.runio`` contract:
metrics.jsonl (``loss`` = total PPO loss; ``pieces_trained`` = decision frames
consumed; ``epsilon``/``beta`` null), TB scalars, periodic safety checkpoints
(milestone-free — only the final result matters), and one best-game replay per
eval.

Device: tries MPS, verifies logits parity vs CPU (< 1e-3), falls back to CPU on
divergence — same pattern as Phase D. ``--smoke`` (< 2 min, tiny envs/rollout/
frames, CPU-safe) exercises the whole pipeline and asserts the run dir is
well-formed, the loss is finite, and advantage normalization is sane.
"""

import _pathshim  # noqa: F401
import argparse
import json
import signal
import time

import numpy as np
import torch
from rich.console import Console

from tetris import bc
from tetris.policy_model import PolicyNet, count_parameters
from tetris.ppo import (
    CLIP_EPS,
    ENT_COEF,
    VF_COEF,
    VecFrameEnv,
    collect_rollout,
    normalize_advantages,
    ppo_update,
)
from tetris.runio import RunWriter


def _build_config(args, device) -> dict:
    return {
        "algorithm": "ppo-clip",
        "smoke": args.smoke,
        "device": device,
        "requested_device": args.device,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "rollout_len": args.rollout_len,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "lr": args.lr,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": CLIP_EPS,
        "entropy_coef": ENT_COEF,
        "value_coef": VF_COEF,
        "max_grad_norm": args.max_grad_norm,
        "max_hours": args.max_hours,
        "max_frames": args.max_frames,
        "eval_every_frames": args.eval_every_frames,
        "eval_games": args.eval_games,
        "eval_max_pieces": args.eval_max_pieces,
        "eval_seed_base": bc.EVAL_SEED_BASE,
        "reward": "clear_points[lines] at lock decision, -10 terminal (no shaping)",
        "policy_init": "PolicyNet from scratch (no BC init) — pure-RL contrast",
        "loss_meaning": "total PPO loss = policy + value_coef*value - entropy_coef*entropy",
        "pieces_trained_meaning": "decision frames consumed (n_envs*rollout_len accumulated)",
        "performance_gate": "NONE (honest-contrast arm; result reported as-is)",
    }


def _eval_and_log(run, model, cfg, device, frames, last_loss, pps, console,
                  phase="ppo"):
    seeds = list(range(bc.EVAL_SEED_BASE, bc.EVAL_SEED_BASE + cfg["eval_games"]))
    ev = bc.evaluate_policy(model, seeds, cfg["eval_max_pieces"], device)
    run.log(phase=phase, pieces_trained=frames, loss=last_loss,
            eval_median_lines=ev.median_lines, eval_mean_lines=ev.mean_lines,
            eval_p10_lines=ev.p10_lines, eval_pieces_per_game=ev.mean_pieces,
            pps_train=pps)
    b = ev.best_index
    run.save_replay(seed=ev.seeds[b], moves=ev.moves[b] or [],
                    final={"lines": ev.lines[b], "pieces": ev.pieces[b]},
                    pieces_trained=frames, median_lines=ev.median_lines)
    run.save_torch_checkpoint(f"nn_step_{frames}", {
        "model_state": model.state_dict(),
        "arch": "PolicyNet",
        "frames": frames,
        "phase": phase,
        "eval_median_lines": ev.median_lines,
    })
    model.train()  # evaluate_policy left the model in eval mode
    console.print(f"[cyan]{phase} eval[/cyan] frames={frames:,} "
                  f"median={ev.median_lines:.1f} mean={ev.mean_lines:.1f} "
                  f"p10={ev.p10_lines:.1f} pcs={ev.mean_pieces:.0f} "
                  f"loss={'-' if last_loss is None else f'{last_loss:.4f}'}")
    return ev


def run_training(run, cfg, model, optimizer, device, console,
                 now_fn=time.monotonic, stop_check=None):
    """The time-boxed PPO loop. ``now_fn``/``stop_check`` are injectable so the
    time-box exit is unit-testable. Returns a results dict."""
    stop_check = stop_check or (lambda: False)
    vec = VecFrameEnv(cfg["n_envs"], base_seed=cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    frames = 0
    updates = 0
    last_eval_frames = 0
    last_total = None
    max_seconds = cfg["max_hours"] * 3600.0
    t_start = now_fn()
    t_last = t_start
    frames_last = 0
    stop_reason = None

    while True:
        batch = collect_rollout(model, vec, cfg["rollout_len"], device, gen)
        stats = ppo_update(model, optimizer, batch, device,
                           epochs=cfg["epochs"],
                           minibatch_size=cfg["minibatch_size"],
                           max_grad_norm=cfg["max_grad_norm"], rng=rng)
        frames += batch["frames"]
        updates += 1
        last_total = stats["total"]

        now = now_fn()
        pps = (frames - frames_last) / max(now - t_last, 1e-9)
        run.log(phase="ppo", pieces_trained=frames, loss=last_total, pps_train=pps)
        t_last, frames_last = now, frames

        if updates == 1 or updates % cfg.get("log_every", 20) == 0:
            console.print(f"[dim]update {updates} frames={frames:,} "
                          f"loss={last_total:.4f} pg={stats['policy']:.4f} "
                          f"vf={stats['value']:.4f} ent={stats['entropy']:.3f} "
                          f"kl={stats['approx_kl']:.4f} pps={pps:.0f}[/dim]")

        if frames - last_eval_frames >= cfg["eval_every_frames"]:
            _eval_and_log(run, model, cfg, device, frames, last_total, pps, console)
            last_eval_frames = frames

        elapsed = now_fn() - t_start
        if frames >= cfg["max_frames"]:
            stop_reason = "max_frames"
        elif elapsed >= max_seconds:
            stop_reason = "max_hours"
        elif stop_check():
            stop_reason = "signal"
        if stop_reason is not None:
            break

    console.print(f"[yellow]time-box hit[/yellow]: {stop_reason} "
                  f"(frames={frames:,}, updates={updates}, "
                  f"elapsed={now_fn() - t_start:.1f}s) — final checkpoint + eval")
    final_ev = _eval_and_log(run, model, cfg, device, frames, last_total, 0.0,
                             console, phase="ppo_final")
    run.save_torch_checkpoint("nn_final", {
        "model_state": model.state_dict(),
        "arch": "PolicyNet",
        "frames": frames,
        "phase": "ppo_final",
        "eval_median_lines": final_ev.median_lines,
        "stop_reason": stop_reason,
    })

    results = {
        "stop_reason": stop_reason,
        "frames_trained": frames,
        "updates": updates,
        "final_median_lines": final_ev.median_lines,
        "final_mean_lines": final_ev.mean_lines,
        "final_p10_lines": final_ev.p10_lines,
        "final_pieces_per_game": final_ev.mean_pieces,
    }
    cfg_path = run.run_dir / "config.json"
    payload = json.loads(cfg_path.read_text())
    payload["results"] = results
    with open(cfg_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return results


def _smoke_checks(model, cfg, device, console):
    """Advantage-normalization + finite-loss sanity on one tiny rollout before
    the smoke training loop (PLAN2.md §7 smoke gate)."""
    from tetris.ppo import ppo_losses

    vec = VecFrameEnv(cfg["n_envs"], base_seed=cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    batch = collect_rollout(model, vec, cfg["rollout_len"], device, gen)
    norm = normalize_advantages(batch["advantages"])
    assert abs(float(norm.mean())) < 1e-4, f"adv mean {norm.mean()} not ~0"
    assert abs(float(norm.std()) - 1.0) < 1e-2, f"adv std {norm.std()} not ~1"
    x = torch.from_numpy(batch["obs"].astype(np.float32) / 255.0).to(device)
    logits, values = model(x)
    losses = ppo_losses(
        logits, values,
        torch.from_numpy(batch["actions"]).to(device),
        torch.from_numpy(batch["logprobs"]).to(device),
        torch.from_numpy(norm).to(device),
        torch.from_numpy(batch["returns"]).to(device),
    )
    total = float(losses["total"].item())
    assert np.isfinite(total), f"non-finite loss {total}"
    console.print(f"[green]smoke checks OK[/green]: adv mean={norm.mean():.2e} "
                  f"std={norm.std():.4f}, total_loss={total:.4f} finite")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pure-RL PPO comparison arm (PLAN2.md §7)")
    ap.add_argument("--run-name", default="ppo_v2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps", help="preferred device (mps|cpu)")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--rollout-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=5_000_000)
    ap.add_argument("--eval-every-frames", type=int, default=250_000)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-max-pieces", type=int, default=10_000)
    args = ap.parse_args(argv)
    if args.smoke:
        args.run_name = "ppo_smoke"
        args.n_envs = 2
        args.rollout_len = 8
        args.epochs = 1
        args.minibatch_size = 16
        args.max_hours = 0.05
        args.max_frames = 48
        args.eval_every_frames = 16
        args.eval_games = 2
        args.eval_max_pieces = 40
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    console = Console()

    device = bc.select_device(args.device)
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

    cfg = _build_config(args, device)
    cfg["log_every"] = 1 if args.smoke else 20

    model = PolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    console.print(f"[cyan]PolicyNet[/cyan]: {count_parameters(model):,} params, "
                  f"device={device}, from scratch (pure-RL)")

    # SIGTERM => graceful stop at the next rollout boundary (checkpoint + exit).
    box = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: box.__setitem__("stop", True))

    if args.smoke:
        _smoke_checks(model, cfg, device, console)

    t0 = time.perf_counter()
    with RunWriter(args.run_name, cfg, phase="ppo") as run:
        results = run_training(run, cfg, model, optimizer, device, console,
                               stop_check=lambda: box["stop"])
    elapsed = time.perf_counter() - t0

    console.print(
        f"[green]run complete[/green] -> {run.run_dir}  ({elapsed:.1f}s)\n"
        f"stop reason       : {results['stop_reason']}\n"
        f"frames trained    : {results['frames_trained']:,}\n"
        f"final median lines: {results['final_median_lines']}\n"
        f"final mean lines  : {results['final_mean_lines']}"
    )

    if args.smoke:
        assert (run.run_dir / "config.json").exists()
        assert (run.run_dir / "metrics.jsonl").stat().st_size > 0
        assert (run.checkpoints_dir / "nn_final.pt").exists()
        assert (run.replays_dir / "index.json").exists()
        console.print("[green]SMOKE PASS[/green]: run dir well-formed, "
                      "loss finite, advantage normalization sane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
