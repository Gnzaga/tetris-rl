"""Toddler-spam diagnosis for the v2 pixel agents (user-requested follow-up).

Evaluates every current bc_v2 checkpoint + ppo_v2 final under the CURRENT
gray-128 renderer on 20 closed-loop games (seeds 950000+, 10k-piece cap — the
agents die at ~25 pieces so this is fast), collecting median/mean lines,
pieces/game, presses-per-piece, press fraction, per-action histogram and the
thrash score (fraction of presses immediately reversed; see
``tetris.bc.thrash_score``). Each is also evaluated with LOG-PRIOR CALIBRATION —
logits + scale*log(natural class prior), scale in {0.5, 1.0} — which the Phase D
debug report showed cuts thrash. Prints a markdown table and writes the full
results (incl. the precomputed logit_bias vectors) to a JSON for the registry /
manifest wiring.

    .venv/bin/python scripts/diagnose_spam.py --out scratchpad/diagnosis.json
"""

import _pathshim  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from tetris import bc
from tetris.export import load_policynet
from tetris.frame_env import ACTIONS, FrameEnv
from tetris.keypress_expert import ExpertPlayer, make_teacher
from tetris.render_obs import render_env  # noqa: F401  (imported for parity of env)

_ROOT = Path(__file__).resolve().parent.parent
RUNS = _ROOT / "runs"

EVAL_SEED_BASE = 950000
EVAL_GAMES = 20
EVAL_CAP = 10000
SCALES = (0.0, 0.5, 1.0)


def _models():
    bc_ck = RUNS / "bc_v2" / "checkpoints"
    ppo_ck = RUNS / "ppo_v2" / "checkpoints"
    return [
        ("bc_nn_step_0", bc_ck / "nn_step_0.pt", "BC 0% (untrained)"),
        ("bc_nn_step_11642", bc_ck / "nn_step_11642.pt", "BC 25%"),
        ("bc_nn_step_23284", bc_ck / "nn_step_23284.pt", "BC 50%"),
        ("bc_nn_step_46568", bc_ck / "nn_step_46568.pt", "BC 100%"),
        ("bc_dagger_0", bc_ck / "nn_dagger_0.pt", "DAgger iter 1"),
        ("bc_dagger_1", bc_ck / "nn_dagger_1.pt", "DAgger final (current demo default)"),
        ("ppo_final", ppo_ck / "nn_final.pt", "PPO final (contrast)"),
    ]


def _natural_prior() -> np.ndarray:
    """Natural class prior (fractions) in ACTIONS order from the training corpus
    meta (runs/bc_data_v2/meta.json class_fractions)."""
    meta = json.loads((RUNS / "bc_data_v2" / "meta.json").read_text())
    cf = meta["class_fractions"]
    return np.array([cf[a] for a in ACTIONS], dtype=np.float64)


def _expert_reference(device: str, seeds=(950000, 950001, 950002), cap=80) -> dict:
    """Presses-per-piece + thrash of the keypress EXPERT itself (the ~2-3
    presses/piece target the agents are measured against)."""
    teacher = make_teacher("td", None, device)
    player = ExpertPlayer(teacher)
    presses = rev = decisions = pieces = 0
    for s in seeds:
        env = FrameEnv(seed=int(s))
        player.reset(env)
        seq = []
        while not env.game_over and env.pieces < cap:
            if env.is_decision_tick:
                a = player.act(env)
                env.apply_action(a)
                seq.append(a)
            env.tick()
        r, p = bc.thrash_score(seq)
        rev += r
        presses += p
        decisions += len(seq)
        pieces += env.pieces
    return {
        "presses_per_piece": presses / max(pieces, 1),
        "press_frac": presses / max(decisions, 1),
        "thrash": rev / max(presses, 1),
        "pieces": pieces,
        "seeds": list(seeds),
        "cap": cap,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pixel-agent toddler-spam diagnosis")
    ap.add_argument("--out", default=str(_ROOT / "scratchpad" / "diagnosis.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--games", type=int, default=EVAL_GAMES)
    args = ap.parse_args(argv)

    prior = _natural_prior()
    log_prior = np.log(prior)  # negative for rare press classes
    seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + args.games))

    print(f"natural prior       : {dict(zip(ACTIONS, np.round(prior, 4)))}")
    print(f"log-prior (bias @1.0): {dict(zip(ACTIONS, np.round(log_prior, 3)))}\n")

    expert = _expert_reference(args.device)
    print(f"EXPERT reference: presses/piece={expert['presses_per_piece']:.2f} "
          f"press_frac={expert['press_frac']:.3f} thrash={expert['thrash']:.3f}\n")

    results = {}
    rows = []
    for mid, ckpt, label in _models():
        if not ckpt.exists():
            print(f"  (skip {mid}: {ckpt} missing)")
            continue
        model = load_policynet(ckpt, device=args.device)
        results[mid] = {"label": label, "checkpoint": str(ckpt), "scales": {}}
        for scale in SCALES:
            bias = (scale * log_prior) if scale else None
            ev = bc.evaluate_policy_rich(model, seeds, EVAL_CAP, args.device, bias)
            results[mid]["scales"][f"{scale}"] = {
                **{k: ev[k] for k in ("median_lines", "mean_lines", "pieces_per_game",
                                      "presses_per_piece", "press_frac", "thrash",
                                      "action_hist", "total_pieces", "total_presses")},
                "logit_bias": (bias.tolist() if bias is not None else None),
            }
            rows.append((label, scale, ev))
            print(f"{label:38s} scale={scale:<3} | med={ev['median_lines']:>4.1f} "
                  f"mean={ev['mean_lines']:>4.2f} pcs/g={ev['pieces_per_game']:>5.1f} "
                  f"pp/piece={ev['presses_per_piece']:>5.2f} "
                  f"press_frac={ev['press_frac']:.3f} thrash={ev['thrash']:.3f}")
        print()

    out = {
        "eval": {"seeds_base": EVAL_SEED_BASE, "games": args.games, "cap": EVAL_CAP,
                 "device": args.device, "renderer": "gray-128 (current)"},
        "natural_prior": {a: float(p) for a, p in zip(ACTIONS, prior)},
        "log_prior": {a: float(x) for a, x in zip(ACTIONS, log_prior)},
        "expert_reference": expert,
        "models": results,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
