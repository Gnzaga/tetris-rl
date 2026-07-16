"""Model-registry CLI — durable bookmarks for every trained iteration.

    python scripts/registry.py seed [--diagnosis scratchpad/diagnosis.json]
    python scripts/registry.py list
    python scripts/registry.py add --id X --family pixel-bc --checkpoint ... [...]
    python scripts/registry.py eval <id> [--games 20]        # fresh closed-loop eval
    python scripts/registry.py export-demo                   # -> demo/models/registry.json

``seed`` builds the full lineage (v1 ValueNet, the four 8k-step minis, attempts
1 & 3, the shipped bc_v2 milestones + DAgger, and ppo_v2) with HISTORICAL eval
numbers for the archived/mismatched-domain checkpoints and FRESH closed-loop eval
numbers (from ``scripts/diagnose_spam.py`` output) for the current gray-128
bc_v2/ppo checkpoints. ``eval`` re-runs the closed-loop eval on one entry with the
CURRENT renderer and rewrites its eval block with ``source: fresh-eval``.
"""

import _pathshim  # noqa: F401
import argparse
import json
from pathlib import Path

from tetris import registry as reg

_ROOT = Path(__file__).resolve().parent.parent
RUNS = _ROOT / "runs"
ARCHIVE = RUNS / "archive"

# Git commits per lineage stage (from the archived config.json / debug report).
_C_TD = "e3cbe13baf93793d6713acd10857f5e69c27421b"
_C_A1 = "fa13766dc0d9191f039d916ee8d98731014a5cd4"   # attempt 1 plain BC
_C_A3 = "97abd02d64b1d37883af8611aacb9689ea257a24"   # attempt 3 DART
_C_A4 = "8897a14"                                    # attempt 4 spaced-stack patch
_C_A5 = "9e97bc3"                                    # attempt 5 gray render patch
_C_V2 = "b8b55761d809a9a2dad161eb926661ba6621e53d"   # shipped bc_v2 / ppo_v2


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT))
    except ValueError:
        return str(p)


def _diag(diagnosis_path: Path) -> dict:
    if not diagnosis_path.exists():
        raise SystemExit(
            f"{diagnosis_path} not found — run scripts/diagnose_spam.py first "
            f"(fresh-eval numbers for the current bc_v2/ppo checkpoints come from it)."
        )
    return json.loads(diagnosis_path.read_text())


def _fresh_eval(diag: dict, model_key: str) -> dict:
    """Build an eval block from the diagnosis scale=0.0 (uncalibrated) run."""
    s = diag["models"][model_key]["scales"]["0.0"]
    return reg.make_eval(
        median_lines=s["median_lines"], mean_lines=s["mean_lines"],
        pieces_per_game=s["pieces_per_game"], presses_per_piece=s["presses_per_piece"],
        press_frac=s["press_frac"], thrash=s["thrash"], source="fresh-eval",
    )


def build_seed_entries(diagnosis_path: Path) -> list[dict]:
    diag = _diag(diagnosis_path)
    bc_ck = RUNS / "bc_v2" / "checkpoints"
    E = []

    # --- v1 ValueNet planner (the immortal contrast target) ------------------
    E.append(reg.make_entry(
        id="v1_td_v1_final", created="2026-07-16T04:46:48+00:00",
        family="v1-valuenet", summary="v1 TD(0) afterstate ValueNet — final (2M placements)",
        git_commit=_C_TD, checkpoint=_rel(RUNS / "td_v1/checkpoints/nn_step_2000000.pt"),
        domain="board-int",
        eval=reg.make_eval(median_lines=3997.0, mean_lines=3976.55,
                           pieces_per_game=9950.2, source="historical-metrics"),
        notes="v1 board-tensor planner (immortal at level-0); the honest contrast "
              "the pixel agents are measured against. Not a keypress policy.",
    ))

    # --- attempt 1: plain BC + 2 DAgger, camouflage-255 render ----------------
    E.append(reg.make_entry(
        id="pixel_attempt1_bc", created="2026-07-16T16:03:26+00:00",
        family="pixel-bc", summary="Attempt 1 — plain BC (5-way balanced), BC-only final",
        git_commit=_C_A1,
        checkpoint=_rel(ARCHIVE / "attempt1_plain_bc/checkpoints/nn_step_58500.pt"),
        domain="camouflage-255",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.0, pieces_per_game=21.05,
                           source="historical-metrics"),
        notes="Camouflage render (active piece 255). Covariate-shift wall: 0.25 "
              "agreement on self-visited states, press recall ~1% off-manifold.",
    ))
    E.append(reg.make_entry(
        id="pixel_attempt1_dagger", created="2026-07-16T16:13:00+00:00",
        family="pixel-bc", summary="Attempt 1 — + 2 DAgger iters (final)",
        git_commit=_C_A1,
        checkpoint=_rel(ARCHIVE / "attempt1_plain_bc/checkpoints/nn_dagger_1.pt"),
        domain="camouflage-255",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.05, pieces_per_game=21.25,
                           source="historical-metrics"),
        notes="DAgger shifted the policy toward thrash (false-press 25% vs BC-only "
              "2.3%) without teaching recovery — median stayed 0.",
    ))

    # --- attempt 3 mini pilot (DART, consecutive stack), camouflage-255 -------
    E.append(reg.make_entry(
        id="mini_bc", created="2026-07-16T16:50:00+00:00",
        family="mini-experiment", summary="Mini — DART pilot 8k steps (consecutive 4-stack)",
        git_commit=_C_A3, checkpoint=_rel(ARCHIVE / "minis/mini_bc.pt"),
        domain="camouflage-255",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.33, pieces_per_game=23.3,
                           source="historical-metrics"),
        notes="8k-step DART pilot (291k frames). Cleared the FIRST line ever seen "
              "from any pixel checkpoint (games [0,0,1]); prompted the full attempt 3.",
    ))

    # --- attempt 3: DART + 50/50 sampler, camouflage-255 ----------------------
    E.append(reg.make_entry(
        id="pixel_attempt3_bc", created="2026-07-16T17:06:02+00:00",
        family="pixel-bc", summary="Attempt 3 — DART + 50/50 sampler, BC-only final",
        git_commit=_C_A3,
        checkpoint=_rel(ARCHIVE / "attempt3_dart/checkpoints/nn_step_46568.pt"),
        domain="camouflage-255",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.0, pieces_per_game=25.55,
                           source="historical-metrics"),
        notes="DART recovery states baked into 3M frames; survival improved (21->26 "
              "pieces) but line-clearing stayed ~0. State coverage fixed, precision not.",
    ))
    E.append(reg.make_entry(
        id="pixel_attempt3_dagger", created="2026-07-16T17:16:00+00:00",
        family="pixel-bc", summary="Attempt 3 — + 2 DAgger iters (final)",
        git_commit=_C_A3,
        checkpoint=_rel(ARCHIVE / "attempt3_dart/checkpoints/nn_dagger_1.pt"),
        domain="camouflage-255",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.1, pieces_per_game=27.15,
                           source="historical-metrics"),
        notes="Longest-surviving camouflage-render agent (~27 pieces); still median 0.",
    ))

    # --- attempt 4: spaced-stack mini, camouflage-255 -------------------------
    E.append(reg.make_entry(
        id="mini_spaced", created="2026-07-16T17:30:00+00:00",
        family="mini-experiment", summary="Mini — spaced stack {t,t-4,t-8,t-12} 8k steps",
        git_commit=_C_A4, checkpoint=_rel(ARCHIVE / "minis/mini_spaced.pt"),
        domain="camouflage-255",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.0, source="historical-metrics"),
        notes="Attempt-4 motion-visibility probe. Stopped at the mini gate: off-manifold "
              "press recall 2.41x (<3x bar); rot_cw held-in recall dropped to 0.385.",
    ))

    # --- attempt 5: gray render minis (+/- aux head), gray-128 ----------------
    E.append(reg.make_entry(
        id="mini_gray_noaux", created="2026-07-16T17:50:00+00:00",
        family="mini-experiment", summary="Mini — gray-128 render, no aux head 8k steps",
        git_commit=_C_A5, checkpoint=_rel(ARCHIVE / "minis/mini_gray_noaux.pt"),
        domain="gray-128",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.0, source="historical-metrics"),
        notes="Attempt-5 ablation. Off-manifold press recall only 1.28x with the piece "
              "made unambiguous — refuted perception as the binding constraint.",
    ))
    E.append(reg.make_entry(
        id="mini_gray", created="2026-07-16T17:55:00+00:00",
        family="mini-experiment", summary="Mini — gray-128 render + aux(0.5) head 8k steps",
        git_commit=_C_A5, checkpoint=_rel(ARCHIVE / "minis/mini_gray.pt"),
        domain="gray-128",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.0, source="historical-metrics"),
        notes="Aux(0.5) crowded out the sparse action signal at the 8k budget "
              "(recall 0.27x). Aux weight lowered to 0.1 for the shipped run.",
    ))

    # --- shipped bc_v2 (attempt 6): gray-128 + aux(0.1) — FRESH eval ----------
    bc_created = {
        "bc_nn_step_0":      ("pixel_bc_v2_000", "2026-07-16T18:13:32+00:00",
                              "bc_v2 — BC 0% (untrained net)", "nn_step_0.pt"),
        "bc_nn_step_11642":  ("pixel_bc_v2_025", "2026-07-16T18:23:00+00:00",
                              "bc_v2 — BC 25%", "nn_step_11642.pt"),
        "bc_nn_step_23284":  ("pixel_bc_v2_050", "2026-07-16T18:33:00+00:00",
                              "bc_v2 — BC 50%", "nn_step_23284.pt"),
        "bc_nn_step_46568":  ("pixel_bc_v2_100", "2026-07-16T18:43:00+00:00",
                              "bc_v2 — BC 100% (final)", "nn_step_46568.pt"),
        "bc_dagger_0":       ("pixel_bc_v2_dagger0", "2026-07-16T18:47:00+00:00",
                              "bc_v2 — DAgger iter 1", "nn_dagger_0.pt"),
        "bc_dagger_1":       ("pixel_bc_v2_dagger1", "2026-07-16T18:53:00+00:00",
                              "bc_v2 — DAgger final (shipped)", "nn_dagger_1.pt"),
    }
    for mkey, (eid, created, summary, fname) in bc_created.items():
        E.append(reg.make_entry(
            id=eid, created=created, family="pixel-bc", summary=summary,
            git_commit=_C_V2, checkpoint=_rel(bc_ck / fname), domain="gray-128",
            eval=_fresh_eval(diag, mkey),
            notes="Shipped best-effort pixel agent (gray-128 render + aux 0.1). "
                  "Fresh 20-game closed-loop eval, current renderer.",
        ))

    E.append(reg.make_entry(
        id="pixel_ppo_v2_final", created="2026-07-16T18:54:46+00:00",
        family="pixel-ppo", summary="ppo_v2 — pure PPO from scratch (time-boxed contrast)",
        git_commit=_C_V2, checkpoint=_rel(RUNS / "ppo_v2/checkpoints/nn_final.pt"),
        domain="gray-128", eval=_fresh_eval(diag, "ppo_final"),
        notes="5M frames, no BC init. The honest pure-RL contrast: near-zero lines.",
    ))
    return E


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_seed(args) -> int:
    entries = build_seed_entries(Path(args.diagnosis))
    reg.save(entries)
    reg.write_demo_index(entries)
    print(f"seeded {len(entries)} entries -> runs/registry.json")
    print(f"  committed index    -> shared/model_registry.json")
    print(f"  demo index         -> demo/models/registry.json")
    return 0


def cmd_list(args) -> int:
    entries = reg.load()
    if not entries:
        print("registry empty — run: scripts/registry.py seed")
        return 0
    print(f"{'id':26s} {'family':16s} {'domain':15s} "
          f"{'median':>7s} {'pcs/g':>6s} {'pp/pc':>6s} {'thrash':>6s} src")
    for e in sorted(entries, key=lambda e: (e["created"], e["id"])):
        ev = e["eval"]
        def f(x, w, p=1): return (f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}")
        print(f"{e['id']:26s} {e['family']:16s} {e['domain']:15s} "
              f"{f(ev.get('median_lines'),7)} {f(ev.get('pieces_per_game'),6)} "
              f"{f(ev.get('presses_per_piece'),6,2)} {f(ev.get('thrash'),6,2)} "
              f"{ev.get('source','?')}")
    return 0


def cmd_add(args) -> int:
    entries = reg.load()
    entry = reg.make_entry(
        id=args.id, created=args.created or reg._utc_now(), family=args.family,
        summary=args.summary, git_commit=args.git_commit or "", checkpoint=args.checkpoint,
        domain=args.domain,
        eval=reg.make_eval(median_lines=args.median, mean_lines=args.mean,
                           pieces_per_game=args.pieces_per_game, source=args.source),
        notes=args.notes,
    )
    errs = reg.validate_entry(entry)
    if errs:
        raise SystemExit("invalid entry:\n  " + "\n  ".join(errs))
    reg.upsert(entries, entry)
    reg.save(entries)
    reg.write_demo_index(entries)
    print(f"added/updated {args.id} ({len(entries)} entries)")
    return 0


def cmd_eval(args) -> int:
    """Re-run closed-loop eval on one entry with the CURRENT renderer."""
    from tetris import bc
    from tetris.export import load_policynet

    entries = reg.load()
    entry = next((e for e in entries if e["id"] == args.id), None)
    if entry is None:
        raise SystemExit(f"no registry entry with id {args.id!r}")
    if entry["family"] == "v1-valuenet":
        raise SystemExit(f"{args.id} is a v1 ValueNet, not a pixel PolicyNet — "
                         f"cannot closed-loop-eval it under the pixel renderer.")
    ckpt = _ROOT / entry["checkpoint"]
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found (local-only weights): {ckpt}")
    if entry["domain"] != "gray-128":
        print(f"WARNING: {args.id} was trained on the {entry['domain']} render; "
              f"evaluating under the current gray-128 render is a DOMAIN MISMATCH "
              f"(numbers are not comparable to its historical eval).")
    device = bc.select_device(args.device)
    try:
        model = load_policynet(ckpt, device=device)
    except RuntimeError as ex:
        raise SystemExit(f"cannot load {ckpt} into the current PolicyNet "
                         f"(architecture predates the aux heads?): {ex}")
    seeds = list(range(bc.EVAL_SEED_BASE, bc.EVAL_SEED_BASE + args.games))
    ev = bc.evaluate_policy_rich(model, seeds, args.max_pieces, device)
    entry["eval"] = reg.make_eval(
        median_lines=ev["median_lines"], mean_lines=ev["mean_lines"],
        pieces_per_game=ev["pieces_per_game"], presses_per_piece=ev["presses_per_piece"],
        press_frac=ev["press_frac"], thrash=ev["thrash"], source="fresh-eval",
    )
    reg.save(entries)
    reg.write_demo_index(entries)
    print(f"{args.id}: median={ev['median_lines']} mean={ev['mean_lines']:.2f} "
          f"pcs/g={ev['pieces_per_game']:.1f} pp/piece={ev['presses_per_piece']:.2f} "
          f"thrash={ev['thrash']:.3f} (source=fresh-eval, updated)")
    return 0


def cmd_export_demo(args) -> int:
    entries = reg.load()
    if not entries:
        raise SystemExit("registry empty — run: scripts/registry.py seed")
    path = reg.write_demo_index(entries)
    print(f"wrote {path} ({len(entries)} entries, weights-free)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Model registry CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("seed", help="build the full lineage registry")
    p.add_argument("--diagnosis", default=str(_ROOT / "scratchpad" / "diagnosis.json"))
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("list", help="list registry entries")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("add", help="add/update one entry")
    p.add_argument("--id", required=True)
    p.add_argument("--family", required=True, choices=reg.FAMILIES)
    p.add_argument("--domain", required=True, choices=reg.DOMAINS)
    p.add_argument("--summary", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--git-commit", default="")
    p.add_argument("--created", default=None)
    p.add_argument("--median", type=float, default=None)
    p.add_argument("--mean", type=float, default=None)
    p.add_argument("--pieces-per-game", type=float, default=None)
    p.add_argument("--source", default="fresh-eval", choices=reg.EVAL_SOURCES)
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("eval", help="re-run closed-loop eval on an entry (current renderer)")
    p.add_argument("id")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--max-pieces", type=int, default=10000)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("export-demo", help="write demo/models/registry.json (weights-free)")
    p.set_defaults(func=cmd_export_demo)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
