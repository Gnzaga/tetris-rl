"""Assemble ``demo/models/`` for the browser demo (PLAN.md §9).

Exports every ValueNet milestone checkpoint of a TD run to ONNX (opset 17,
dynamic batch), gates each on the PyTorch<->onnxruntime parity check, then writes
``demo/models/manifest.json`` describing every demo agent (random, Dellacherie,
the CEM-optimized linear agent, and the ValueNet milestones) plus a self-test and
the training curve. Finally it vendors onnxruntime-web into ``demo/vendor/`` so
the demo runs fully offline with no server-side inference.

The demo is NOT shipped pre-built: ``demo/models/`` and ``demo/vendor/`` are
gitignored, so run this locally before serving the demo::

    python scripts/export_demo.py --td-run td_v1 --cem-run cem_v1

The same command works unchanged against the smoke runs used to gate this phase
(``--td-run td_smoke --cem-run cem_smoke``). Pass ``--skip-vendor`` to skip the
npm/onnxruntime-web vendoring step (e.g. when the runtime is already vendored).

Vendoring runs ``npm --prefix demo i onnxruntime-web``, copies the minified
wasm-only ESM bundle plus the ``.wasm`` binaries it references into
``demo/vendor/``, then deletes ``demo/node_modules``. The demo sets
``ort.env.wasm.wasmPaths = './vendor/'`` so it loads them offline.
"""

import _pathshim  # noqa: F401
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tetris.agents import DELLACHERIE_WEIGHTS
from tetris.export import PARITY_BOARDS, PARITY_TOL, export_checkpoint, onnx_values

_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _ROOT / "runs"
DEMO_DIR = _ROOT / "demo"
DEFAULT_OUT = DEMO_DIR / "models"
VENDOR_DIR = DEMO_DIR / "vendor"

ENGINE_VERSION = "1"

# ValueNet milestone percent -> human label (PLAN.md §9 manifest example).
_NN_LABELS = {
    0: "ValueNet — untrained",
    10: "ValueNet — 10% trained",
    30: "ValueNet — 30% trained",
    60: "ValueNet — 60% trained",
    100: "ValueNet — fully trained",
}

# The wasm-only ESM bundle is self-contained (inlines its worker/glue) and
# references only the plain simd-threaded wasm binary — the minimal offline set.
_ORT_BUNDLE = "ort.wasm.bundle.min.mjs"


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _nearest_eval(metrics: list[dict[str, Any]], pieces_trained: int) -> dict[str, Any]:
    """Eval stats from the metrics row whose pieces_trained is nearest target."""
    evals = [m for m in metrics if m.get("eval_median_lines") is not None]
    if not evals:
        return {}
    row = min(evals, key=lambda m: abs((m.get("pieces_trained") or 0) - pieces_trained))
    return {
        "median_lines": row.get("eval_median_lines"),
        "mean_lines": row.get("eval_mean_lines"),
        "p10_lines": row.get("eval_p10_lines"),
        "pieces_per_game": row.get("eval_pieces_per_game"),
        "pieces_trained": row.get("pieces_trained"),
    }


def _curve(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Training curve for the demo chart: ascending (pieces_trained, median)."""
    pts = [
        {
            "pieces_trained": int(m["pieces_trained"]),
            "eval_median_lines": m["eval_median_lines"],
        }
        for m in metrics
        if m.get("eval_median_lines") is not None and m.get("pieces_trained") is not None
    ]
    pts.sort(key=lambda p: p["pieces_trained"])
    return pts


def _cem_eval(cem_final: dict[str, Any]) -> dict[str, Any]:
    """CEM eval object: the 200k-cap final eval, flagged as capped (PLAN.md §9)."""
    ev = cem_final.get("eval", {})
    hit_cap = int(ev.get("games_hit_cap", 0) or 0)
    return {
        "mean_lines": ev.get("mean_lines"),
        "median_lines": ev.get("median_lines"),
        "p10_lines": ev.get("p10_lines"),
        "games": ev.get("games"),
        "games_hit_cap": hit_cap,
        "max_pieces": ev.get("max_pieces"),
        "capped": hit_cap > 0,
    }


def _selftest(final_onnx: Path, seed: int = 12345) -> dict[str, Any]:
    """3 random boards (20 uint16 rows each) + their FINAL-model onnx values."""
    import numpy as np

    rng = np.random.default_rng(seed)
    boards = rng.integers(0, 1 << 10, size=(3, 20)).astype(np.uint16)
    values = onnx_values(final_onnx, boards)
    return {
        "boards": [[int(v) for v in row] for row in boards],
        "expected_values": [float(v) for v in values],
    }


def build_manifest(
    td_run: str,
    cem_run: str,
    out_dir: Path = DEFAULT_OUT,
    runs_dir: Path = RUNS_DIR,
    parity_boards: int = PARITY_BOARDS,
    parity_tol: float = PARITY_TOL,
    verbose: bool = True,
) -> dict[str, Any]:
    """Export milestones + write ``<out_dir>/manifest.json``; returns the manifest.

    Fails hard if any milestone's ONNX export misses the parity gate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    td_dir = runs_dir / td_run
    cem_dir = runs_dir / cem_run
    td_config = _read_json(td_dir / "config.json")
    milestones = td_config.get("milestones")
    if not milestones:
        raise SystemExit(
            f"{td_dir}/config.json has no 'milestones' block — is this a finished TD run?"
        )
    metrics = _read_metrics(td_dir)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    nn_agents: list[dict[str, Any]] = []
    final_onnx: Path | None = None
    for pct in sorted(int(p) for p in milestones):
        info = milestones[str(pct)]
        ckpt = td_dir / "checkpoints" / f"{info['checkpoint']}.pt"
        nn_id = f"nn_{pct:03d}"
        onnx_name = f"{nn_id}.onnx"
        report = export_checkpoint(
            ckpt,
            out_dir / onnx_name,
            n_boards=parity_boards,
            tol=parity_tol,
        )
        _log(
            f"  {nn_id}: {info['checkpoint']}.pt -> {onnx_name} "
            f"(parity max|d|={report['max_abs_diff']:.2e})"
        )
        nn_agents.append(
            {
                "id": nn_id,
                "type": "onnx",
                "label": _NN_LABELS.get(pct, f"ValueNet — {pct}% trained"),
                "path": onnx_name,
                "eval": _nearest_eval(metrics, int(info["pieces_trained"])),
            }
        )
        if pct == 100:
            final_onnx = out_dir / onnx_name
    if final_onnx is None:
        raise SystemExit("no 100% milestone found — cannot build self-test")

    cem_final = _read_json(cem_dir / "checkpoints" / "cem_final.json")
    cem_weights = [float(w) for w in cem_final["weights"]]
    if len(cem_weights) != 8:
        raise SystemExit(f"cem weights must be length 8, got {len(cem_weights)}")

    agents: list[dict[str, Any]] = [
        {"id": "random", "type": "random", "label": "Random"},
        {
            "id": "dellacherie",
            "type": "linear",
            "label": "Dellacherie (hand-tuned, 2003)",
            "weights": [float(w) for w in DELLACHERIE_WEIGHTS],
        },
        {
            "id": "cem",
            "type": "linear",
            "label": "CEM-optimized (this repo)",
            "weights": cem_weights,
            "eval": _cem_eval(cem_final),
        },
        *nn_agents,
    ]

    manifest = {
        "engine_version": ENGINE_VERSION,
        "td_run": td_run,
        "cem_run": cem_run,
        "agents": agents,
        "selftest": _selftest(final_onnx),
        "curve": _curve(metrics),
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    _log(f"wrote {out_dir / 'manifest.json'} ({len(agents)} agents, "
         f"{len(manifest['curve'])} curve points)")
    return manifest


def vendor_onnxruntime_web(verbose: bool = True) -> list[Path]:
    """npm-install onnxruntime-web, copy the ESM bundle + wasm into demo/vendor/.

    Deletes ``demo/node_modules`` afterward. ``demo/vendor/`` is gitignored, so
    this reproduces the runtime on demand. Returns the copied file paths.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    _log("vendoring onnxruntime-web (npm --prefix demo i onnxruntime-web)...")
    subprocess.run(
        ["npm", "--prefix", str(DEMO_DIR), "i", "onnxruntime-web"],
        check=True,
    )
    dist = DEMO_DIR / "node_modules" / "onnxruntime-web" / "dist"
    bundle = dist / _ORT_BUNDLE
    if not bundle.exists():
        raise SystemExit(f"expected ESM bundle not found: {bundle}")

    copied: list[Path] = []
    dst_bundle = VENDOR_DIR / _ORT_BUNDLE
    shutil.copy2(bundle, dst_bundle)
    copied.append(dst_bundle)

    # Copy every .wasm the bundle references (fall back to all wasm binaries).
    wasm_names = sorted(set(re.findall(r"ort-wasm[\w.-]*\.wasm", bundle.read_text())))
    wasm_paths = [dist / n for n in wasm_names] or sorted(dist.glob("*.wasm"))
    for w in wasm_paths:
        if w.exists():
            dst = VENDOR_DIR / w.name
            shutil.copy2(w, dst)
            copied.append(dst)

    shutil.rmtree(DEMO_DIR / "node_modules", ignore_errors=True)
    (DEMO_DIR / "package-lock.json").unlink(missing_ok=True)

    _log("vendored: " + ", ".join(p.name for p in copied))
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build demo/models/ + vendor onnxruntime-web")
    ap.add_argument("--td-run", default="td_v1", help="TD run name under runs/")
    ap.add_argument("--cem-run", default="cem_v1", help="CEM run name under runs/")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="output models dir")
    ap.add_argument("--skip-vendor", action="store_true", help="skip onnxruntime-web vendoring")
    ap.add_argument("--parity-boards", type=int, default=PARITY_BOARDS)
    args = ap.parse_args(argv)

    print(f"exporting demo from td_run={args.td_run}, cem_run={args.cem_run}")
    build_manifest(
        td_run=args.td_run,
        cem_run=args.cem_run,
        out_dir=Path(args.out_dir),
        parity_boards=args.parity_boards,
    )
    if args.skip_vendor:
        print("skipping vendoring (--skip-vendor)")
    else:
        vendor_onnxruntime_web()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
