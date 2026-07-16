"""Durable model registry — bookmarks for every trained iteration (user request).

The pixel-agent work shipped many checkpoints across six documented attempts
(``.superpowers/sdd/v2-phase-D-debug-report.md``); ``runs/`` is gitignored, so the
progression is invisible after a clone. This module is a small JSON registry that
BOOKMARKS each iteration with its lineage, git commit, obs/render domain and a
closed-loop eval summary, so "progress over time" survives.

Two files (both written by :func:`save`):

* ``runs/registry.json`` — the canonical registry (local; references local
  checkpoint paths under the gitignored ``runs/``).
* ``shared/model_registry.json`` — a committed copy of the INDEX so the history
  survives a fresh clone. It carries the SAME metadata; **the weights themselves
  are local-only** (the ``checkpoint`` paths point into the gitignored ``runs/``
  and ``runs/archive/`` — documented in the file's ``note`` field).

Entry schema (validated by :func:`validate_entry`, tested):
    id, created (ISO), family, summary, git_commit, checkpoint (local path),
    domain (obs/render domain), eval{median_lines, mean_lines, pieces_per_game,
    presses_per_piece, press_frac, thrash, source}, notes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = _ROOT / "runs" / "registry.json"
SHARED_INDEX = _ROOT / "shared" / "model_registry.json"
DEMO_INDEX = _ROOT / "demo" / "models" / "registry.json"

SCHEMA_VERSION = 1
FAMILIES = ("v1-valuenet", "pixel-bc", "pixel-ppo", "mini-experiment")
DOMAINS = ("camouflage-255", "gray-128", "board-int")
EVAL_SOURCES = ("historical-metrics", "fresh-eval")

_SHARED_NOTE = (
    "Committed INDEX of the model registry (durable across clones). The "
    "checkpoint WEIGHTS are local-only: `checkpoint` paths reference the "
    "gitignored runs/ (and runs/archive/) trees. Regenerate weights locally or "
    "re-run scripts/registry.py to refresh eval numbers."
)

_EVAL_KEYS = ("median_lines", "mean_lines", "pieces_per_game",
              "presses_per_piece", "press_frac", "thrash", "source")


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def make_eval(median_lines=None, mean_lines=None, pieces_per_game=None,
              presses_per_piece=None, press_frac=None, thrash=None,
              source="historical-metrics") -> dict:
    return {
        "median_lines": median_lines,
        "mean_lines": mean_lines,
        "pieces_per_game": pieces_per_game,
        "presses_per_piece": presses_per_piece,
        "press_frac": press_frac,
        "thrash": thrash,
        "source": source,
    }


def make_entry(id, created, family, summary, git_commit, checkpoint, domain,
               eval, notes="") -> dict:
    return {
        "id": id,
        "created": created,
        "family": family,
        "summary": summary,
        "git_commit": git_commit,
        "checkpoint": checkpoint,
        "domain": domain,
        "eval": eval,
        "notes": notes,
    }


def validate_entry(e: dict) -> list[str]:
    """Return a list of schema-violation strings (empty == valid)."""
    errs: list[str] = []
    for k in ("id", "created", "family", "summary", "git_commit", "checkpoint",
              "domain", "eval", "notes"):
        if k not in e:
            errs.append(f"missing key: {k}")
    if errs:
        return errs
    if not isinstance(e["id"], str) or not e["id"]:
        errs.append("id must be a non-empty string")
    if e["family"] not in FAMILIES:
        errs.append(f"family {e['family']!r} not in {FAMILIES}")
    if e["domain"] not in DOMAINS:
        errs.append(f"domain {e['domain']!r} not in {DOMAINS}")
    try:
        datetime.fromisoformat(e["created"])
    except (ValueError, TypeError):
        errs.append(f"created {e['created']!r} is not ISO-8601")
    ev = e["eval"]
    if not isinstance(ev, dict):
        errs.append("eval must be a dict")
    else:
        for k in _EVAL_KEYS:
            if k not in ev:
                errs.append(f"eval missing key: {k}")
        if ev.get("source") not in EVAL_SOURCES:
            errs.append(f"eval.source {ev.get('source')!r} not in {EVAL_SOURCES}")
    return errs


def validate(entries: list[dict]) -> list[str]:
    """Validate a list; also enforces unique ids. Returns all error strings."""
    errs: list[str] = []
    seen = set()
    for e in entries:
        eid = e.get("id", "?")
        for msg in validate_entry(e):
            errs.append(f"[{eid}] {msg}")
        if eid in seen:
            errs.append(f"[{eid}] duplicate id")
        seen.add(eid)
    return errs


# --------------------------------------------------------------------------
# Load / save
# --------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path: str | Path | None = None,
         shared_path: str | Path | None = None) -> list[dict]:
    """Load the registry entries list.

    Falls back to the COMMITTED index (``shared/model_registry.json``) when the
    canonical ``runs/registry.json`` is absent — on a fresh clone (runs/ is
    gitignored) the lineage must be reconstituted from the committed copy, so
    that the first post-clone ``save``/``register_from_run`` merges onto the
    full history instead of truncating it. Returns [] only when neither exists.
    (Paths default to the module's REGISTRY_PATH / SHARED_INDEX at call time.)
    """
    path = REGISTRY_PATH if path is None else path
    shared_path = SHARED_INDEX if shared_path is None else shared_path
    for p in (Path(path), Path(shared_path)):
        if p.exists():
            doc = json.loads(p.read_text())
            return doc.get("entries", []) if isinstance(doc, dict) else doc
    return []


def _write_doc(entries: list[dict], path: Path, note: str | None = None) -> None:
    """Write one registry document. If ``path`` already holds the SAME entry
    set, the file is left untouched (its ``generated`` timestamp is preserved)
    so the committed index never churns on no-op rewrites."""
    if path.exists():
        try:
            old = json.loads(path.read_text())
            if isinstance(old, dict) and old.get("entries") == entries:
                return
        except (json.JSONDecodeError, OSError):
            pass
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated": _utc_now(),
        "note": note or "Model registry — durable iteration bookmarks.",
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def save(entries: list[dict], path: str | Path | None = None,
         also_shared: bool = True) -> None:
    """Write ``runs/registry.json`` and (default) the committed
    ``shared/model_registry.json`` index copy. Raises on schema violations.
    Files whose entry set is unchanged are left untouched (no timestamp churn)."""
    path = REGISTRY_PATH if path is None else path
    errs = validate(entries)
    if errs:
        raise ValueError("registry validation failed:\n  " + "\n  ".join(errs))
    entries = sorted(entries, key=lambda e: (e["created"], e["id"]))
    _write_doc(entries, Path(path))
    if also_shared:
        _write_doc(entries, SHARED_INDEX, note=_SHARED_NOTE)


def upsert(entries: list[dict], entry: dict) -> list[dict]:
    """Insert ``entry`` or replace the existing one with the same id (in place)."""
    for i, e in enumerate(entries):
        if e["id"] == entry["id"]:
            entries[i] = entry
            return entries
    entries.append(entry)
    return entries


# --------------------------------------------------------------------------
# Demo index (weights-free)
# --------------------------------------------------------------------------


def demo_index(entries: list[dict]) -> list[dict]:
    """Compact, weights-free records for the demo's Model History panel. Drops
    the local checkpoint path; keeps lineage + eval for the chart/labels."""
    return [
        {
            "id": e["id"],
            "created": e["created"],
            "family": e["family"],
            "label": e["summary"],
            "domain": e["domain"],
            "eval": {k: e["eval"].get(k) for k in _EVAL_KEYS},
            "notes": e.get("notes", ""),
        }
        for e in sorted(entries, key=lambda e: (e["created"], e["id"]))
    ]


def register_from_run(run_dir: str | Path, family: str, domain: str,
                      ckpt_name: str, summary: str, notes: str = "",
                      entry_id: str | None = None) -> dict | None:
    """Auto-register a trainer's FINAL checkpoint (post-run hook for
    train_bc.py / train_ppo.py). Reads the run's ``config.json`` for the git
    commit and its last closed-loop eval row from ``metrics.jsonl``; upserts an
    entry into ``runs/registry.json`` + the committed index. Never raises — a
    bookkeeping failure must not fail a training run (returns None on error)."""
    try:
        run_dir = Path(run_dir)
        cfg = json.loads((run_dir / "config.json").read_text())
        rows = []
        mpath = run_dir / "metrics.jsonl"
        if mpath.exists():
            for line in mpath.read_text().splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        evals = [r for r in rows if r.get("eval_median_lines") is not None]
        last = evals[-1] if evals else {}
        eid = entry_id or f"{run_dir.name}_final"
        ckpt = run_dir / "checkpoints" / f"{ckpt_name}.pt"
        entry = make_entry(
            id=eid, created=cfg.get("created_at") or _utc_now(), family=family,
            summary=summary, git_commit=cfg.get("git_commit") or "",
            checkpoint=str(ckpt.relative_to(_ROOT)) if ckpt.is_relative_to(_ROOT) else str(ckpt),
            domain=domain,
            eval=make_eval(median_lines=last.get("eval_median_lines"),
                           mean_lines=last.get("eval_mean_lines"),
                           pieces_per_game=last.get("eval_pieces_per_game"),
                           source="historical-metrics"),
            notes=notes or f"Auto-registered by the {run_dir.name} trainer post-run hook.",
        )
        entries = load()
        upsert(entries, entry)
        save(entries)
        return entry
    except Exception as ex:  # pragma: no cover - defensive bookkeeping guard
        print(f"[registry] auto-register skipped ({ex})")
        return None


def write_demo_index(entries: list[dict], path: str | Path = DEMO_INDEX) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated": _utc_now(),
        "note": "Weights-free registry index for the demo Model History panel.",
        "entries": demo_index(entries),
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path
