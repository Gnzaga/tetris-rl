"""Validation of shared/pieces.json — the frozen tetromino rotation tables (PLAN.md §2).

These tests guard the single source of truth loaded by both the Python and JS
engines. They assert structural invariants (4 cells per rotation, bounding-box
dimensions consistent with the cell offsets, distinct rotation states only) and
the frozen per-piece rotation counts and piece order.
"""

import json
from pathlib import Path

import pytest

PIECES_PATH = Path(__file__).resolve().parent.parent / "shared" / "pieces.json"

# Frozen spec (PLAN.md §2): piece order defines indices 0..6, with the number of
# distinct rotation states each piece has.
EXPECTED_PIECE_ORDER = ["I", "O", "T", "S", "Z", "J", "L"]
EXPECTED_ROTATION_COUNTS = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}


@pytest.fixture(scope="module")
def data():
    with open(PIECES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pieces(data):
    return data["pieces"]


def test_exactly_seven_pieces(pieces):
    assert len(pieces) == 7


def test_piece_order(pieces):
    assert [p["name"] for p in pieces] == EXPECTED_PIECE_ORDER


def test_rotation_counts(pieces):
    for piece in pieces:
        name = piece["name"]
        assert len(piece["rotations"]) == EXPECTED_ROTATION_COUNTS[name], (
            f"piece {name} has {len(piece['rotations'])} rotations, "
            f"expected {EXPECTED_ROTATION_COUNTS[name]}"
        )


def _iter_rotations(pieces):
    for piece in pieces:
        for idx, rot in enumerate(piece["rotations"]):
            yield piece["name"], idx, rot


def test_each_rotation_has_four_cells(pieces):
    for name, idx, rot in _iter_rotations(pieces):
        cells = rot["cells"]
        assert len(cells) == 4, f"{name}[{idx}] has {len(cells)} cells, expected 4"


def test_cells_are_unique_within_rotation(pieces):
    for name, idx, rot in _iter_rotations(pieces):
        cells = [tuple(c) for c in rot["cells"]]
        assert len(set(cells)) == 4, f"{name}[{idx}] has duplicate cells: {cells}"


def test_bounding_box_consistent_with_offsets(pieces):
    for name, idx, rot in _iter_rotations(pieces):
        rows = [c[0] for c in rot["cells"]]
        cols = [c[1] for c in rot["cells"]]
        assert min(rows) == 0, f"{name}[{idx}] min row {min(rows)} != 0"
        assert min(cols) == 0, f"{name}[{idx}] min col {min(cols)} != 0"
        assert max(rows) + 1 == rot["height"], (
            f"{name}[{idx}] max row+1 {max(rows) + 1} != height {rot['height']}"
        )
        assert max(cols) + 1 == rot["width"], (
            f"{name}[{idx}] max col+1 {max(cols) + 1} != width {rot['width']}"
        )


def test_dimensions_fit_board_width(pieces):
    # Every rotation must be placeable on a 10-wide board (PLAN.md §2 board spec).
    for name, idx, rot in _iter_rotations(pieces):
        assert 1 <= rot["width"] <= 10, f"{name}[{idx}] width {rot['width']} out of range"
        assert 1 <= rot["height"] <= 20, f"{name}[{idx}] height {rot['height']} out of range"


def test_rotation_states_are_distinct(pieces):
    # No two rotation states of the same piece may have identical cell sets
    # (the tables store only distinct rotations).
    for piece in pieces:
        seen = set()
        for idx, rot in enumerate(piece["rotations"]):
            key = frozenset(tuple(c) for c in rot["cells"])
            assert key not in seen, (
                f"piece {piece['name']} rotation {idx} duplicates an earlier rotation"
            )
            seen.add(key)
