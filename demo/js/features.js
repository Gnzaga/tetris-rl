// BCTS board features on bitboards (PLAN.md §2, features 3-8).
//
// A board is 20 rows of 10-bit ints (bit c set = column c filled), row 0 top,
// row 19 bottom. `boardFeatures` returns the 6 board-only BCTS features of a
// single post-clear board; the placement-dependent landing_height (1) and
// eroded_piece_cells (2) are produced by the engine. This is a faithful port of
// the pure-Python `board_features` reference — the numeric results are integers,
// so JS and Python agree exactly.

export const WIDTH = 10;
export const HEIGHT = 20;
export const FULL_ROW = (1 << WIDTH) - 1; // 0x3FF

function popcount(x) {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  return (Math.imul(x, 0x01010101) >> 24) & 0xff;
}

function rowTransitions(v) {
  // Sequence = [wall, b0..b9, wall]; count adjacent differences. Packed as
  // bit0=left wall, bits1..10=cells, bit11=right wall.
  const a = ((v << 1) | 0x801) & 0xfff;
  return popcount((a ^ (a >> 1)) & 0x7ff);
}

export function boardFeatures(rows) {
  if (rows.length !== HEIGHT) {
    throw new Error(`expected ${HEIGHT} rows, got ${rows.length}`);
  }

  // (3) row transitions.
  let rowTrans = 0;
  for (let r = 0; r < HEIGHT; r++) rowTrans += rowTransitions(rows[r]);

  // (4) column transitions: top border empty, floor filled.
  let colTrans = popcount(rows[0] & FULL_ROW);
  for (let r = 0; r < HEIGHT - 1; r++) {
    colTrans += popcount((rows[r] ^ rows[r + 1]) & FULL_ROW);
  }
  colTrans += popcount((rows[HEIGHT - 1] ^ FULL_ROW) & FULL_ROW);

  // (5)(7)(8): single top-down scan maintaining per-column filled counts.
  let holes = 0;
  let holeDepth = 0;
  let rowsWithHoles = 0;
  const colFilled = new Array(WIDTH).fill(0);
  let aboveOr = 0;
  for (let r = 0; r < HEIGHT; r++) {
    const row = rows[r];
    let anyHole = false;
    for (let c = 0; c < WIDTH; c++) {
      const bit = 1 << c;
      if (aboveOr & bit && !(row & bit)) {
        // Empty now, filled somewhere above => a hole.
        holes++;
        // hole_depth counts ALL filled cells above each hole in its column
        // (not just the contiguous run) — pinned by tests/test_features.py.
        holeDepth += colFilled[c];
        anyHole = true;
      }
      if (row & bit) colFilled[c]++;
    }
    if (anyHole) rowsWithHoles++;
    aboveOr |= row;
  }

  // (6) cumulative wells: empty cell whose left & right neighbors are filled
  // (walls filled); triangular sum over maximal vertical runs.
  let cumWells = 0;
  const runLen = new Array(WIDTH).fill(0);
  for (let r = 0; r < HEIGHT; r++) {
    const row = rows[r];
    const left = ((row << 1) | 1) & FULL_ROW;
    const right = ((row >> 1) | (1 << (WIDTH - 1))) & FULL_ROW;
    const well = ~row & left & right & FULL_ROW;
    for (let c = 0; c < WIDTH; c++) {
      if (well & (1 << c)) {
        runLen[c]++;
        cumWells += runLen[c];
      } else {
        runLen[c] = 0;
      }
    }
  }

  return [rowTrans, colTrans, holes, cumWells, holeDepth, rowsWithHoles];
}
