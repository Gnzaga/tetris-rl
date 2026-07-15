"""Deterministic RNG and 7-bag randomizer (PLAN.md §2).

`Mulberry32` is a bit-exact port of the JS `mulberry32` generator: every step is
masked to 32 bits so Python matches JS `>>> 0` semantics exactly. `SevenBag`
implements the frozen Fisher-Yates shuffle so both engines draw identical piece
sequences from a given seed.
"""

from __future__ import annotations

_U32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """32-bit integer multiply, matching JS `Math.imul` low-32-bit result."""
    return (a * b) & _U32


class Mulberry32:
    """mulberry32 PRNG seeded with a uint32 (PLAN.md §2)."""

    __slots__ = ("state",)

    def __init__(self, seed: int):
        self.state = seed & _U32

    def next_uint32(self) -> int:
        a = (self.state + 0x6D2B79F5) & _U32
        self.state = a
        t = _imul(a ^ (a >> 15), a | 1)
        t = ((t + _imul(t ^ (t >> 7), t | 61)) & _U32) ^ t
        t &= _U32
        return (t ^ (t >> 14)) & _U32

    def next_float(self) -> float:
        """Uniform float in [0, 1): next_uint32() / 2**32."""
        return self.next_uint32() / 4294967296.0

    def clone(self) -> "Mulberry32":
        r = Mulberry32(0)
        r.state = self.state
        return r


class SevenBag:
    """7-bag randomizer over piece indices [0..6] (PLAN.md §2).

    Refills a bag with [0..6], Fisher-Yates shuffles it using the frozen loop,
    and draws from the front.
    """

    __slots__ = ("rng", "_bag")

    def __init__(self, rng: Mulberry32):
        self.rng = rng
        self._bag: list[int] = []

    def _refill(self) -> None:
        bag = [0, 1, 2, 3, 4, 5, 6]
        for i in range(6, 0, -1):
            j = int(self.rng.next_float() * (i + 1))
            bag[i], bag[j] = bag[j], bag[i]
        self._bag = bag

    def next_piece(self) -> int:
        if not self._bag:
            self._refill()
        return self._bag.pop(0)
