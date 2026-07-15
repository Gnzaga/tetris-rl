"""RNG and 7-bag tests (PLAN.md §2, §4)."""

from tetris.rng import Mulberry32, SevenBag


def test_mulberry32_is_deterministic():
    a = Mulberry32(12345)
    b = Mulberry32(12345)
    seq_a = [a.next_uint32() for _ in range(100)]
    seq_b = [b.next_uint32() for _ in range(100)]
    assert seq_a == seq_b


def test_mulberry32_outputs_are_uint32():
    r = Mulberry32(1)
    for _ in range(1000):
        v = r.next_uint32()
        assert 0 <= v <= 0xFFFFFFFF


def test_next_float_in_unit_interval():
    r = Mulberry32(7)
    for _ in range(1000):
        f = r.next_float()
        assert 0.0 <= f < 1.0


def test_bag_windows_contain_each_piece_once():
    # Every aligned 7-draw window (a full bag) contains each of [0..6] once.
    bag = SevenBag(Mulberry32(2024))
    draws = [bag.next_piece() for _ in range(7 * 50)]
    for w in range(0, len(draws), 7):
        window = sorted(draws[w : w + 7])
        assert window == [0, 1, 2, 3, 4, 5, 6], f"bad window at {w}: {window}"


def test_bag_is_seed_deterministic():
    a = SevenBag(Mulberry32(99))
    b = SevenBag(Mulberry32(99))
    assert [a.next_piece() for _ in range(200)] == [b.next_piece() for _ in range(200)]
