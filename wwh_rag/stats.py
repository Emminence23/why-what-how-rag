"""Paired comparison of two systems answered on the same questions."""
from math import comb


def mcnemar_exact(base_correct, ours_correct):
    """Two-sided exact McNemar test.

    Only questions where the systems disagree count: b = only baseline right,
    c = only ours right. Under 'no difference' each such question is a fair coin.
    Returns (b, c, p_value).
    """
    b = sum(1 for x, y in zip(base_correct, ours_correct) if x and not y)
    c = sum(1 for x, y in zip(base_correct, ours_correct) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)
