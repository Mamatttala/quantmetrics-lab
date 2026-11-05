"""Offline L2 PELT segmentation and separate two-sided sequential CUSUM alerts."""

import numpy as np

from app.core.errors import ValidationFailure


def pelt(x, penalty=10.0, minimum=10):
    n = len(x)
    if n > 20000:
        raise ValidationFailure(
            "PELT currently supports at most 20,000 observations per experiment."
        )
    if n < minimum:
        raise ValidationFailure("PELT requires at least one minimum-length segment.")
    # Each candidate owns centered moments relative to its first observation.
    # Welford's nonnegative increment avoids subtracting large squared sums;
    # local origins also protect late, low-variance segments after large jumps.
    means = np.zeros((n, x.shape[1]))
    costs = np.zeros(n)

    objective = np.full(n + 1, np.inf)
    objective[0] = -penalty
    parent = np.zeros(n + 1, dtype=int)
    # Delayed pruning: a candidate dominated at t is kept until t+minimum, so the
    # dominating split has enough room to create a valid subsequent segment.
    candidates = [0]
    expiry = {}
    for end in range(minimum, n + 1):
        new = end - minimum
        if new == 0 or new >= minimum:
            centered = x[new:end] - x[new]
            means[new] = centered.mean(axis=0)
            costs[new] = ((centered - means[new]) ** 2).sum()
        if new >= minimum:
            candidates.append(new)
        candidates = [s for s in candidates if expiry.get(s, n + 1) > end]
        starts = np.asarray(candidates, dtype=int)
        continuing = starts[starts != new]
        lengths = end - continuing
        delta = (x[end - 1] - x[continuing]) - means[continuing]
        means[continuing] += delta / lengths[:, None]
        costs[continuing] += (delta * delta).sum(axis=1) * ((lengths - 1) / lengths)
        if not np.isfinite(costs[starts]).all():
            raise ValidationFailure("PELT segment SSE exceeds finite floating-point range.")
        values = objective[starts] + costs[starts] + penalty
        best = int(values.argmin())
        objective[end] = values[best]
        parent[end] = starts[best]
        for s, value in zip(starts, values, strict=True):
            if value - penalty > objective[end]:
                expiry.setdefault(int(s), end + minimum)
    ends = [n]
    while parent[ends[-1]] > 0:
        ends.append(int(parent[ends[-1]]))
    boundaries = sorted(ends)
    states = np.empty(n, dtype=int)
    start = 0
    for state, end in enumerate(boundaries):
        states[start:end] = state
        start = end
    return states, boundaries[:-1], float(objective[n])


def cusum(x, center, scale, threshold=5.0, drift=0.5, minimum=5):
    high = np.zeros(x.shape[1])
    low = np.zeros(x.shape[1])
    state = 0
    last = -minimum
    states, alerts, magnitudes = [], [], []
    for i, row in enumerate((x - center) / scale):
        high = np.maximum(0, high + row - drift)
        low = np.maximum(0, low - row - drift)
        magnitude = float(max(high.max(), low.max()))
        alert = magnitude > threshold and i - last >= minimum
        if alert:
            alerts.append(i)
            state += 1
            last = i
            high[:] = 0
            low[:] = 0
        states.append(state)
        magnitudes.append(magnitude)
    return np.asarray(states), alerts, magnitudes
