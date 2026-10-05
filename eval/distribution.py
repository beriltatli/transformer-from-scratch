import math

import numpy as np


def summarize(scores: list[float]) -> dict[str, float]:
    """Median, quartiles and the worst decile of sentence-level scores (higher = better).

    worst_decile_mean is the mean of the lowest ceil(n / 10) scores, the sentences a corpus
    average hides; p10 is the boundary of that decile (numpy linear interpolation).
    """
    x = np.sort(np.asarray(scores, dtype=float))
    q1, median, q3, p10 = np.percentile(x, [25, 50, 75, 10])
    k = math.ceil(len(x) / 10)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p10": float(p10),
        "worst_decile_mean": float(x[:k].mean()),
    }
