import math

import torch


def inverse_sqrt_factor(step: int, warmup: int, ramp: bool = True) -> float:
    """Multiplier on peak_lr at optimizer step `step` (1-based).

    With ramp: linear 0 -> 1 over `warmup` steps, then sqrt(warmup / step). This is Vaswani et
    al.'s d^-0.5 * min(step^-0.5, step * warmup^-1.5) with the constant pulled into peak_lr,
    peak_lr = (d * warmup)^-0.5 * factor, so the peak is set directly instead of implied.
    Without ramp: 1 until `warmup`, then the same decay. The two schedules are identical after
    `warmup`, so the ablation isolates the ramp alone.
    """
    step = max(step, 1)
    if step < warmup:
        return step / warmup if ramp else 1.0
    return math.sqrt(warmup / step)


def make_scheduler(optimizer: torch.optim.Optimizer, warmup: int, ramp: bool = True) -> torch.optim.lr_scheduler.LambdaLR:
    # LambdaLR evaluates the lambda at 0 for the first optimizer step; +1 makes it 1-based.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: inverse_sqrt_factor(i + 1, warmup, ramp))
