"""Boolean masks with one convention throughout: True means "this query may attend to this key".

Every mask leaves this module 4-D, shaped (B|1, 1, T|1, S) to line up with attention scores
(B, H, T, S). Broadcasting is decided here and nowhere else, because a 2-D (B, S) mask handed
straight to the scores aligns with the trailing (T, S) axes and, when B == T, silently masks
queries by batch index instead of keys by padding.
"""

import torch
from torch import Tensor


def causal_mask(size: int, device: torch.device | None = None) -> Tensor:
    # Lower triangle including the diagonal: position t sees 0..t. diagonal=1 here would let
    # t see t+1, which trains beautifully and generates garbage.
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()


def pad_mask(tokens: Tensor, pad_id: int) -> Tensor:
    return tokens != pad_id


def attention_mask(pad: Tensor | None = None, causal: Tensor | None = None) -> Tensor | None:
    """Combine a (B, S) key padding mask and a (T, T) causal mask into (B, 1, T, S).

    Padding masks keys only. Padded *query* rows still attend to the real keys and produce
    finite junk; masking them too would create all-masked rows for no benefit, and the loss
    ignores those positions anyway.
    """
    if pad is None and causal is None:
        return None
    if pad is not None and pad.dim() != 2:
        raise ValueError(f"pad mask must be (B, S), got {tuple(pad.shape)}")
    if causal is not None and (causal.dim() != 2 or causal.size(0) != causal.size(1)):
        raise ValueError(f"causal mask must be (T, T), got {tuple(causal.shape)}")
    if pad is not None and causal is not None and pad.size(1) != causal.size(1):
        raise ValueError(
            f"causal mask is over {causal.size(1)} keys but pad mask over {pad.size(1)}; "
            "causal masking only makes sense for self-attention"
        )

    if pad is None:
        return causal[None, None]
    if causal is None:
        return pad[:, None, None, :]
    return pad[:, None, None, :] & causal[None, None]
