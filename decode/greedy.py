import torch
from torch import Tensor

from model.transformer import Transformer
from tokenizer.bpe import BOS, EOS, PAD


@torch.no_grad()
def greedy_decode(
    model: Transformer, src: Tensor, max_len_a: float, max_len_b: int
) -> tuple[list[list[int]], list[bool]]:
    """Token ids per sentence (no BOS, no EOS) and whether each one emitted EOS before its cap.

    The cap is per sentence, a * src_len + b target tokens, so one long source in a batch does
    not let the short ones run on. The whole prefix is re-decoded each step (no KV cache):
    O(T^2) per sentence, fine for Tatoeba lengths.
    """
    memory, src_pad = model.encode(src)
    batch = src.size(0)
    caps = (max_len_a * src_pad.sum(1) + max_len_b).long()
    ys = torch.full((batch, 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(batch, dtype=torch.bool, device=src.device)
    finished = torch.zeros_like(done)
    for step in range(int(caps.max())):
        logits = model.decode(ys, memory, src_pad)[0][:, -1]
        nxt = torch.where(done, PAD, logits.argmax(-1))
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        finished |= ~done & (nxt == EOS)
        done |= (nxt == EOS) | (step + 1 >= caps)
        if done.all():
            break
    outputs = []
    for row in ys[:, 1:].tolist():
        out = []
        for tok in row:
            if tok in (EOS, PAD):
                break
            out.append(tok)
        outputs.append(out)
    return outputs, finished.tolist()
