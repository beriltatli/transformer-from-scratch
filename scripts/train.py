import json
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from model.schedule import make_scheduler
from model.transformer import Transformer, TransformerConfig
from scripts.config import ROOT, load_config, pick_device
from tokenizer.bpe import BOS, EOS, PAD, BPE, train_bpe


# Lengths are padded up to a multiple of this. On MPS every new tensor shape compiles a new
# kernel graph (0.5-2 s each); bucketing keeps the shape set to a few dozen for the whole run.
LENGTH_MULTIPLE = 8


def bucket(length: int) -> int:
    return -(-length // LENGTH_MULTIPLE) * LENGTH_MULTIPLE


def pad_batch(seqs: list, length: int | None = None) -> Tensor:
    out = torch.full((len(seqs), length or max(map(len, seqs))), PAD, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
    return out


def make_batches(src: list, tgt: list, max_tokens: int, rng: random.Random) -> list[list[int]]:
    """Index batches grouped by (target bucket, source bucket), each with a fixed row count.

    A group's rows are all padded to the same bucket lengths, and the row count depends only
    on the group, so every batch in a group has one tensor shape. max_tokens bounds
    rows * max(target bucket, source bucket), i.e. padded positions, not real tokens.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(len(src)):
        groups.setdefault((bucket(len(tgt[i]) - 1), bucket(len(src[i]))), []).append(i)
    batches = []
    for (tgt_len, src_len), members in sorted(groups.items()):
        rng.shuffle(members)
        rows = max(1, max_tokens // max(tgt_len, src_len))
        batches.extend(members[k : k + rows] for k in range(0, len(members), rows))
    rng.shuffle(batches)
    return batches


def collate(src: list, tgt: list, idx: list[int]) -> tuple[Tensor, Tensor]:
    return (
        pad_batch([src[i] for i in idx], bucket(max(len(src[i]) for i in idx))),
        # +1: shift() drops one column, and the decoder input should land on the bucket length.
        pad_batch([tgt[i] for i in idx], bucket(max(len(tgt[i]) for i in idx) - 1) + 1),
    )


def label_smoothed_nll(logits: Tensor, target: Tensor, smoothing: float) -> tuple[Tensor, Tensor, int]:
    """Summed smoothed loss, summed plain NLL, and the number of non-pad target tokens.

    Smoothing as in Szegedy et al.: (1 - eps) on the gold token, eps spread uniformly over the
    vocabulary. The plain NLL is returned separately because perplexity must be computed from
    it; the smoothed loss has a floor above zero and is not a likelihood.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    real = target != PAD
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[real]
    uniform = -logp.mean(-1)[real]
    loss = (1 - smoothing) * nll + smoothing * uniform
    return loss.sum(), nll.sum(), int(real.sum())


def shift(tgt: Tensor) -> tuple[Tensor, Tensor]:
    """Sequences BOS y1 .. yn EOS -> decoder input BOS y1 .. yn and target y1 .. yn EOS.

    Padding of the shorter rows stays aligned: the input drops the last column, which for a
    padded row is PAD, not that row's EOS.
    """
    return tgt[:, :-1], tgt[:, 1:]


@torch.no_grad()
def evaluate_nll(model: Transformer, src: list[np.ndarray], tgt: list[np.ndarray], max_tokens: int, device: torch.device) -> float:
    """Teacher-forced token NLL in nats, no label smoothing, dropout off. exp() of it is perplexity."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for idx in make_batches(src, tgt, max_tokens, random.Random(0)):
        src_b, tgt_b = collate(src, tgt, idx)
        tgt_in, tgt_out = shift(tgt_b.to(device))
        _, nll, count = label_smoothed_nll(model(src_b.to(device), tgt_in), tgt_out, 0.0)
        total, n = total + nll.item(), n + count
    model.train(was_training)
    return total / n


def build_model(model_cfg: dict, src_vocab: int, tgt_vocab: int) -> Transformer:
    return Transformer(TransformerConfig(src_vocab=src_vocab, tgt_vocab=tgt_vocab, **model_cfg))


def train_steps(
    model: Transformer,
    src: list[np.ndarray],
    tgt: list[np.ndarray],
    *,
    steps: int,
    max_tokens: int,
    peak_lr: float,
    warmup: int,
    warmup_ramp: bool,
    smoothing: float,
    clip_norm: float | None,
    device: torch.device,
    seed: int,
    log_every: int = 0,
    valid: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
    eval_every: int = 0,
) -> list[dict]:
    """Train for `steps` optimizer steps and return one record per step."""
    rng = random.Random(seed)
    # Adam betas and eps from Vaswani et al.; beta2 = 0.98 rather than 0.999 makes the second
    # moment react faster to the large early gradients.
    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = make_scheduler(optimizer, warmup, warmup_ramp)
    model.to(device).train()
    history: list[dict] = []
    batches: list[list[int]] = []
    start = time.time()
    while len(history) < steps:
        if not batches:
            batches = make_batches(src, tgt, max_tokens, rng)
        idx = batches.pop()
        src_b, tgt_b = collate(src, tgt, idx)
        src_b = src_b.to(device)
        tgt_in, tgt_out = shift(tgt_b.to(device))
        loss, nll, n_tokens = label_smoothed_nll(model(src_b, tgt_in), tgt_out, smoothing)
        optimizer.zero_grad(set_to_none=True)
        (loss / n_tokens).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm or float("inf"))
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        record = {
            "step": len(history) + 1,
            "loss": loss.item() / n_tokens,
            "nll": nll.item() / n_tokens,
            "grad_norm": float(grad_norm),
            "lr": lr,
            "tokens": n_tokens,
            "seconds": time.time() - start,
        }
        if valid and eval_every and record["step"] % eval_every == 0:
            record["valid_nll"] = evaluate_nll(model, *valid, max_tokens, device)
        history.append(record)
        if log_every and record["step"] % log_every == 0:
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()}), flush=True)
    return history


def encode_corpus(bpe: BPE, texts: list[str], bos: bool) -> list[np.ndarray]:
    # int16 arrays, not lists of Python ints: ~2 bytes per token instead of ~36. On an 8 GB
    # unified-memory machine the lists alone pushed training into swap. int16 caps the vocab.
    if len(bpe) > np.iinfo(np.int16).max:
        raise ValueError(f"vocab {len(bpe)} does not fit int16")
    return [np.array(bpe.encode(t, bos=bos, eos=True), dtype=np.int16) for t in texts]


def load_split(data_dir: Path, name: str) -> tuple[list[str], list[str]]:
    return (data_dir / f"{name}.en").read_text().splitlines(), (data_dir / f"{name}.tr").read_text().splitlines()


def train_tokenizers(cfg: dict, data_dir: Path) -> tuple[BPE, BPE]:
    src_texts, tgt_texts = load_split(data_dir, "train")
    vocab = cfg["tokenizer"]["vocab_size"]
    out = []
    for lang, texts in (("en", src_texts), ("tr", tgt_texts)):
        path = data_dir / f"bpe_{lang}_{vocab}.json"
        if path.exists():
            out.append(BPE.load(path))
        else:
            bpe = train_bpe(texts, vocab, cfg["tokenizer"]["min_frequency"])
            bpe.save(path)
            out.append(bpe)
    return out[0], out[1]


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    data_dir = ROOT / cfg["data"]["dir"]
    src_bpe, tgt_bpe = train_tokenizers(cfg, data_dir)
    src_texts, tgt_texts = load_split(data_dir, "train")
    src = encode_corpus(src_bpe, src_texts, bos=False)
    tgt = encode_corpus(tgt_bpe, tgt_texts, bos=True)
    valid_en, valid_tr = load_split(data_dir, "valid")
    valid = (encode_corpus(src_bpe, valid_en, bos=False), encode_corpus(tgt_bpe, valid_tr, bos=True))
    model = build_model(cfg["model"], len(src_bpe), len(tgt_bpe))
    t = cfg["train"]
    history = train_steps(
        model, src, tgt, steps=t["steps"], max_tokens=t["max_tokens"], peak_lr=t["peak_lr"],
        warmup=t["warmup"], warmup_ramp=t["warmup_ramp"], smoothing=t["label_smoothing"],
        clip_norm=t["clip_norm"], device=device, seed=cfg["seed"], log_every=200,
        valid=valid, eval_every=t["eval_every"],
    )
    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt / "model.pt")
    (ckpt / "history.json").write_text(json.dumps(history))


if __name__ == "__main__":
    main()
