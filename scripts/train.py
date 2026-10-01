import json
import random
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from model.schedule import make_scheduler
from model.transformer import Transformer, TransformerConfig
from scripts.config import ROOT, load_config, pick_device
from tokenizer.bpe import BOS, EOS, PAD, BPE, train_bpe


def pad_batch(seqs: list[list[int]]) -> Tensor:
    out = torch.full((len(seqs), max(map(len, seqs))), PAD, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq)
    return out


def make_batches(src: list[list[int]], tgt: list[list[int]], max_tokens: int, rng: random.Random) -> list[list[int]]:
    """Index batches of similar target length, each at most `max_tokens` padded target tokens.

    Sorting by length keeps padding (wasted compute) low; shuffling within near-equal lengths
    first and then shuffling batch order keeps batches from being identical every epoch.
    """
    order = sorted(range(len(src)), key=lambda i: (len(tgt[i]), len(src[i]), rng.random()))
    batches, current, longest = [], [], 0
    for i in order:
        longest_if_added = max(longest, len(tgt[i]))
        if current and longest_if_added * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, longest_if_added = [], len(tgt[i])
        current.append(i)
        longest = longest_if_added
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


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


def build_model(model_cfg: dict, src_vocab: int, tgt_vocab: int) -> Transformer:
    return Transformer(TransformerConfig(src_vocab=src_vocab, tgt_vocab=tgt_vocab, **model_cfg))


def train_steps(
    model: Transformer,
    src: list[list[int]],
    tgt: list[list[int]],
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
        src_b = pad_batch([src[i] for i in idx]).to(device)
        tgt_in, tgt_out = shift(pad_batch([tgt[i] for i in idx]).to(device))
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
        history.append(record)
        if log_every and record["step"] % log_every == 0:
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()}), flush=True)
    return history


def encode_corpus(bpe: BPE, texts: list[str], bos: bool) -> list[list[int]]:
    return [bpe.encode(t, bos=bos, eos=True) for t in texts]


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
    model = build_model(cfg["model"], len(src_bpe), len(tgt_bpe))
    t = cfg["train"]
    history = train_steps(
        model, src, tgt, steps=t["steps"], max_tokens=t["max_tokens"], peak_lr=t["peak_lr"],
        warmup=t["warmup"], warmup_ramp=t["warmup_ramp"], smoothing=t["label_smoothing"],
        clip_norm=t["clip_norm"], device=device, seed=cfg["seed"], log_every=200,
    )
    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt / "model.pt")
    (ckpt / "history.json").write_text(json.dumps(history))


if __name__ == "__main__":
    main()
