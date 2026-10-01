"""Ten sentence pairs to near-zero loss and exact regeneration, through the real training path.

A model that cannot memorise ten pairs has a bug; finding it here costs seconds.
"""

from dataclasses import dataclass

import torch

from decode.greedy import greedy_decode
from scripts.config import ROOT, load_config
from scripts.train import build_model, encode_corpus, pad_batch, train_steps
from tokenizer.bpe import train_bpe

PAIRS = [
    ("I can't come tomorrow.", "Yarın gelemeyeceğim."),
    ("Where is the train station?", "Tren istasyonu nerede?"),
    ("She is reading a book in the garden.", "Bahçede kitap okuyor."),
    ("We have been waiting for two hours.", "İki saattir bekliyoruz."),
    ("Did you see my keys?", "Anahtarlarımı gördün mü?"),
    ("The children are playing outside.", "Çocuklar dışarıda oynuyor."),
    ("I don't understand this question.", "Bu soruyu anlamıyorum."),
    ("Tom bought a new car last week.", "Tom geçen hafta yeni bir araba aldı."),
    ("Could you close the window, please?", "Pencereyi kapatabilir misiniz, lütfen?"),
    ("It is very cold in Istanbul today.", "Bugün İstanbul'da hava çok soğuk."),
]


@dataclass
class OverfitResult:
    history: list[dict]
    hypotheses: list[str]
    references: list[str]


def run_overfit(cfg: dict) -> OverfitResult:
    o = cfg["overfit"]
    torch.manual_seed(cfg["seed"])
    src_texts, tgt_texts = [s for s, _ in PAIRS], [t for _, t in PAIRS]
    src_bpe = train_bpe(src_texts, o["vocab_size"])
    tgt_bpe = train_bpe(tgt_texts, o["vocab_size"])
    src = encode_corpus(src_bpe, src_texts, bos=False)
    tgt = encode_corpus(tgt_bpe, tgt_texts, bos=True)
    model_cfg = {**cfg["model"], **o["model"]}
    model = build_model(model_cfg, len(src_bpe), len(tgt_bpe))
    device = torch.device("cpu")
    # Label smoothing off and dropout off: both put a floor under the loss, and the claim
    # under test is "can drive the loss to zero".
    history = train_steps(
        model, src, tgt, steps=o["steps"], max_tokens=10_000, peak_lr=o["peak_lr"],
        warmup=o["warmup"], warmup_ramp=True, smoothing=0.0, clip_norm=1.0, device=device,
        seed=cfg["seed"],
    )
    model.eval()
    out, _ = greedy_decode(model, pad_batch(src).to(device), cfg["decode"]["max_len_a"], cfg["decode"]["max_len_b"])
    return OverfitResult(history, [tgt_bpe.decode(ids) for ids in out], tgt_texts)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = run_overfit(load_config())
    steps = [r["step"] for r in result.history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(steps, [r["nll"] for r in result.history])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("token NLL (nats, log scale)")
    ax.set_title("Overfitting ten sentence pairs")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "overfit_curve.png", dpi=150)

    for step in (1, 50, 100, 200, 300, len(steps)):
        r = result.history[step - 1]
        print(f"step {step:4d}  nll {r['nll']:.6f}  lr {r['lr']:.2e}")
    exact = sum(h == r for h, r in zip(result.hypotheses, result.references))
    for h, r in zip(result.hypotheses, result.references):
        print(("OK  " if h == r else "DIFF") + f"  {h}" + ("" if h == r else f"   (ref: {r})"))
    print(f"exact: {exact}/{len(result.references)}  wall: {result.history[-1]['seconds']:.1f}s")


if __name__ == "__main__":
    main()
