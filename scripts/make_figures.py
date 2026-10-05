"""README figures from the training history, the evaluation results and the corpus statistics.

Each figure is written twice, figures/<name>.light.png and figures/<name>.dark.png, so the README
can pick one per GitHub theme with <picture>. Run after scripts.train and scripts.evaluate.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.scores import sentence_scores
from scripts.config import ROOT, load_config
from scripts.train import load_split

# Two categorical slots (blue, orange), validated for colour-vision deficiency on both
# surfaces; ink and hairline greys for everything that is not data.
THEMES = {
    "light": {
        "surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "c1": "#2a78d6", "c2": "#eb6834", "dim": "#c3c2b7",
    },
    "dark": {
        "surface": "#0d1117", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "c1": "#3987e5", "c2": "#d95926", "dim": "#52514e",
    },
}
SYSTEMS = ["transformer", "retrieval", "dictionary", "copy_source"]
LABELS = {"transformer": "Transformer", "retrieval": "Retrieval", "dictionary": "Dictionary", "copy_source": "Copy source"}


def style(ax, t: dict, grid_axis: str = "y") -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def figure(t: dict, *args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def title(fig, t: dict, text: str, sub: str) -> None:
    fig.text(0.012, 0.975, text, color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.012, 0.915, sub, color=t["ink2"], fontsize=9.5, va="top")


def legend(ax, t: dict, **kwargs) -> None:
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def scores_by_system(results: dict, t: dict):
    fig, ax = figure(t, figsize=(8, 3.8))
    y = np.arange(len(SYSTEMS))[::-1]
    h = 0.34
    chrf = [results[s]["corpus"]["chrF++"]["score"] for s in SYSTEMS]
    bleu = [results[s]["corpus"]["BLEU"]["score"] for s in SYSTEMS]
    ax.barh(y + h / 2 + 0.01, chrf, h, color=t["c1"], label="chrF++ (characters + words)")
    ax.barh(y - h / 2 - 0.01, bleu, h, color=t["c2"], label="BLEU (whole words only)")
    for yi, c, b in zip(y, chrf, bleu):
        ax.text(c + 0.8, yi + h / 2, f"{c:.1f}", va="center", fontsize=9, color=t["ink"])
        ax.text(b + 0.8, yi - h / 2, f"{b:.1f}", va="center", fontsize=9, color=t["ink2"])
    ax.set_yticks(y, [LABELS[s] for s in SYSTEMS])
    ax.set_xlim(0, 100)
    ax.set_xlabel("corpus score on 2,000 test sentences (higher is better)")
    style(ax, t, "x")
    legend(ax, t, loc="lower right")
    title(fig, t, "Test-set scores", "Same 2,000 sentences, same references, sacrebleu 2.6.0")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def training_curve(history: list[dict], warmup: int, t: dict):
    fig, (top, bottom) = figure(t, 2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    steps = np.array([r["step"] for r in history])
    nll = np.array([r["nll"] for r in history])
    window = 200
    smooth = np.convolve(nll, np.ones(window) / window, mode="valid")
    top.plot(steps[window - 1 :], smooth, color=t["c1"], linewidth=2, label="training batches (200-step mean)")
    vs = [(r["step"], r["valid_nll"]) for r in history if "valid_nll" in r]
    top.plot(*zip(*vs), color=t["c2"], linewidth=2, marker="o", markersize=6,
             markeredgecolor=t["surface"], markeredgewidth=2, label="validation set (2,000 held-out pairs)")
    last_step, last = vs[-1]
    top.annotate(f"{last:.2f} nats\nperplexity {np.exp(last):.1f}", (last_step, last), xytext=(-8, 22),
                 textcoords="offset points", ha="right", fontsize=9, color=t["ink"])
    top.set_ylabel("loss per token (nats)")
    top.set_ylim(0, 8)
    style(top, t)
    legend(top, t, loc="upper right")

    bottom.plot(steps, [r["lr"] * 1e4 for r in history], color=t["muted"], linewidth=2)
    for ax in (top, bottom):
        ax.axvline(warmup, color=t["axis"], linewidth=1)
    bottom.text(warmup + 120, 6.0, "warm-up ends", fontsize=8.5, color=t["ink2"], va="top")
    bottom.set_ylabel("learning rate\n(×10⁻⁴)")
    bottom.set_xlabel("optimizer step")
    bottom.set_xlim(0, steps[-1])
    style(bottom, t)
    title(fig, t, "Training", f"{steps[-1]:,} steps, ~{sum(r['tokens'] for r in history) / 1e6:.0f}M target tokens, Apple M3 (MPS)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def near_duplicate_split(results: dict, t: dict):
    fig, ax = figure(t, figsize=(8, 3.8))
    y = np.arange(len(SYSTEMS))[::-1]
    h = 0.34
    other = [results[s]["sentence_chrF++"]["not_near_dup"]["median"] for s in SYSTEMS]
    dup = [results[s]["sentence_chrF++"]["near_dup"]["median"] for s in SYSTEMS]
    ax.barh(y + h / 2 + 0.01, other, h, color=t["c1"], label=f"no close match in training ({results['n_test'] - results['n_near_dup']:,})")
    ax.barh(y - h / 2 - 0.01, dup, h, color=t["c2"], label=f"close match in training ({results['n_near_dup']:,})")
    for yi, o, d in zip(y, other, dup):
        ax.text(o + 0.8, yi + h / 2, f"{o:.1f}", va="center", fontsize=9, color=t["ink"])
        ax.text(d + 0.8, yi - h / 2, f"{d:.1f}", va="center", fontsize=9, color=t["ink2"])
    ax.set_yticks(y, [LABELS[s] for s in SYSTEMS])
    ax.set_xlim(0, 100)
    ax.set_xlabel("median sentence chrF++")
    style(ax, t, "x")
    legend(ax, t, loc="lower right")
    title(fig, t, "Memorising or translating?",
          "Test sentences split by whether a training source is ≥ 90% similar (difflib ratio)")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def sentence_distribution(per_sentence: dict[str, np.ndarray], t: dict):
    fig, ax = figure(t, figsize=(8, 3.8))
    bins = np.arange(0, 105, 5)
    for name, colour in (("transformer", t["c1"]), ("retrieval", t["c2"])):
        counts, _ = np.histogram(per_sentence[name], bins=bins)
        ax.stairs(counts, bins, color=colour, linewidth=2, label=f"{LABELS[name]} (median {np.median(per_sentence[name]):.1f})")
    ax.set_xlim(0, 100)
    ax.set_xlabel("sentence chrF++ (100 = identical to the reference)")
    ax.set_ylabel("test sentences")
    style(ax, t)
    legend(ax, t, loc="upper left")
    title(fig, t, "Score per sentence", "A corpus average hides the spread: how many sentences land in each 5-point band")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def tokens_per_word(stats: dict, chosen: int, t: dict):
    fig, ax = figure(t, figsize=(8, 3.6))
    sizes = sorted(stats["tokenizer"], key=int)
    x = np.arange(len(sizes))
    w = 0.32
    en = [stats["tokenizer"][s]["tokens_per_word_en_test"] for s in sizes]
    tr = [stats["tokenizer"][s]["tokens_per_word_tr_test"] for s in sizes]
    ax.bar(x - w / 2 - 0.01, en, w, color=t["c1"], label="English")
    ax.bar(x + w / 2 + 0.01, tr, w, color=t["c2"], label="Turkish")
    for xi, e, r in zip(x, en, tr):
        ax.text(xi - w / 2, e + 0.03, f"{e:.2f}", ha="center", fontsize=9, color=t["ink2"])
        ax.text(xi + w / 2, r + 0.03, f"{r:.2f}", ha="center", fontsize=9, color=t["ink"])
    ax.set_xticks(x, [f"{int(s):,}" + (" (used)" if int(s) == chosen else "") for s in sizes])
    ax.set_xlabel("BPE vocabulary size per language")
    ax.set_ylabel("tokens per word")
    ax.set_ylim(0, 2.1)
    style(ax, t)
    legend(ax, t, loc="upper right")
    title(fig, t, "Turkish words cost more tokens", "Average BPE pieces per test-set word; 1.0 would mean every word is a single token")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def main() -> None:
    cfg = load_config()
    results = json.loads((ROOT / "results" / "eval.json").read_text())
    history = json.loads((ROOT / "checkpoints" / "history.json").read_text())
    stats = json.loads((ROOT / cfg["data"]["dir"] / "corpus_stats.json").read_text())
    _, refs = load_split(ROOT / cfg["data"]["dir"], "test")
    per_sentence = {
        name: np.array(sentence_scores((ROOT / "results" / f"test.{name}.tr").read_text().splitlines(), refs))
        for name in ("transformer", "retrieval")
    }
    out = ROOT / "figures"
    for mode, t in THEMES.items():
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
        figs = {
            "scores": scores_by_system(results, t),
            "training": training_curve(history, cfg["train"]["warmup"], t),
            "near_duplicates": near_duplicate_split(results, t),
            "sentence_scores": sentence_distribution(per_sentence, t),
            "tokens_per_word": tokens_per_word(stats, cfg["tokenizer"]["vocab_size"], t),
        }
        for name, fig in figs.items():
            fig.savefig(out / f"{name}.{mode}.png", dpi=160, facecolor=t["surface"])
            plt.close(fig)
    print(f"wrote {len(figs) * len(THEMES)} figures to {out}")


if __name__ == "__main__":
    main()
