"""Score the trained model and the three baselines on the test set.

Every system gets corpus scores (with signatures) and the distribution of sentence chrF++,
over all of test and separately over the test sentences flagged as near-duplicates of a
training source. A model whose gain over retrieval is concentrated in the near-duplicate
slice is mostly remembering.
"""

import json

import numpy as np
import torch

from baselines import copy_source
from baselines.dictionary import Dictionary
from baselines.retrieval import Retrieval
from decode.greedy import greedy_decode
from eval.distribution import summarize
from eval.scores import HEADLINE, corpus_scores, sentence_scores
from scripts.config import ROOT, load_config, pick_device
from scripts.train import build_model, encode_corpus, load_split, pad_batch
from tokenizer.bpe import BPE


@torch.no_grad()
def translate_model(model, src_bpe: BPE, tgt_bpe: BPE, src: list[str], cfg: dict, device, batch_size: int = 128) -> tuple[list[str], list[bool]]:
    """Greedy translations in input order, and whether each one stopped on EOS before its cap."""
    model.eval()
    ids = encode_corpus(src_bpe, src, bos=False)
    # Length-sorted batches: padding is wasted decoder steps, and greedy runs to the longest cap.
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    hyps, finished = [""] * len(ids), [False] * len(ids)
    for k in range(0, len(order), batch_size):
        idx = order[k : k + batch_size]
        out, done = greedy_decode(
            model, pad_batch([ids[i] for i in idx]).to(device), cfg["decode"]["max_len_a"], cfg["decode"]["max_len_b"]
        )
        for i, toks, d in zip(idx, out, done):
            hyps[i], finished[i] = tgt_bpe.decode(toks), d
    return hyps, finished


def average_ranks(x: list[float]) -> np.ndarray:
    """Ranks with ties sharing their mean rank, so Spearman is Pearson on these.

    Sentence chrF++ ties heavily (every exact match is 100); argsort-of-argsort would break
    those ties by input order and move the correlation.
    """
    x = np.asarray(x, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2)[inverse]


def report(hyps: list[str], refs: list[str], near_dup: np.ndarray) -> dict:
    sent = np.array(sentence_scores(hyps, refs))
    return {
        "corpus": corpus_scores(hyps, refs),
        f"sentence_{HEADLINE}": {
            "all": summarize(sent),
            "near_dup": summarize(sent[near_dup]),
            "not_near_dup": summarize(sent[~near_dup]),
        },
    }


def main() -> None:
    cfg = load_config()
    data_dir = ROOT / cfg["data"]["dir"]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    test_src, test_ref = load_split(data_dir, "test")
    near_dup = np.array(json.loads((data_dir / "test_near_dup.json").read_text()), dtype=bool)
    assert len(near_dup) == len(test_src), "test_near_dup.json is from a different split; rerun prepare_data"

    vocab = cfg["tokenizer"]["vocab_size"]
    src_bpe = BPE.load(data_dir / f"bpe_en_{vocab}.json")
    tgt_bpe = BPE.load(data_dir / f"bpe_tr_{vocab}.json")
    device = pick_device(cfg["device"])
    ckpt = torch.load(ROOT / "checkpoints" / "model.pt", map_location="cpu")
    model = build_model(ckpt["config"]["model"], len(src_bpe), len(tgt_bpe))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    systems: dict[str, list[str]] = {}
    systems["transformer"], finished = translate_model(model, src_bpe, tgt_bpe, test_src, cfg, device)
    del model
    systems["copy_source"] = copy_source.translate(test_src)
    train_src, train_tgt = load_split(data_dir, "train")
    systems["dictionary"] = Dictionary.build(train_src, train_tgt).translate(test_src)
    systems["retrieval"], retrieval_sim = Retrieval(train_src, train_tgt).translate(test_src)

    results = {name: report(hyps, test_ref, near_dup) for name, hyps in systems.items()}
    results["transformer"]["hit_length_cap"] = len(finished) - sum(finished)
    results["n_test"], results["n_near_dup"] = len(test_src), int(near_dup.sum())
    # Does the model do better exactly where a close training neighbour exists?
    model_sent = sentence_scores(systems["transformer"], test_ref)
    results["spearman_model_chrf_vs_retrieval_sim"] = float(
        np.corrcoef(average_ranks(model_sent), average_ranks(retrieval_sim))[0, 1]
    )

    (out / "eval.json").write_text(json.dumps(results, indent=2))
    for name, hyps in systems.items():
        (out / f"test.{name}.tr").write_text("\n".join(hyps) + "\n")
    print(f"{'system':<12} {'chrF++':>7} {'chrF':>6} {'BLEU':>6} {'TER':>6}  median  worst10%  near-dup/other median")
    for name in systems:
        c, s = results[name]["corpus"], results[name][f"sentence_{HEADLINE}"]
        print(
            f"{name:<12} {c['chrF++']['score']:7.1f} {c['chrF']['score']:6.1f} {c['BLEU']['score']:6.1f} {c['TER']['score']:6.1f}"
            f"  {s['all']['median']:6.1f}  {s['all']['worst_decile_mean']:8.1f}  {s['near_dup']['median']:.1f}/{s['not_near_dup']['median']:.1f}"
        )
    print(f"hit length cap: {results['transformer']['hit_length_cap']}/{len(test_src)}")
    print(f"spearman(model chrF++, retrieval sim): {results['spearman_model_chrf_vs_retrieval_sim']:.3f}")
    print(json.dumps({n: results[n]["corpus"]["chrF++"]["signature"] for n in ("transformer",)}))


if __name__ == "__main__":
    main()
