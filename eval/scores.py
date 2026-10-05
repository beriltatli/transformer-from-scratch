"""Corpus and sentence scores through sacrebleu, always with the signature that pins the
configuration. A BLEU without its signature (tokeniser, smoothing, casing) is not a number."""

from sacrebleu.metrics import BLEU, CHRF, TER

# chrF++ is chrF with word unigrams and bigrams added (word_order=2, Popović 2017); chrF is
# characters only. Reporting both shows how much of chrF++ comes from word order.
HEADLINE = "chrF++"


def _metrics() -> dict:
    return {"chrF++": CHRF(word_order=2), "chrF": CHRF(), "BLEU": BLEU(), "TER": TER()}


def corpus_scores(hyps: list[str], refs: list[str]) -> dict[str, dict]:
    out = {}
    for name, metric in _metrics().items():
        score = metric.corpus_score(hyps, [refs])
        out[name] = {"score": score.score, "signature": str(metric.get_signature())}
    return out


def sentence_scores(hyps: list[str], refs: list[str], metric: str = HEADLINE) -> list[float]:
    if metric == "chrF++":
        scorer = CHRF(word_order=2)
    elif metric == "chrF":
        scorer = CHRF()
    elif metric == "BLEU":
        # Sentence BLEU without effective order is 0 for any sentence missing a 4-gram match,
        # which is most short sentences; effective order drops the absent n-gram orders.
        scorer = BLEU(effective_order=True)
    elif metric == "TER":
        scorer = TER()
    else:
        raise ValueError(metric)
    return [scorer.sentence_score(h, [r]).score for h, r in zip(hyps, refs, strict=True)]
