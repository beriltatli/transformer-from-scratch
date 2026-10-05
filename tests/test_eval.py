import pytest

from baselines import copy_source
from baselines.dictionary import Dictionary
from baselines.retrieval import Retrieval
from eval.distribution import summarize
from eval.scores import corpus_scores, sentence_scores


def test_summarize_known_values() -> None:
    s = summarize([float(i) for i in range(10, 0, -1)])
    # numpy linear interpolation on 1..10: position (n - 1) * q.
    assert s["median"] == 5.5 and s["q1"] == 3.25 and s["q3"] == 7.75 and s["iqr"] == 4.5
    assert s["p10"] == pytest.approx(1.9)
    assert s["worst_decile_mean"] == 1.0 and s["n"] == 10


def test_worst_decile_rounds_up() -> None:
    assert summarize([0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])["worst_decile_mean"] == 5.0


def test_identical_output_scores_perfectly_with_signatures() -> None:
    refs = ["Tom geçen hafta yeni bir araba aldı.", "Pencereyi kapatabilir misiniz, lütfen?"]
    scores = corpus_scores(refs, refs)
    assert scores["chrF++"]["score"] == 100 and scores["chrF"]["score"] == 100
    assert scores["BLEU"]["score"] == pytest.approx(100) and scores["TER"]["score"] == 0
    assert "nrefs:1" in scores["BLEU"]["signature"] and "tok:13a" in scores["BLEU"]["signature"]
    assert "nw:2" in scores["chrF++"]["signature"] and "nw:0" in scores["chrF"]["signature"]


def test_corpus_bleu_is_zero_for_a_perfect_copy_of_short_sentences() -> None:
    # No sentence reaches 4 tokens under 13a ("Yarın gelemeyeceğim ." is 3), so the 4-gram
    # precision is 0/0, smoothed to 0, and the geometric mean is 0. On a corpus whose median
    # sentence is five words, this is not an edge case.
    refs = ["Yarın gelemeyeceğim.", "Geliyorum."]
    assert corpus_scores(refs, refs)["BLEU"]["score"] == 0.0
    assert corpus_scores(refs, refs)["chrF++"]["score"] == 100.0


def test_sentence_scores_extremes_and_partial_credit() -> None:
    assert sentence_scores(["Yarın gelemeyeceğim."], ["Yarın gelemeyeceğim."]) == [100.0]
    assert sentence_scores(["xyz"], ["abc"]) == [0.0]
    # The suffix example from the README: one word, different person suffix. Word-level BLEU
    # gives no 1-gram match, chrF++ still credits the shared stem.
    chrf = sentence_scores(["gelemeyeceksin"], ["gelemeyeceğim"])[0]
    bleu = sentence_scores(["gelemeyeceksin"], ["gelemeyeceğim"], "BLEU")[0]
    assert bleu == 0.0 and chrf > 40


def test_copy_source_is_identity() -> None:
    assert copy_source.translate(["Tom is here."]) == ["Tom is here."]


def test_dictionary_picks_max_dice() -> None:
    src = ["the cat", "the dog", "a cat"]
    tgt = ["kedi", "köpek", "bir kedi"]
    d = Dictionary.build(src, tgt, min_count=1)
    # cat: c=2, kedi: c=2, together 2 -> 1.0. the: c=2; with köpek (c=1) 2*1/3, with kedi 2*1/4.
    assert d.table["cat"] == ("kedi", 1.0)
    assert d.table["the"] == ("köpek", pytest.approx(2 / 3))
    d.min_dice = 0.7
    assert d.translate_one("The cat .") == "Kedi."
    assert d.translate_one("Mary saw the cat") == "Mary kedi"


def test_retrieval_returns_nearest_target() -> None:
    r = Retrieval(["the red car", "a blue house", "the red house"], ["kırmızı araba", "mavi ev", "kırmızı ev"])
    out, sims = r.translate(["red car", "blue house", "zebra"])
    assert out[:2] == ["kırmızı araba", "mavi ev"]
    assert sims[0] > 0.8 and sims[2] == 0.0


def test_average_ranks_share_ties() -> None:
    from scripts.evaluate import average_ranks

    assert average_ranks([100.0, 3.0, 100.0, 7.0]).tolist() == [2.5, 0.0, 2.5, 1.0]


def test_report_slices_near_duplicates() -> None:
    import numpy as np

    from scripts.evaluate import report

    refs = ["Yarın gelemeyeceğim.", "Tren istasyonu nerede?", "Bu soruyu anlamıyorum."]
    hyps = [refs[0], "qqq", refs[2]]
    r = report(hyps, refs, np.array([True, False, True]))["sentence_chrF++"]
    assert r["near_dup"]["n"] == 2 and r["near_dup"]["median"] == 100.0
    assert r["not_near_dup"]["n"] == 1 and r["not_near_dup"]["median"] == 0.0
