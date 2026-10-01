"""`tokenizers` appears here only, as a reference for merge order."""

import json
import math

import pytest

from tokenizer.bpe import BOS, EOS, SPECIALS, UNK, UNK_SURFACE, WORD_START, BPE, pretokenize, train_bpe
from tokenizer.stats import single_token_word_rate, suffix_survival, tokens_per_word, type_token_ratio, unseen_type_rate, words

# Sennrich et al. (2016), Fig. 1: low x5, lower x2, newest x6, widest x3.
SENNRICH = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3

TURKISH = [
    "Yarın gelemeyeceğim.",
    "Çocuklar dışarıda oynuyor.",
    "Bugün İstanbul'da hava çok soğuk.",
    "Anahtarlarımı gördün mü?",
    "Tom 1999'da 3 kitap yazdı, değil mi?",
]


def test_sennrich_toy_merges() -> None:
    bpe = train_bpe(SENNRICH, vocab_size=len(SPECIALS) + 11 + 4, min_frequency=2)
    # Alphabet ids by code point: d0 e1 i2 l3 n4 o5 r6 s7 t8 w9 ▁10 (U+2581 sorts after ASCII).
    # "es"(1,7) and "st"(7,8) both occur 6 + 3 = 9 times: "es" has the smaller ids. Then "▁l"(10,3),
    # "lo"(3,5), "ow"(5,9) all occur 5 + 2 = 7: "lo". Then "▁lo"(10,12) beats "low"(12,9), both 7.
    assert bpe.merges == [("e", "s"), ("es", "t"), ("l", "o"), (WORD_START, "lo")]
    assert bpe.merge_counts == [9, 9, 7, 7]
    assert bpe.merge_ties == [True, False, True, True]


def test_pretokenize_separates_punctuation_and_digits() -> None:
    assert pretokenize("İstanbul'da 3 kedi.") == [
        WORD_START + "İstanbul", "'", "da", WORD_START + "3", WORD_START + "kedi", "."
    ]


@pytest.mark.parametrize("vocab_size", [40, 80, 400])
def test_roundtrip_is_lossless(vocab_size: int) -> None:
    bpe = train_bpe(TURKISH * 3, vocab_size=vocab_size)
    for text in TURKISH:
        assert bpe.decode(bpe.encode(text, bos=True, eos=True)) == text


def test_bos_eos_and_unknown_characters() -> None:
    bpe = train_bpe(TURKISH, vocab_size=60)
    ids = bpe.encode("ya€", bos=True, eos=True)
    assert ids[0] == BOS and ids[-1] == EOS and UNK in ids
    assert bpe.decode(ids) == "ya" + UNK_SURFACE


def test_training_is_deterministic_and_save_load_roundtrips(tmp_path) -> None:
    a = train_bpe(TURKISH * 2, vocab_size=90)
    b = train_bpe(list(reversed(TURKISH * 2)), vocab_size=90)
    assert a.merges == b.merges
    a.save(tmp_path / "bpe.json")
    loaded = BPE.load(tmp_path / "bpe.json")
    assert loaded.token_to_id == a.token_to_id
    assert all(loaded.encode(t) == a.encode(t) for t in TURKISH)


def test_merges_match_huggingface() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    corpus = [s for s in TURKISH for _ in range(4)] + ["evlerimizde evlerden kitaplar kitaplık okuyorum okuyorsun"] * 3
    mine = train_bpe(corpus, vocab_size=150, min_frequency=2)
    assert sum(mine.merge_ties) > 10, "a tie-free corpus would not exercise the tie-break rule"

    hf = tokenizers.Tokenizer(tokenizers.models.BPE())
    hf.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=150, min_frequency=2, show_progress=False)
    hf.train_from_iterator([" ".join(pretokenize(s)) for s in corpus], trainer)
    hf_merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" "))
                 for m in json.loads(hf.to_str())["model"]["merges"]]

    # HF's vocab_size has no room reserved for our four specials, so it may run 4 merges longer.
    assert len(mine.merges) > 50
    assert mine.merges == hf_merges[: len(mine.merges)]


def test_words_handles_turkish_dotted_i() -> None:
    assert words("İSTANBUL'da Irmak", "tr") == ["istanbul", "da", "ırmak"]


def test_tokens_per_word_counts_only_real_tokens() -> None:
    bpe = train_bpe(["ab ab"], vocab_size=len(SPECIALS) + 3 + 2)
    # Alphabet {▁, a, b}; merges ▁a, ▁ab: each word is one token.
    assert tokens_per_word(bpe, ["ab ab ab"]) == 1.0


def test_type_token_ratio_and_unseen_rate() -> None:
    assert type_token_ratio(["a b a b"], "en", n_tokens=4) == 0.5
    assert type_token_ratio(["a b a b c d"], "en", n_tokens=2) == 1.0
    assert unseen_type_rate(["ev evler"], ["ev evde evden"], "tr") == pytest.approx(2 / 3)


def test_suffix_survival() -> None:
    bpe = BPE(merges=[("l", "e"), ("le", "r"), (WORD_START, "e"), (WORD_START + "e", "v")],
              alphabet=[WORD_START, "e", "l", "r", "v"])
    report = suffix_survival(bpe, ["evler evler"], suffixes=("ler", "den"))
    assert report["ler"] == {"in_vocab": True, "words": 2, "final_token_is_suffix": 1.0}
    assert report["den"]["words"] == 0 and math.isnan(report["den"]["final_token_is_suffix"])


def test_single_token_word_rate() -> None:
    bpe = BPE(merges=[(WORD_START, "e"), (WORD_START + "e", "v")], alphabet=[WORD_START, "e", "l", "r", "v"])
    assert single_token_word_rate(bpe, ["ev evler ev"], "tr") == pytest.approx(2 / 3)
