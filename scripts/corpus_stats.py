"""Corpus statistics and BPE statistics per vocabulary size; written to data/corpus_stats.json."""

import json
import time

from scripts.config import ROOT, load_config
from scripts.train import load_split
from tokenizer.bpe import BPE, train_bpe
from tokenizer.stats import single_token_word_rate, suffix_survival, tokens_per_word, type_token_ratio, unseen_type_rate, words

VOCAB_SIZES = (4000, 8000, 16000)
TTR_SAMPLE = 200_000  # words per side; TTR is only comparable at equal sample size


def main() -> None:
    cfg = load_config()
    data_dir = ROOT / cfg["data"]["dir"]
    train_en, train_tr = load_split(data_dir, "train")
    test_en, test_tr = load_split(data_dir, "test")

    en_words = sum(len(s.split()) for s in train_en)
    tr_words = sum(len(s.split()) for s in train_tr)
    corpus = {
        "train_pairs": len(train_en),
        "test_pairs": len(test_en),
        "train_words_en": en_words,
        "train_words_tr": tr_words,
        "tr_words_per_en_word": tr_words / en_words,
        f"ttr_en_at_{TTR_SAMPLE}": type_token_ratio(train_en, "en", TTR_SAMPLE),
        f"ttr_tr_at_{TTR_SAMPLE}": type_token_ratio(train_tr, "tr", TTR_SAMPLE),
        "test_word_types_unseen_en": unseen_type_rate(train_en, test_en, "en"),
        "test_word_types_unseen_tr": unseen_type_rate(train_tr, test_tr, "tr"),
        "train_word_types_en": len({w for s in train_en for w in words(s, "en")}),
        "train_word_types_tr": len({w for s in train_tr for w in words(s, "tr")}),
    }

    tokenizer = {}
    for vocab in VOCAB_SIZES:
        row = {}
        for lang, train, test in (("en", train_en, test_en), ("tr", train_tr, test_tr)):
            path = data_dir / f"bpe_{lang}_{vocab}.json"
            start = time.time()
            if path.exists():
                bpe = BPE.load(path)
            else:
                bpe = train_bpe(train, vocab, cfg["tokenizer"]["min_frequency"])
                bpe.save(path)
            row[f"train_seconds_{lang}"] = round(time.time() - start, 1)
            row[f"vocab_{lang}"] = len(bpe)
            row[f"tokens_per_word_{lang}_test"] = tokens_per_word(bpe, test)
            row[f"single_token_words_{lang}_test"] = single_token_word_rate(bpe, test, lang)
            if lang == "tr":
                row["suffixes"] = suffix_survival(bpe, test)
        tokenizer[vocab] = row

    report = {"corpus": corpus, "tokenizer": tokenizer}
    (data_dir / "corpus_stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
