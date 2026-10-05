"""Word-by-word lookup: no reordering, no morphology, no context.

Each English word maps to the Turkish word with the highest Dice coefficient over sentence
co-occurrence, 2 c(e, t) / (c(e) + c(t)). Raw "most frequent co-occurring word" would map
nearly everything to "bir" or "Tom", which co-occur with everything; Dice is the smallest
change that discounts that, and is still counting, not alignment.
"""

import re

import numpy as np

from tokenizer.stats import words

_TOKEN = re.compile(r"[^\W\d_]+|\d+|[^\w\s]")


class Dictionary:
    def __init__(self, table: dict[str, tuple[str, float]], min_dice: float) -> None:
        self.table = table
        self.min_dice = min_dice

    @classmethod
    def build(cls, src: list[str], tgt: list[str], min_dice: float = 0.0, min_count: int = 2) -> "Dictionary":
        src_words = [sorted(set(words(s, "en"))) for s in src]
        tgt_words = [sorted(set(words(t, "tr"))) for t in tgt]
        src_vocab = {w: i for i, w in enumerate(sorted({w for ws in src_words for w in ws}))}
        tgt_vocab = {w: i for i, w in enumerate(sorted({w for ws in tgt_words for w in ws}))}
        src_count = np.zeros(len(src_vocab), dtype=np.int64)
        tgt_count = np.zeros(len(tgt_vocab), dtype=np.int64)

        # Pair (e, t) is stored as one int64 key e * |T| + t; np.unique then counts them. A
        # Counter of string tuples over ~20M co-occurrences does not fit in 8 GB.
        keys = []
        for es, ts in zip(src_words, tgt_words):
            e_ids = np.fromiter((src_vocab[w] for w in es), dtype=np.int64)
            t_ids = np.fromiter((tgt_vocab[w] for w in ts), dtype=np.int64)
            src_count[e_ids] += 1
            tgt_count[t_ids] += 1
            keys.append((e_ids[:, None] * len(tgt_vocab) + t_ids[None, :]).ravel())
        pairs, together = np.unique(np.concatenate(keys), return_counts=True)
        keep = together >= min_count
        pairs, together = pairs[keep], together[keep]
        e_idx, t_idx = pairs // len(tgt_vocab), pairs % len(tgt_vocab)
        dice = 2 * together / (src_count[e_idx] + tgt_count[t_idx])

        # Best t per e: sort by (e, -dice), take the first row of each e.
        order = np.lexsort((-dice, e_idx))
        first = np.ones(len(order), dtype=bool)
        first[1:] = e_idx[order][1:] != e_idx[order][:-1]
        src_names = np.array(sorted(src_vocab, key=src_vocab.get), dtype=object)
        tgt_names = np.array(sorted(tgt_vocab, key=tgt_vocab.get), dtype=object)
        best = order[first]
        table = {src_names[e_idx[i]]: (tgt_names[t_idx[i]], float(dice[i])) for i in best}
        return cls(table, min_dice)

    def translate_one(self, sentence: str) -> str:
        out = []
        for tok in _TOKEN.findall(sentence):
            key = tok.lower()
            if key in self.table:
                word, dice = self.table[key]
                if dice >= self.min_dice:
                    out.append(word)
            elif not tok[0].isalpha() or tok[0].isupper():
                # Punctuation, digits and unseen capitalised words (names) pass through.
                out.append(tok)
        text = " ".join(out)
        text = re.sub(r" ([.,!?;:])", r"\1", text)
        return text[:1].upper() + text[1:]

    def translate(self, src: list[str]) -> list[str]:
        return [self.translate_one(s) for s in src]
