import re
from collections import Counter

from tokenizer.bpe import BPE, WORD_START

# Multi-character suffixes only: single letters like -a/-ı end too many stems by accident to
# say anything about morphology. Vowel-harmony variants are listed separately because BPE
# learns them as separate merges. The progressive is -(I)yor: the linking vowel belongs to the
# suffix, and BPE learns "iyor", "uyor", ..., never a bare "yor" at a word end.
TURKISH_SUFFIXES = (
    "lar", "ler", "dan", "den", "tan", "ten", "iyor", "ıyor", "uyor", "üyor", "iyorum", "ıyorum",
    "dım", "dim", "mış", "miş", "lık", "lik", "acak", "ecek", "sınız", "siniz", "mak", "mek",
)


def words(text: str, lang: str) -> list[str]:
    # Turkish dotted/dotless i: str.lower() maps "I" to "i" and "İ" to "i̇" (two code points),
    # both wrong for Turkish, so they are mapped first.
    if lang == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    return re.findall(r"[^\W\d_]+", text.lower())


def tokens_per_word(bpe: BPE, texts: list[str]) -> float:
    n_tokens = sum(len([t for t in bpe.tokens(s) if t.strip(WORD_START)]) for s in texts)
    n_words = sum(len(s.split()) for s in texts)
    return n_tokens / n_words


def type_token_ratio(texts: list[str], lang: str, n_tokens: int) -> float:
    """TTR on the first `n_tokens` words. TTR falls with sample size, so two sides are only
    comparable at the same n."""
    sample: list[str] = []
    for s in texts:
        sample.extend(words(s, lang))
        if len(sample) >= n_tokens:
            break
    sample = sample[:n_tokens]
    return len(set(sample)) / len(sample)


def unseen_type_rate(train: list[str], test: list[str], lang: str) -> float:
    seen = {w for s in train for w in words(s, lang)}
    test_types = {w for s in test for w in words(s, lang)}
    return len(test_types - seen) / len(test_types)


def suffix_survival(bpe: BPE, texts: list[str], suffixes: tuple[str, ...] = TURKISH_SUFFIXES) -> dict:
    """Per suffix: is it a vocabulary item, and for words ending in it, how often is the final
    BPE token exactly that suffix (cut off cleanly, nothing of the stem attached)."""
    counts = Counter(w for s in texts for w in words(s, "tr"))
    report = {}
    for suffix in suffixes:
        ending = [(w, n) for w, n in counts.items() if w.endswith(suffix) and len(w) > len(suffix) + 1]
        total = sum(n for _, n in ending)
        clean = sum(n for w, n in ending if bpe.tokens(w)[-1] == suffix)
        report[suffix] = {
            "in_vocab": suffix in bpe.token_to_id,
            "words": total,
            "final_token_is_suffix": clean / total if total else float("nan"),
        }
    return report


def single_token_word_rate(bpe: BPE, texts: list[str], lang: str) -> float:
    """Share of running words that BPE keeps whole. On Turkish this is the share where
    morphology is invisible to the model as structure: "▁okuyorum" is one opaque symbol."""
    ws = [w for s in texts for w in words(s, lang)]
    return sum(len(bpe.tokens(w)) == 1 for w in ws) / len(ws)
