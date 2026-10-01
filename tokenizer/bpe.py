"""Byte-pair encoding over Unicode characters, trained and applied per language side.

Characters rather than bytes: every Turkish letter (ç ğ ı ö ş ü İ) is one symbol, so a merge
table and a segmentation read as text, which the suffix analysis depends on.
"""

import heapq
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD, UNK, BOS, EOS = range(4)
WORD_START = "▁"
UNK_SURFACE = "⁇"

# Letters, digit runs, or one non-word character. Merges never cross these boundaries, so
# punctuation cannot fuse with word endings ("geldim." -> "▁geldim" + "."), which would
# otherwise smear sentence-final punctuation across the suffix inventory.
_CHUNK = re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]")


def pretokenize(text: str) -> list[str]:
    chunks = []
    for word in unicodedata.normalize("NFC", text).split():
        pieces = _CHUNK.findall(word)
        pieces[0] = WORD_START + pieces[0]
        chunks.extend(pieces)
    return chunks


def _pairs(symbols: list[str]) -> Counter:
    return Counter(zip(symbols, symbols[1:]))


def _merge_symbols(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    merged, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


@dataclass
class BPE:
    merges: list[tuple[str, str]]
    alphabet: list[str]
    merge_counts: list[int] = field(default_factory=list)
    # merge_ties[i]: another pair had the same count when merge i was chosen, so the choice
    # came from the tie-break rule rather than from the data.
    merge_ties: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ranks = {pair: r for r, pair in enumerate(self.merges)}
        tokens = list(SPECIALS) + self.alphabet + [a + b for a, b in self.merges]
        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = tokens
        self._cache: dict[str, list[str]] = {}

    def __len__(self) -> int:
        return len(self.id_to_token)

    def _segment(self, chunk: str) -> list[str]:
        if chunk in self._cache:
            return self._cache[chunk]
        symbols = list(chunk)
        # Apply merges in training order: always the lowest-ranked pair present. Greedy
        # longest-match against the vocabulary would give different, untrained segmentations.
        while len(symbols) > 1:
            pair = min(zip(symbols, symbols[1:]), key=lambda p: self.ranks.get(p, len(self.ranks)))
            if pair not in self.ranks:
                break
            symbols = _merge_symbols(symbols, pair)
        self._cache[chunk] = symbols
        return symbols

    def tokens(self, text: str) -> list[str]:
        return [tok for chunk in pretokenize(text) for tok in self._segment(chunk)]

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(tok, UNK) for tok in self.tokens(text)]
        return [BOS] * bos + ids + [EOS] * eos

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for i in ids:
            if i == UNK:
                pieces.append(UNK_SURFACE)
            elif i >= len(SPECIALS):
                pieces.append(self.id_to_token[i])
        return "".join(pieces).replace(WORD_START, " ").strip()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"alphabet": self.alphabet, "merges": self.merges}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "BPE":
        raw = json.loads(path.read_text())
        return cls([tuple(m) for m in raw["merges"]], raw["alphabet"])


def train_bpe(texts: list[str], vocab_size: int, min_frequency: int = 2) -> BPE:
    """Merge the most frequent adjacent pair until the vocabulary (specials included) is full
    or no pair occurs `min_frequency` times.

    Ties go to the pair with the smallest (left id, right id), ids being alphabet by code point
    and then merged tokens in creation order. Ties are the norm, not the exception: every
    adjacent pair inside one word type shares that word's count. This is the rule HuggingFace
    `tokenizers` uses, which makes the full merge sequence comparable in tests.
    """
    chunk_counts = Counter(chunk for text in texts for chunk in pretokenize(text))
    words = [list(chunk) for chunk in chunk_counts]
    counts = list(chunk_counts.values())
    alphabet = sorted({ch for chunk in chunk_counts for ch in chunk})

    pair_counts: Counter = Counter()
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    for idx, symbols in enumerate(words):
        for pair, n in _pairs(symbols).items():
            pair_counts[pair] += n * counts[idx]
            where[pair].add(idx)
    token_id = {ch: i for i, ch in enumerate(alphabet)}

    def key(pair: tuple[str, str]) -> tuple[int, int, int, tuple[str, str]]:
        return (-pair_counts[pair], token_id[pair[0]], token_id[pair[1]], pair)

    heap = [key(pair) for pair in pair_counts]
    heapq.heapify(heap)

    merges: list[tuple[str, str]] = []
    merge_counts: list[int] = []
    merge_ties: list[bool] = []
    n_merges = vocab_size - len(SPECIALS) - len(alphabet)
    while len(merges) < n_merges and heap:
        neg, _, _, pair = heapq.heappop(heap)
        # Lazy deletion: the heap holds stale counts; only an entry matching the live count
        # is real. Every count change pushes a fresh entry, so the live one is always present.
        if -neg != pair_counts[pair] or -neg == 0:
            continue
        if -neg < min_frequency:
            break
        while heap and -heap[0][0] != pair_counts[heap[0][3]]:
            heapq.heappop(heap)
        token_id[pair[0] + pair[1]] = len(token_id)
        merges.append(pair)
        merge_counts.append(-neg)
        merge_ties.append(bool(heap) and -heap[0][0] == -neg)

        changed: Counter = Counter()
        # `where` may list words that no longer contain the pair (entries are never removed);
        # those re-merge to themselves and contribute a net delta of zero.
        for idx in where.pop(pair):
            old = words[idx]
            new = _merge_symbols(old, pair)
            if new == old:
                continue
            for p, n in _pairs(old).items():
                changed[p] -= n * counts[idx]
            for p, n in _pairs(new).items():
                changed[p] += n * counts[idx]
                where[p].add(idx)
            words[idx] = new
        for p, delta in changed.items():
            if delta:
                pair_counts[p] += delta
                heapq.heappush(heap, key(p))
        del pair_counts[pair]

    return BPE(merges, alphabet, merge_counts, merge_ties)
