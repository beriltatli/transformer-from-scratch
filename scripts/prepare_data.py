"""Download Tatoeba en-tr, clean it, and split it so no sentence on either side crosses splits.

Tatoeba links one English sentence to several Turkish translations and one Turkish sentence to
several English paraphrases. Splitting on the source alone would put "Yorgunum." in test under
"I am tired." and in train under "I'm tired.", so pairs are grouped by connected component over
both sides (union-find on normalised text) and whole groups are assigned to a split. Test and
valid keep one pair per group, so a frequent sentence cannot dominate the held-out score.
"""

import difflib
import io
import json
import random
import re
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from scripts.config import ROOT, load_config


def normalise_key(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def download(url: str, raw_dir: Path) -> tuple[list[str], list[str]]:
    en_path, tr_path = raw_dir / "Tatoeba.en-tr.en", raw_dir / "Tatoeba.en-tr.tr"
    if not en_path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(raw_dir)
    return en_path.read_text().splitlines(), tr_path.read_text().splitlines()


def clean(en: list[str], tr: list[str], max_words: int) -> list[tuple[str, str]]:
    pairs = set()
    for s, t in zip(en, tr, strict=True):
        s = unicodedata.normalize("NFC", " ".join(s.split()))
        t = unicodedata.normalize("NFC", " ".join(t.split()))
        if not s or not t or len(s.split()) > max_words or len(t.split()) > max_words:
            continue
        pairs.add((s, t))
    # sorted so the split depends on the seed only, not on set iteration order
    return sorted(pairs)


def group_pairs(pairs: list[tuple[str, str]]) -> list[list[int]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for s, t in pairs:
        parent[find("en:" + normalise_key(s))] = find("tr:" + normalise_key(t))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (s, _) in enumerate(pairs):
        groups[find("en:" + normalise_key(s))].append(i)
    return sorted(groups.values())


def split(pairs, groups, valid_size: int, test_size: int, seed: int):
    rng = random.Random(seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    test = [pairs[rng.choice(groups[g])] for g in order[:test_size]]
    valid = [pairs[rng.choice(groups[g])] for g in order[test_size : test_size + valid_size]]
    train = [pairs[i] for g in order[test_size + valid_size :] for i in groups[g]]
    return train, valid, test


def near_duplicates(train_src: list[str], test_src: list[str], ratio: float) -> int:
    """Test sources with a train source at difflib ratio >= `ratio` after normalisation.

    Candidates come from an inverted index on each test sentence's rarest word, so this misses
    a near-duplicate that differs exactly in that word. It is a lower bound.
    """
    train_keys = [normalise_key(s) for s in train_src]
    freq: dict[str, int] = defaultdict(int)
    index: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(train_keys):
        for w in set(key.split()):
            freq[w] += 1
            index[w].append(i)
    count = 0
    for s in test_src:
        key = normalise_key(s)
        words = [w for w in key.split() if w in index]
        if not words:
            continue
        rarest = min(words, key=lambda w: freq[w])
        if any(
            difflib.SequenceMatcher(None, key, train_keys[i]).ratio() >= ratio
            for i in index[rarest][:2000]
        ):
            count += 1
    return count


def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["data"]["dir"]
    en, tr = download(cfg["data"]["url"], out / "raw")
    pairs = clean(en, tr, cfg["data"]["max_words"])
    groups = group_pairs(pairs)
    train, valid, test = split(pairs, groups, cfg["data"]["valid_size"], cfg["data"]["test_size"], cfg["seed"])

    for name, part in (("train", train), ("valid", valid), ("test", test)):
        (out / f"{name}.en").write_text("\n".join(s for s, _ in part) + "\n")
        (out / f"{name}.tr").write_text("\n".join(t for _, t in part) + "\n")

    train_src_keys = {normalise_key(s) for s, _ in train}
    train_tgt_keys = {normalise_key(t) for _, t in train}
    report = {
        "raw_pairs": len(en),
        "clean_unique_pairs": len(pairs),
        "groups": len(groups),
        "largest_group": max(len(g) for g in groups),
        "train": len(train), "valid": len(valid), "test": len(test),
        "test_src_exact_in_train": sum(normalise_key(s) in train_src_keys for s, _ in test),
        "test_tgt_exact_in_train": sum(normalise_key(t) in train_tgt_keys for _, t in test),
        "test_src_near_dup_in_train": near_duplicates(
            [s for s, _ in train], [s for s, _ in test], cfg["data"]["near_dup_ratio"]
        ),
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
