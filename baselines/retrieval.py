"""Return the training target whose source is nearest to the input (TF-IDF cosine).

If this scores close to the model, either the test set overlaps the training set or the model
is doing little more than lookup. The nearest-neighbour similarity is returned per sentence
so that question can be asked sentence by sentence.
"""

import math
from collections import Counter

import torch

from tokenizer.stats import words


class Retrieval:
    def __init__(self, train_src: list[str], train_tgt: list[str]) -> None:
        self.train_tgt = train_tgt
        docs = [Counter(words(s, "en")) for s in train_src]
        df = Counter(w for d in docs for w in d)
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(docs)
        # Smoothed idf, as in scikit-learn: log((1 + n) / (1 + df)) + 1, never zero.
        self.idf = torch.tensor([math.log((1 + n) / (1 + df[w])) + 1 for w in sorted(df)])
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            for w, tf in d.items():
                rows.append(r)
                cols.append(self.vocab[w])
                vals.append(tf * self.idf[self.vocab[w]].item())
        matrix = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals), (n, len(self.vocab)), check_invariants=True)
        norms = torch.zeros(n).index_add_(0, torch.tensor(rows), torch.tensor(vals) ** 2).sqrt().clamp_min(1e-12)
        self.matrix = (matrix * (1 / norms)[:, None]).coalesce().to_sparse_csr()

    def _queries(self, sentences: list[str]) -> torch.Tensor:
        q = torch.zeros(len(self.vocab), len(sentences))
        for j, s in enumerate(sentences):
            for w, tf in Counter(words(s, "en")).items():
                if w in self.vocab:
                    q[self.vocab[w], j] = tf * self.idf[self.vocab[w]]
        return q / q.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def translate(self, src: list[str], chunk: int = 64) -> tuple[list[str], list[float]]:
        # Chunked: the full (train x test) score matrix is 672k x 2000 floats, 5 GB.
        out, sims = [], []
        for k in range(0, len(src), chunk):
            scores = self.matrix @ self._queries(src[k : k + chunk])
            best, idx = scores.max(dim=0)
            out.extend(self.train_tgt[i] for i in idx.tolist())
            sims.extend(best.tolist())
        return out, sims
