"""
The rankings under comparison. Each takes a source row and returns the top 5 corpus rows, best first.

`stage1` and `rerank` reproduce production (stage 1 here is an exact cosine scan, which is what Postgres
runs at this corpus size). Every other system changes exactly one thing, from the variants deferred in
Phases 2 and 3.
"""

import functools
import re
import time
from collections import defaultdict

import numpy as np
import torch

from app.ml_models import rerank
from app.ml_models.retrieve import STAGE1_LIMIT
from app.utils import sbert_helper
from eval.corpus import Corpus, Doc

TOP_K = 5
QUERY_CAP_TOKENS = 128
BOILERPLATE = re.compile(r"\s*continue reading\.\.\.\s*$")

DESCRIPTIONS = {
    "stage1": "production stage 1: SBERT cosine on description + categories",
    "stage1_categories_first": "stage 1 with categories before the description (never cut by SBERT's 256 tokens)",
    "rerank": "production: stage 1 top 50, cross-encoder on description vs description",
    "rerank_titles": "cross-encoder query = source title, candidate = title + description",
    "rerank_query_cap": "query cut to its first 128 tokens, only the candidate truncated",
    "rerank_strip_boilerplate": "'continue reading...' removed from query and candidates",
    "rerank_max_length_256": "cross-encoder max_length 256 instead of 512",
    "rerank_top20": "cross-encoder reranks stage-1 top 20 instead of 50",
    "rerank_int8": "cross-encoder with dynamic int8 quantization",
}
SYSTEMS = list(DESCRIPTIONS)


def strip_boilerplate(text: str) -> str:
    return BOILERPLATE.sub("", text)


class Systems:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.tokenizer, self.model = rerank.get_model()
        self.seconds: dict[str, list[float]] = defaultdict(list)  # per system, time spent ranking (stage 1 excluded)

    def rank(self, name: str, source_row: int) -> list[int]:
        method = getattr(self, f"_{name}")
        self._neighbours(source_row)  # stage 1 is shared; keep it out of the timing
        start = time.perf_counter()
        rows = method(source_row)
        self.seconds[name].append(time.perf_counter() - start)
        return rows

    def links(self, name: str, source_row: int) -> list[str]:
        return [self.corpus.docs[row].link for row in self.rank(name, source_row)]

    # --- building blocks

    @staticmethod
    def _nearest(embeddings: np.ndarray, source_row: int, k: int) -> list[int]:
        similarity = embeddings @ embeddings[source_row]
        similarity[source_row] = -np.inf
        top = np.argpartition(-similarity, k)[:k]
        return top[np.argsort(-similarity[top])].tolist()

    @functools.cache
    def _neighbours(self, source_row: int) -> list[int]:
        return self._nearest(self.corpus.embeddings, source_row, STAGE1_LIMIT)

    def _cross_encode(
        self,
        source_row: int,
        candidates: list[int],
        query=lambda doc: doc.text,
        text=lambda doc: doc.text,
        model=None,
        **score_kwargs,
    ) -> list[int]:
        docs = self.corpus.docs
        pairs = [[query(docs[source_row]), text(docs[row])] for row in candidates]
        scores = rerank.score_pairs(self.tokenizer, model or self.model, pairs, **score_kwargs)
        order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
        return [candidates[j] for j in order[:TOP_K]]

    def _cap_query(self, doc: Doc) -> str:
        offsets = self.tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        return doc.text if len(offsets) <= QUERY_CAP_TOKENS else doc.text[: offsets[QUERY_CAP_TOKENS - 1][1]]

    @functools.cached_property
    def _categories_first_embeddings(self) -> np.ndarray:
        texts = [f"{', '.join(doc.categories)} {doc.text}".strip().lower() for doc in self.corpus.docs]
        return sbert_helper.get_model().encode(texts, normalize_embeddings=True, batch_size=64)

    @functools.cached_property
    def _int8_model(self):
        from torch.ao.quantization import quantize_dynamic

        if "fbgemm" not in torch.backends.quantized.supported_engines:  # x86 has fbgemm; ARM (e.g. a Mac) only qnnpack
            torch.backends.quantized.engine = "qnnpack"
        return quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    # --- systems

    def _stage1(self, source_row: int) -> list[int]:
        return self._neighbours(source_row)[:TOP_K]

    def _stage1_categories_first(self, source_row: int) -> list[int]:
        return self._nearest(self._categories_first_embeddings, source_row, TOP_K)

    def _rerank(self, source_row: int) -> list[int]:
        return self._cross_encode(source_row, self._neighbours(source_row))

    def _rerank_titles(self, source_row: int) -> list[int]:
        return self._cross_encode(
            source_row,
            self._neighbours(source_row),
            query=lambda doc: doc.title,
            text=lambda doc: f"{doc.title}. {doc.text}",
        )

    def _rerank_query_cap(self, source_row: int) -> list[int]:
        return self._cross_encode(source_row, self._neighbours(source_row), query=self._cap_query, truncation="only_second")

    def _rerank_strip_boilerplate(self, source_row: int) -> list[int]:
        strip = lambda doc: strip_boilerplate(doc.text)  # noqa: E731
        return self._cross_encode(source_row, self._neighbours(source_row), query=strip, text=strip)

    def _rerank_max_length_256(self, source_row: int) -> list[int]:
        return self._cross_encode(source_row, self._neighbours(source_row), max_length=256)

    def _rerank_top20(self, source_row: int) -> list[int]:
        return self._cross_encode(source_row, self._neighbours(source_row)[:20])

    def _rerank_int8(self, source_row: int) -> list[int]:
        return self._cross_encode(source_row, self._neighbours(source_row), model=self._int8_model)
