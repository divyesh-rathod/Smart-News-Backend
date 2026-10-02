# app/ml_models/rerank.py

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512
BATCH_SIZE = 16

# Every forward pass runs on this one thread. Concurrent requests queue here instead of each running
# torch's intra-op thread pool on the same cores at once.
_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cross-encoder")


@functools.cache
def get_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the cross-encoder on first use, so importing this module doesn't download weights."""
    if settings.TORCH_NUM_THREADS:
        torch.set_num_threads(settings.TORCH_NUM_THREADS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(settings.MODEL_DEVICE).eval()
    logger.info("Loaded %s on %s with %d torch threads", MODEL_NAME, settings.MODEL_DEVICE, torch.get_num_threads())
    return tokenizer, model


def score_pairs(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    pairs: list[list[str]],
    batch_size: int = BATCH_SIZE,
) -> list[float]:
    """Cross-encoder score for each [query, text] pair, in input order."""
    # Batches of similar length pad less than one batch padded to the longest pair.
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    scores = [0.0] * len(pairs)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            inputs = tokenizer(
                [pairs[i] for i in batch],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(model.device)
            logits = model(**inputs).logits
            if logits.size(-1) == 1:
                batch_scores = logits.squeeze(-1)
            else:
                batch_scores = F.softmax(logits, dim=1)[:, 1]
            for i, score in zip(batch, batch_scores.float().cpu().tolist()):
                scores[i] = score
    return scores


def _score_with_loaded_model(pairs: list[list[str]]) -> list[float]:
    tokenizer, model = get_model()
    return score_pairs(tokenizer, model, pairs)


async def rerank_top_k(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Score every (query, candidate text) pair with the cross-encoder and return the top_n, best first."""
    pairs = [[query, c["cleaned_text"] or ""] for c in candidates]
    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(_inference_executor, _score_with_loaded_model, pairs)

    scored_with_metadata = [
        {
            'article_id': candidate['article_id'],
            'cleaned_text': candidate['cleaned_text'],
            'category_1': candidate['category_1'],
            'category_2': candidate['category_2'],
            'title': candidate['title'],
            'link': candidate['link'],
            'score': score
        }
        for candidate, score in zip(candidates, scores)
    ]

    scored_with_metadata.sort(key=lambda x: x['score'], reverse=True)
    logger.debug("Reranked %d candidates, returning top %d", len(candidates), top_n)
    return scored_with_metadata[:top_n]
