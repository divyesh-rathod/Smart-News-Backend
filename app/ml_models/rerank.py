# app/ml_models/rerank.py

import asyncio
import functools
import logging

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@functools.cache
def get_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the cross-encoder on first use, so importing this module doesn't download weights."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model


async def rerank_top_k(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
    device: str | None = None
) -> list[dict]:
    """Score every (query, candidate text) pair with the cross-encoder and return the top_n, best first."""
    tokenizer, model = get_model()
    texts = [c["cleaned_text"] or "" for c in candidates]
    pairs = [[query, text] for text in texts]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    def _sync_inference() -> list[float]:
        with torch.no_grad():
            logits = model(**inputs).logits

        if logits.size(-1) == 1:
            scores_tensor = logits.squeeze(-1)
        else:
            scores_tensor = F.softmax(logits, dim=1)[:, 1]

        return scores_tensor.cpu().tolist()

    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(None, _sync_inference)

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
