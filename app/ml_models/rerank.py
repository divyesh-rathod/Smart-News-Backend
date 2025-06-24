# app/ml_models/rerank.py

import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import List, Tuple
from app.db.models.processed_article import ProcessedArticle

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

async def rerank_top_k(
    query: str,
    candidates: List[Tuple[ProcessedArticle, str, str, float]],
    top_n: int = 5,
    device: str | None = None
) -> List[Tuple[ProcessedArticle, str, str, float]]:
    
 
    print(f"🔍 Starting rerank with {len(candidates)} candidates")
    print(f"🔍 First candidate type: {type(candidates[0])}")
    print(f"🔍 Query text length: {len(query)}")
    
    try:
        # Dictionary access (this worked before)
        processed_articles = [{"cleaned_text": c["cleaned_text"]} for c in candidates]
        print(f"🔍 Extracted {len(processed_articles)} processed articles")
    except Exception as e:
        print(f"❌ Dictionary extraction failed: {e}")
        return []

    try:
        # ❌ THIS LINE IS PROBABLY FAILING:
        texts = [cand.cleaned_text or "" for cand in processed_articles]
        print(f"🔍 Extracted {len(texts)} texts")
    except Exception as e:
        print(f"❌ Text extraction failed: {e}")
        # ✅ FIX: Use dictionary access instead
        texts = [item["cleaned_text"] or "" for item in processed_articles]
        print(f"🔍 Fixed: Extracted {len(texts)} texts using dict access")

    try:
        pairs = [[query, txt] for txt in texts]
        print(f"🔍 Created {len(pairs)} query-text pairs")
    except Exception as e:
        print(f"❌ Pairs creation failed: {e}")
        return []

    try:
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        print(f"🔍 Tokenization successful, input shape: {inputs['input_ids'].shape}")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return []

    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("🔍 Moved inputs to device successfully")
    except Exception as e:
        print(f"❌ Device move failed: {e}")
        return []

    loop = asyncio.get_running_loop()

    def _sync_inference():
        print("🔍 Starting inference in thread pool")
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                print("🔍 Model forward pass successful")
                logits = outputs.logits
                print(f"🔍 Logits shape: {logits.shape}")
                
                if logits.size(-1) == 1:
                    scores_tensor = logits.squeeze(-1)
                else:
                    scores_tensor = F.softmax(logits, dim=1)[:, 1]
                
                scores = scores_tensor.cpu().tolist()
                print(f"🔍 Generated {len(scores)} scores")
                return scores
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return []

    try:
        scores = await loop.run_in_executor(None, _sync_inference)
        print(f"🔍 Thread pool execution completed, got {len(scores)} scores")
    except Exception as e:
        print(f"❌ Thread pool execution failed: {e}")
        return []

    try:
        scored_with_metadata = []
        for i, score in enumerate(scores):
            # ❌ THIS LINE WILL ALSO FAIL:
            # art_obj, title, link, _ = candidates[i]  # candidates are dicts now!
            
            # ✅ FIX: Use dictionary access
            candidate = candidates[i]
            scored_with_metadata.append({
                'article_id': candidate['article_id'],
                'cleaned_text': candidate['cleaned_text'],
                'category_1': candidate['category_1'],
                'category_2': candidate['category_2'],
                'title': candidate['title'],
                'link': candidate['link'],
                'score': score
            })
        
        print(f"🔍 Created {len(scored_with_metadata)} scored items")
    except Exception as e:
        print(f"❌ Metadata reconstruction failed: {e}")
        return []

    try:
        # Sort by score (highest first) and return top_n
        scored_with_metadata.sort(key=lambda x: x['score'], reverse=True)
        result = scored_with_metadata[:top_n]
        print(f"🔍 Returning top {len(result)} results")
        return result
    except Exception as e:
        print(f"❌ Sorting/slicing failed: {e}")
        return []