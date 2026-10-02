# app/preprocessing/embed_runner.py

import asyncio
import logging
from app.utils.sbert_helper import embed_articles  # async helper

logger = logging.getLogger(__name__)

async def main() -> int:
    total_embedded = 0

    # Stop when a batch finds nothing left to embed
    while batch_count := await embed_articles(batch_size=100):
        total_embedded += batch_count
        logger.info("%d articles embedded and saved", batch_count)

    logger.info("Done. Total documents embedded: %d", total_embedded)
    return total_embedded

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
