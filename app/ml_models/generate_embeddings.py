# app/preprocessing/embed_runner.py

import asyncio
import logging
from app.utils.sbert_helper import embed_articles  # async helper

logger = logging.getLogger(__name__)

async def main():
    total_embedded = 0

    while True:
        # Process a batch of articles (returns e.g. "50 articles embedded and saved.")
        result_message = await embed_articles(batch_size=100)

        # Extract the count from the message
        try:
            batch_count = int(result_message.split()[0])
        except (ValueError, IndexError):
            logger.error("Unexpected result message format: %s", result_message)
            break

        total_embedded += batch_count
        logger.info("%s", result_message)

        # Stop when no more articles to embed
        if batch_count == 0:
            break

    logger.info("Done. Total documents embedded: %d", total_embedded)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
