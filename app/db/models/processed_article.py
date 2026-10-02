# app/db/models/processed_article.py
from sqlalchemy import Column, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from app.db.models.article import Base

class ProcessedArticle(Base):
    __tablename__ = 'processed_articles'
    __table_args__ = (
        # Stage-1 retrieval orders by cosine distance (<=>), which only a vector_cosine_ops index serves.
        Index(
            "ix_processed_articles_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    # article_id is both the PK and a foreign key
    article_id = Column(
        UUID(as_uuid=True),
        sa.ForeignKey("articles.id"),
        primary_key=True,
        nullable=False
    )

    cleaned_text = Column(Text, nullable=True)
    category_1 = Column(Text, nullable=True)
    category_2 = Column(Text, nullable=True)
    processed_at = Column(DateTime(timezone=True),
                          server_default=sa.text('CURRENT_TIMESTAMP'),
                          nullable=False)
    embedding = Column(Vector(384), nullable=True)  # from SBERT

    article = relationship(
        "Article",
        back_populates="processed_article",
        uselist=False
    )
