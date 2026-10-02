# app/db/models/user_article_like.py

from sqlalchemy import Column, ForeignKey, SmallInteger, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class Like(Base):
    __tablename__ = "like"

    # Composite primary key: (user_id, article_id)
    user_id    = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    article_id = Column(
        UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )


    is_liked = Column(SmallInteger, nullable=False,server_default=text("0"))

    user    = relationship("User", back_populates="article_likes", uselist=False)
    article = relationship("Article", back_populates="user_likes", uselist=False)
