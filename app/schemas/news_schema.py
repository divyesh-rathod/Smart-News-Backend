from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID


class UnseenArticlesQuery(BaseModel):

    cursor: Optional[datetime] = Field(
        None,
        description="Fetch only articles with pub_date < this timestamp (RFC3339)."
    )
    limit: int = Field(
        20,
        ge=1, le=100,
        description="Maximum number of articles to return. Defaults to 20."
    )
    offset:  int = Field(
        0, ge=0,
        description="Number of articles to skip (for paging)."
    )

class UnseenProcessedArticle(BaseModel):
  
    article_id: UUID
    cleaned_text: str
    category_1: Optional[str]
    category_2: Optional[str]
    processed_at: datetime
    pub_date:    datetime
    title:       str
    link:        str
    description: Optional[str]
    categories:  Optional[List[str]]

    class Config:
        from_attributes = True
        validate_by_name = True


class UnseenArticlesResponse(BaseModel):
  
    results:     List[UnseenProcessedArticle]
    next_cursor: Optional[datetime] = None


class UpdateLastReadRequest(BaseModel):
    # Optionally you could let the client specify a custom timestamp,
    # or simply ignore it and set `last_read_date = now()` server‐side.
    last_read_date: datetime | None = None





class ArticleBase(BaseModel):
    article_id: UUID
    title: str
    link: str
    cleaned_text: str
    category_1: Optional[str] = None
    category_2: Optional[str] = None

class ArticleScore(ArticleBase):
    score: float

    class Config:
        from_attributes = True  # if you ever want to return ORM models directly

class ToggleLikeResponse(BaseModel):
    message: str
    liked: bool
    top5: List[ArticleScore]
    similar: List[ArticleScore]

