import numpy as np
import pytest

from app.utils import sbert_helper


@pytest.mark.model
def test_generate_embedding_is_unit_length():
    embedding = sbert_helper.generate_embedding("the central bank raised interest rates")

    assert len(embedding) == 384
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_generate_embedding_skips_empty_text():
    assert sbert_helper.generate_embedding("") is None
    assert sbert_helper.generate_embedding(None) is None
