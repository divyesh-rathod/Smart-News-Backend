import pytest


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("PUT", "/api/V1/news/like/not-a-uuid", {"liked": True}),
        ("POST", "/api/V1/news/mark-as-read/not-a-uuid", None),
    ],
)
def test_a_malformed_article_id_is_rejected_before_reaching_the_database(client, method, path, body):
    response = client.request(method, path, json=body)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["path", "article_id"]
