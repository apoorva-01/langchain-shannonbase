import json

import pytest

from langchain_shannonbase import _sql


def test_vector_literal_is_json_array():
    assert json.loads(_sql.vector_literal([1, 2.5, 3])) == [1.0, 2.5, 3.0]


def test_create_table_has_vector_dim():
    sql = _sql.create_table_sql("docs", 384)
    assert "VECTOR(384)" in sql
    assert "`docs`" in sql


def test_insert_wraps_embedding_in_string_to_vector():
    assert "STRING_TO_VECTOR(%s)" in _sql.insert_sql("docs")
    assert "ON DUPLICATE KEY UPDATE" in _sql.insert_sql("docs")


def test_search_uses_distance_and_metric():
    sql = _sql.search_sql("docs", "cosine")
    assert "DISTANCE(embedding, STRING_TO_VECTOR(%s), 'COSINE')" in sql
    assert "ORDER BY dist ASC LIMIT %s" in sql


def test_search_rejects_unknown_metric():
    with pytest.raises(ValueError):
        _sql.search_sql("docs", "nope")


def test_delete_has_one_placeholder_per_id():
    assert _sql.delete_sql("docs", 3).count("%s") == 3


def test_distance_to_score():
    assert _sql.distance_to_score(0.0) == 1.0
    assert _sql.distance_to_score(0.25) == 0.75
