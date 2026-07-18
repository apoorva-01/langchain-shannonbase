import json

import pytest

from langchain_shannonbase import _sql

S = _sql.Schema("docs")


def test_vector_literal_is_json_array():
    assert json.loads(_sql.vector_literal([1, 2.5, 3])) == [1.0, 2.5, 3.0]


def test_create_table_has_vector_dim():
    sql = _sql.create_table_sql(S, 384)
    assert "VECTOR(384)" in sql
    assert "`docs`" in sql


def test_insert_wraps_embedding_in_string_to_vector():
    assert "STRING_TO_VECTOR(%s)" in _sql.insert_sql(S)
    assert "ON DUPLICATE KEY UPDATE" in _sql.insert_sql(S)


def test_search_uses_distance_and_metric():
    sql = _sql.search_sql(S, "cosine")
    assert "DISTANCE(`embedding`, STRING_TO_VECTOR(%s), 'COSINE')" in sql
    assert "ORDER BY dist ASC LIMIT %s" in sql


def test_search_rejects_unknown_metric():
    with pytest.raises(ValueError):
        _sql.search_sql(S, "nope")


def test_delete_has_one_placeholder_per_id():
    assert _sql.delete_sql(S, 3).count("%s") == 3


def test_custom_column_names_are_used():
    s = _sql.Schema("kb", id="doc_id", content="body", embedding="vec")
    sql = _sql.create_table_sql(s, 8)
    assert "`kb`" in sql and "`doc_id`" in sql and "`body`" in sql and "`vec` VECTOR(8)" in sql


def test_bad_identifier_is_rejected():
    with pytest.raises(ValueError):
        _sql.Schema("docs; DROP TABLE users")


def test_distance_to_score():
    assert _sql.distance_to_score(0.0) == 1.0
    assert _sql.distance_to_score(0.25) == 0.75


def test_create_table_has_fulltext_index():
    assert "FULLTEXT (`content`)" in _sql.create_table_sql(S, 8)


def test_keyword_search_uses_fulltext_match():
    sql = _sql.keyword_search_sql(S)
    assert "MATCH(`content`) AGAINST(%s IN NATURAL LANGUAGE MODE)" in sql
    assert "ORDER BY score DESC LIMIT %s" in sql


def test_keyword_search_binds_filter_clauses():
    sql = _sql.keyword_search_sql(S, filter_clauses=["`metadata`->>'$.lang' = %s"])
    # filter clause AND the match predicate: one %s for the SELECT match, one for
    # each filter clause, one for the WHERE match, one for LIMIT.
    assert sql.count("%s") == 4
    assert "lang" in sql
