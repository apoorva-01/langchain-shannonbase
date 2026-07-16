"""Metadata filters, compiled two ways: to a Python predicate for the in-memory
backend, and to a parameterized SQL WHERE fragment for MySQL.

    {"topic": "refunds"}                     equality
    {"topic": {"$in": ["a", "b"]}}           membership
    {"views": {"$gt": 100, "$lte": 500}}     numeric comparison
    {"topic": {"$ne": "spam"}}               inequality

Values are always bound as parameters; only validated identifiers reach the SQL.
"""
from __future__ import annotations

from typing import List, Tuple

from ._sql import ident

_OPS = {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"}


def _check(op: str) -> None:
    if op not in _OPS:
        raise ValueError(f"unknown filter operator {op!r}; use one of {sorted(_OPS)}")


def matches(flt: dict, meta: dict) -> bool:
    for key, cond in flt.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, operand in cond.items():
                _check(op)
                if op == "$eq" and not (val == operand):
                    return False
                if op == "$ne" and not (val != operand):
                    return False
                if op == "$in" and val not in operand:
                    return False
                if op == "$nin" and val in operand:
                    return False
                if op in ("$gt", "$gte", "$lt", "$lte"):
                    if val is None:
                        return False
                    if op == "$gt" and not (val > operand):
                        return False
                    if op == "$gte" and not (val >= operand):
                        return False
                    if op == "$lt" and not (val < operand):
                        return False
                    if op == "$lte" and not (val <= operand):
                        return False
        elif val != cond:
            return False
    return True


def to_sql(flt: dict, metadata_col: str) -> Tuple[List[str], list]:
    clauses: List[str] = []
    params: list = []
    for key, cond in flt.items():
        ident(key)
        text = f"`{metadata_col}`->>'$.{key}'"          # JSON_UNQUOTE, for text compares
        number = f"JSON_EXTRACT(`{metadata_col}`, '$.{key}')"  # keeps JSON numbers numeric
        if not isinstance(cond, dict):
            clauses.append(f"{text} = %s")
            params.append(cond)
            continue
        for op, operand in cond.items():
            _check(op)
            if op == "$eq":
                clauses.append(f"{text} = %s")
                params.append(operand)
            elif op == "$ne":
                clauses.append(f"({text} <> %s OR {text} IS NULL)")
                params.append(operand)
            elif op == "$in":
                ph = ", ".join(["%s"] * len(operand))
                clauses.append(f"{text} IN ({ph})")
                params.extend(operand)
            elif op == "$nin":
                ph = ", ".join(["%s"] * len(operand))
                clauses.append(f"({text} NOT IN ({ph}) OR {text} IS NULL)")
                params.extend(operand)
            elif op == "$gt":
                clauses.append(f"{number} > %s")
                params.append(operand)
            elif op == "$gte":
                clauses.append(f"{number} >= %s")
                params.append(operand)
            elif op == "$lt":
                clauses.append(f"{number} < %s")
                params.append(operand)
            elif op == "$lte":
                clauses.append(f"{number} <= %s")
                params.append(operand)
    return clauses, params
