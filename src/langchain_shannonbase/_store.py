"""Storage backends behind the vector store.

MySQLStore talks to a real ShannonBase / MySQL 9 / HeatWave instance. InMemoryStore
emulates the same behavior in pure Python (cosine over stored vectors) so the
vector store can be fully unit-tested offline — same pattern that keeps the tests
deterministic and CI-friendly without provisioning a database.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from . import _sql


@dataclass
class Row:
    id: str
    content: str
    metadata: dict
    distance: float


