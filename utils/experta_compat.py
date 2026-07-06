"""
Compatibility helpers for Experta's legacy dependency stack.

Experta 1.9.4 pins frozendict 1.2, which still imports Mapping from the
collections module. Python 3.10+ moved that ABC to collections.abc.
"""

from __future__ import annotations

import collections
import collections.abc


def patch_collections_abc() -> None:
    """Expose legacy collections ABC aliases expected by old dependencies."""
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping


patch_collections_abc()
