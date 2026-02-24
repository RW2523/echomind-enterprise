"""
Catalog database: documents_catalog table and connection.
Table created in core.db init_db(); this module re-exports get_conn for catalog use.
"""
from __future__ import annotations
from ..core.db import get_conn

__all__ = ["get_conn"]
