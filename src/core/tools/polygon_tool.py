"""Polygon lasso tool — click-to-place vertices, double-click to close.

Uses the same winding-number test as the freeform lasso but with
user-defined discrete vertices.
"""

from src.core.tools.lasso_tool import lasso_select as _polygon_select


polygon_select = _polygon_select
