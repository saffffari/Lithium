"""Registry of committed measurements (line distances and angles).

Each completed measurement session produces a MeasureItem stored here.
Items can be organised into MeasureGroup folders. The registry is
in-memory; no cross-session persistence (measurements reset on reload).

Anchor model
------------
Anchors live in **cloud-local** space: each anchor stores the
``file_key`` of the cloud it was placed on plus a 3-vector in that
cloud's mesh-local frame. To get its current world position, callers
pass a resolver callable that knows how to transform local → world
using the current scene state (in HOLOGRAM the model matrix composes
``reference_local_model(cloud_pose, ref_pose)``; in LIGHT TABLE it
composes the entry's ``model_transform``).

The cloud-local representation is what makes measurements survive
reference-vertebra switches in HOLOGRAM — when the user re-anchors
the scene to a different bone, every other bone's world transform
changes, but the per-bone local frames don't. World-space anchors
detach from their bones in that scenario; cloud-local anchors follow.

Usage::

    registry = MeasureRegistry()
    a0 = Anchor(cloud_key=key_l4, local_pos=np.array([1.0, 2.0, 3.0]))
    a1 = Anchor(cloud_key=key_l5, local_pos=np.array([0.5, 1.0, 0.0]))
    item = registry.add_item('measure_line', [a0, a1], color)

    # At render / display time, supply a resolver:
    def resolve(anchor):
        entry = entries_by_key.get(anchor.cloud_key)
        if entry is None:
            return None
        return (entry.current_model @ [*anchor.local_pos, 1.0])[:3]
    value = item.get_value(resolve)
"""

import uuid
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Default measurement colour — same red as the live overlay.
MEASURE_COLOR = (1.0, 0.15, 0.15, 1.0)


# --------------------------------------------------------------------------
# Anchor + resolver
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Anchor:
    """A measurement anchor in cloud-local coordinates.

    ``cloud_key`` identifies the cloud the anchor lives on; ``local_pos``
    is a 3-vector in that cloud's mesh-local frame (millimetres). The
    actual world position depends on the cloud's current model matrix,
    which is mode- and reference-frame-dependent — so we never store
    world coordinates directly. Use ``AnchorResolver`` to look up the
    world position at any moment.

    Immutable (frozen) so we can share anchors freely between the live
    session and the registry without aliasing surprises. A retargeted
    anchor (user drags it elsewhere) is created as a new Anchor.
    """
    cloud_key: str
    local_pos: np.ndarray

    def __post_init__(self):
        # Normalise the stored vector so downstream code can assume a
        # contiguous float32 (3,) without re-checking every read.
        arr = np.asarray(self.local_pos, dtype=np.float32).reshape(3)
        # frozen dataclass requires object.__setattr__ to override.
        object.__setattr__(self, "local_pos", arr)


# Resolver: maps an Anchor to its current world position, or None when
# the cloud no longer exists (e.g. user removed it between commit and
# render). Renderers / value calculators are expected to skip None.
AnchorResolver = Callable[[Anchor], "np.ndarray | None"]


def make_resolver(entries_by_key: dict, model_of: Callable | None = None):
    """Build a resolver that transforms anchors to world space.

    ``entries_by_key`` is ``{file_key: entry}`` for every cloud
    currently in scene. ``model_of(entry) -> np.ndarray(4,4)`` returns
    the entry's current model matrix; when None we fall back to
    ``entry.model_transform`` (LIGHT TABLE convention). The
    HOLOGRAM render path supplies an explicit ``model_of`` because the
    HOLOGRAM scene composes a reference-local matrix that lives only
    in the render loop, not on the entry.
    """
    def _resolve(anchor: Anchor) -> "np.ndarray | None":
        entry = entries_by_key.get(anchor.cloud_key)
        if entry is None:
            return None
        if model_of is not None:
            M = model_of(entry)
        else:
            M = getattr(entry, "model_transform", None)
            if M is None:
                M = np.eye(4, dtype=np.float32)
        if M is None:
            return None
        h = np.array(
            [anchor.local_pos[0], anchor.local_pos[1], anchor.local_pos[2], 1.0],
            dtype=np.float64,
        )
        return (np.asarray(M, dtype=np.float64) @ h)[:3].astype(np.float32)
    return _resolve


# --------------------------------------------------------------------------
# Pick result — what the 3D-viewport pickers return
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PickResult:
    """The result of clicking in a 3D viewport.

    Carries both the world-space hit position (what the cursor is
    visually pointing at) AND the per-cloud provenance needed to
    create a cloud-local Anchor from it. Callers that only need the
    world position read ``.world_pos``; callers that want to commit
    a measurement build an Anchor via ``.as_anchor()``.
    """
    world_pos: np.ndarray
    cloud_key: str
    local_pos: np.ndarray
    entry: object | None = None

    def as_anchor(self) -> Anchor:
        return Anchor(cloud_key=self.cloud_key, local_pos=self.local_pos)


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Angle in degrees at point B in the A-B-C chain."""
    ba = a - b
    bc = c - b
    la = float(np.linalg.norm(ba))
    lc = float(np.linalg.norm(bc))
    if la < 1e-12 or lc < 1e-12:
        return None
    cos_a = float(np.dot(ba, bc)) / (la * lc)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


class MeasureItem:
    """A single committed measurement (line or angle).

    Anchors are stored in cloud-local space (see ``Anchor``). To get
    a value (distance / angle) or a world position, callers supply an
    ``AnchorResolver`` that knows the current scene state — typically
    obtained from ``make_resolver(entries_by_key, model_of=...)``.

    Why the resolver pattern: a measurement spanning two bones has to
    follow both bones through HOLOGRAM reference-frame switches. The
    only way to do that is to resolve the world position at the moment
    of need, not freeze it at creation time.
    """

    def __init__(self, mode: str, anchors: list, name: str = "",
                 color: tuple = MEASURE_COLOR):
        self.id: str = str(uuid.uuid4())
        self.mode: str = mode   # 'measure_line' | 'measure_angle' | 'measure_landmark'
        # Anchors are immutable; we accept anything Anchor-shaped (the
        # MeasureState commit path passes Anchor objects directly) but
        # also tolerate a (cloud_key, local_pos) pair tuple as a small
        # convenience for tests and external callers.
        self.anchors: list[Anchor] = [_as_anchor(a) for a in anchors]
        self.name: str = name
        self.color: tuple = color
        self.group_id: str | None = None   # None → ungrouped

    def get_value(self, resolve: AnchorResolver) -> float | None:
        """Return the scalar measurement value (distance, angle °, or None for landmarks).

        Returns None when any anchor's cloud can't be resolved
        (cloud removed from project, etc.) — caller treats that as
        "n/a" rather than a zero-valued measurement.
        """
        if self.mode == 'measure_line':
            if len(self.anchors) < 2:
                return None
            a = resolve(self.anchors[0])
            b = resolve(self.anchors[1])
            if a is None or b is None:
                return None
            return float(np.linalg.norm(b - a))
        elif self.mode == 'measure_angle':
            if len(self.anchors) < 3:
                return None
            a = resolve(self.anchors[0])
            b = resolve(self.anchors[1])
            c = resolve(self.anchors[2])
            if a is None or b is None or c is None:
                return None
            return _angle_at_vertex(a, b, c)
        # Landmarks have no scalar value — they're just a named position.
        return None

    def get_position(self, resolve: AnchorResolver) -> np.ndarray | None:
        """Return the 3D world position (landmarks only). None if
        the anchor's cloud can't be resolved."""
        if self.mode == 'measure_landmark' and len(self.anchors) >= 1:
            return resolve(self.anchors[0])
        return None

    def resolved_anchors(
        self, resolve: AnchorResolver,
    ) -> list["np.ndarray | None"]:
        """Bulk-resolve every anchor to world. Renderers use this to
        do screen projection without re-resolving per call."""
        return [resolve(a) for a in self.anchors]


def _as_anchor(a) -> Anchor:
    """Coerce ``a`` to an Anchor. Accepts Anchor (pass-through) or a
    ``(cloud_key, local_pos)`` 2-tuple. Anything else is a programming
    error and raises — we want the type discipline now that anchors
    aren't bare ndarrays anymore."""
    if isinstance(a, Anchor):
        return a
    if isinstance(a, tuple) and len(a) == 2:
        cloud_key, local_pos = a
        return Anchor(cloud_key=str(cloud_key), local_pos=np.asarray(local_pos))
    raise TypeError(
        f"Expected Anchor or (cloud_key, local_pos) tuple; got {type(a).__name__}. "
        "Bare ndarrays are no longer accepted — convert to Anchor first."
    )


class MeasureGroup:
    """A collapsible folder grouping related measurements."""

    def __init__(self, name: str):
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.expanded: bool = True


class MeasureRegistry:
    """Ordered list of MeasureItems + MeasureGroups.

    Items are stored in insertion order. Groups are separate; each item
    carries a group_id reference. Ungrouped items have group_id = None.
    """

    def __init__(self):
        self.items: list[MeasureItem] = []
        self.groups: list[MeasureGroup] = []
        self._line_counter: int = 0
        self._angle_counter: int = 0
        self._landmark_counter: int = 0

    # ------------------------------------------------------------------
    # Auto-naming helpers
    # ------------------------------------------------------------------

    def _auto_name(self, mode: str) -> str:
        if mode == 'measure_line':
            self._line_counter += 1
            return f"Distance {self._line_counter}"
        elif mode == 'measure_angle':
            self._angle_counter += 1
            return f"Angle {self._angle_counter}"
        else:
            self._landmark_counter += 1
            return f"Landmark {self._landmark_counter}"

    # ------------------------------------------------------------------
    # Item management
    # ------------------------------------------------------------------

    def add_item(self, mode: str, anchors: list,
                 color: tuple = MEASURE_COLOR) -> MeasureItem:
        """Create and append a new MeasureItem. Returns the new item."""
        name = self._auto_name(mode)
        item = MeasureItem(mode, anchors, name, color)
        self.items.append(item)
        return item

    def remove_items(self, ids: set) -> None:
        """Remove all items whose id is in `ids`."""
        self.items = [it for it in self.items if it.id not in ids]

    def get_item(self, item_id: str) -> MeasureItem | None:
        for it in self.items:
            if it.id == item_id:
                return it
        return None

    # ------------------------------------------------------------------
    # Group management
    # ------------------------------------------------------------------

    def add_group(self, name: str | None = None) -> MeasureGroup:
        """Create and append a new MeasureGroup. Returns the new group."""
        n = name or f"Group {len(self.groups) + 1}"
        grp = MeasureGroup(n)
        self.groups.append(grp)
        return grp

    def remove_group(self, group_id: str, ungroup_items: bool = True) -> None:
        """Remove a group. If ungroup_items, items become ungrouped."""
        self.groups = [g for g in self.groups if g.id != group_id]
        if ungroup_items:
            for it in self.items:
                if it.group_id == group_id:
                    it.group_id = None

    def get_group(self, group_id: str) -> MeasureGroup | None:
        for g in self.groups:
            if g.id == group_id:
                return g
        return None

    def items_in_group(self, group_id: str | None) -> list:
        """Items belonging to a group (None = ungrouped)."""
        return [it for it in self.items if it.group_id == group_id]

    def send_to_group(self, item_ids: set, group_id: str | None) -> None:
        """Move items to a group (None = ungrouped)."""
        for it in self.items:
            if it.id in item_ids:
                it.group_id = group_id

    # ------------------------------------------------------------------
    # Ordered traversal helper (for the panel tree)
    # ------------------------------------------------------------------

    def flat_tree(self) -> list:
        """Return a display-order list of (kind, object) tuples.

        kind values:
          'group'    → MeasureGroup header
          'item'     → MeasureItem (indented if in a group)
          'ungrouped'→ MeasureItem not in any group (no indent)
        """
        result = []
        for grp in self.groups:
            result.append(('group', grp))
            if grp.expanded:
                for it in self.items_in_group(grp.id):
                    result.append(('item', it))
        for it in self.items_in_group(None):
            result.append(('ungrouped', it))
        return result
