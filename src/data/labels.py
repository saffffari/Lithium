"""Hierarchical label registry for point cloud annotation.

Labels are integer IDs (0-255) with associated metadata (name, color, parent,
visibility, lock). ID 0 is always "Unlabeled" and cannot be deleted.
"""

from dataclasses import dataclass, field, asdict
import json
import numpy as np


# Extended palette — Okabe-Ito colorblind-safe core + warm OP-1 accents.
# 16 entries so a 4x4 swatch grid fills exactly. Ordering: warm accents
# first (for the default-auto-assign loop), then blues/greens, then neutrals.
DEFAULT_PALETTE = [
    (0.976, 0.573, 0.141, 1.0),  # ORANGE (default first label color)
    (0.95, 0.40, 0.30, 1.0),  # CORAL
    (0.95, 0.60, 0.15, 1.0),  # ORANGE
    (1.00, 0.75, 0.20, 1.0),  # AMBER
    (0.90, 0.35, 0.55, 1.0),  # PINK
    (0.84, 0.37, 0.00, 1.0),  # vermillion
    (0.20, 0.90, 0.40, 1.0),  # GREEN
    (0.00, 0.62, 0.45, 1.0),  # bluish green
    (0.15, 0.75, 0.65, 1.0),  # TEAL
    (0.34, 0.71, 0.91, 1.0),  # sky blue
    (0.00, 0.45, 0.70, 1.0),  # deep blue
    (0.60, 0.50, 0.90, 1.0),  # LAVENDER
    (0.80, 0.47, 0.65, 1.0),  # reddish purple
    (0.94, 0.89, 0.26, 1.0),  # yellow
    (0.75, 0.75, 0.75, 1.0),  # light grey
    (0.40, 0.40, 0.40, 1.0),  # dark grey
]

UNLABELED_COLOR = (0.5, 0.5, 0.5, 0.3)
MAX_LABELS = 255


@dataclass
class LabelInfo:
    """Metadata for a single label."""
    id: int
    name: str
    color: tuple  # (r, g, b, a) in [0, 1]
    parent_id: int | None = None
    visible: bool = True
    locked: bool = False

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'color': list(self.color),
            'parent_id': self.parent_id,
            'visible': self.visible,
            'locked': self.locked,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LabelInfo':
        return cls(
            id=data['id'],
            name=data['name'],
            color=tuple(data['color']),
            parent_id=data.get('parent_id'),
            visible=data.get('visible', True),
            locked=data.get('locked', False),
        )


class LabelRegistry:
    """Hierarchical registry of labels for an annotation project."""

    def __init__(self):
        self._labels: dict[int, LabelInfo] = {}
        self._next_id: int = 1
        self._palette_index: int = 0
        # Optional mutation-notify hook. The app installs this to
        # auto-persist the taxonomy to ~/.lithium/schema.json on every
        # add/remove/rename/recolor/visibility/lock change. Default is
        # a no-op so the class stays self-contained for tests and tools
        # that don't care about persistence.
        self._on_change_callback = None
        # ID 0 is always "Unlabeled"
        self._labels[0] = LabelInfo(
            id=0,
            name="Unlabeled",
            color=UNLABELED_COLOR,
            parent_id=None,
            visible=True,
            locked=False,
        )

    def _on_change(self) -> None:
        """Fire the mutation-notify hook if installed. Silent on error."""
        cb = self._on_change_callback
        if cb is None:
            return
        try:
            cb(self)
        except Exception as e:
            print(f"LabelRegistry on_change callback failed: {e}")

    def add_label(self, name: str, color: tuple = None,
                  parent_id: int | None = None) -> int:
        """Add a new label. Returns the new label ID."""
        if len(self._labels) >= MAX_LABELS + 1:  # +1 for unlabeled
            raise ValueError(f"Cannot exceed {MAX_LABELS} labels")
        if parent_id is not None and parent_id not in self._labels:
            raise ValueError(f"Parent label {parent_id} does not exist")
        if color is None:
            color = DEFAULT_PALETTE[self._palette_index % len(DEFAULT_PALETTE)]
            self._palette_index += 1

        label_id = self._next_id
        self._next_id += 1
        self._labels[label_id] = LabelInfo(
            id=label_id,
            name=name,
            color=tuple(color),
            parent_id=parent_id,
        )
        self._on_change()
        return label_id

    def add_label_at(self, label_id: int, name: str, color: tuple,
                     parent_id: int | None = None) -> int:
        """Insert a label at a specific ID. Used for standard ontologies
        (e.g. VERSE) where the integer IDs are part of the convention.
        Silently skips if the ID is already occupied. Returns the label_id.
        """
        if not (1 <= label_id <= MAX_LABELS):
            raise ValueError(f"Label ID {label_id} out of range (1..{MAX_LABELS})")
        if label_id in self._labels:
            return label_id
        if len(self._labels) >= MAX_LABELS + 1:
            raise ValueError(f"Cannot exceed {MAX_LABELS} labels")
        self._labels[label_id] = LabelInfo(
            id=label_id,
            name=name,
            color=tuple(color),
            parent_id=parent_id,
        )
        # Keep _next_id ahead of all occupied IDs
        if label_id >= self._next_id:
            self._next_id = label_id + 1
        self._on_change()
        return label_id

    def remove_label(self, label_id: int) -> bool:
        """Remove a label. Returns True if removed, False if not found or is root."""
        if label_id == 0:
            return False  # Cannot remove unlabeled
        if label_id not in self._labels:
            return False
        # Reparent children to this label's parent
        parent_id = self._labels[label_id].parent_id
        for child in self._labels.values():
            if child.parent_id == label_id:
                child.parent_id = parent_id
        del self._labels[label_id]
        self._on_change()
        return True

    def get(self, label_id: int) -> LabelInfo | None:
        return self._labels.get(label_id)

    def all_labels(self) -> list[LabelInfo]:
        return list(self._labels.values())

    def get_children(self, parent_id: int | None) -> list[LabelInfo]:
        return [l for l in self._labels.values() if l.parent_id == parent_id]

    def get_hierarchy(self) -> list[tuple[LabelInfo, int]]:
        """Return labels in DFS order with depth level. Root labels first."""
        result: list[tuple[LabelInfo, int]] = []

        def dfs(parent: int | None, depth: int):
            for child in sorted(self.get_children(parent), key=lambda l: l.id):
                result.append((child, depth))
                dfs(child.id, depth + 1)

        # Always start with unlabeled at depth 0
        result.append((self._labels[0], 0))
        dfs(None, 0)
        # Remove duplicate root (unlabeled is parent None, but we added it manually)
        seen_ids = set()
        deduped = []
        for label, depth in result:
            if label.id not in seen_ids:
                seen_ids.add(label.id)
                deduped.append((label, depth))
        return deduped

    def set_color(self, label_id: int, color: tuple):
        if label_id in self._labels:
            self._labels[label_id].color = tuple(color)
            self._on_change()

    def set_name(self, label_id: int, name: str):
        if label_id in self._labels and label_id != 0:
            self._labels[label_id].name = name
            self._on_change()

    def set_visible(self, label_id: int, visible: bool):
        if label_id in self._labels:
            self._labels[label_id].visible = visible
            self._on_change()

    def set_locked(self, label_id: int, locked: bool):
        if label_id in self._labels:
            self._labels[label_id].locked = locked
            self._on_change()

    def get_color_lut(self) -> np.ndarray:
        """Return (256, 4) float32 RGBA texture data for GPU label lookup.

        Unused label slots default to the unlabeled color with alpha=0.
        Hidden labels have alpha=0. Locked labels are desaturated toward
        grey so the user can spot them in the viewport at a glance.
        """
        lut = np.zeros((256, 4), dtype=np.float32)
        lut[0] = UNLABELED_COLOR
        for label_id, info in self._labels.items():
            if 0 <= label_id < 256:
                color = list(info.color)
                if len(color) == 3:
                    color.append(1.0)
                if info.locked:
                    # Mix toward luminance (Rec. 709) by 55% — faded but
                    # still recognizably the original color
                    lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                    color[0] = color[0] * 0.45 + lum * 0.55
                    color[1] = color[1] * 0.45 + lum * 0.55
                    color[2] = color[2] * 0.45 + lum * 0.55
                if not info.visible:
                    color[3] = 0.0
                lut[label_id] = color
        return lut

    def to_json(self) -> dict:
        return {
            'version': 1,
            'next_id': self._next_id,
            'palette_index': self._palette_index,
            'labels': [l.to_dict() for l in self._labels.values()],
        }

    @classmethod
    def from_json(cls, data: dict) -> 'LabelRegistry':
        registry = cls()
        registry._labels.clear()
        registry._next_id = max(1, min(int(data.get('next_id', 1)), MAX_LABELS + 1))
        registry._palette_index = data.get('palette_index', 0)
        skipped = []
        for label_data in data.get('labels', []):
            info = LabelInfo.from_dict(label_data)
            # GPU LUT is exactly 256 slots (ids 0..255). Silently dropping
            # out-of-range ids would corrupt the labelmap — warn instead.
            if not (0 <= info.id <= MAX_LABELS):
                skipped.append(info.id)
                continue
            registry._labels[info.id] = info
        if skipped:
            print(
                f"LabelRegistry.from_json: dropped {len(skipped)} labels with "
                f"out-of-range ids (valid: 0..{MAX_LABELS}): {skipped[:8]}"
                f"{'…' if len(skipped) > 8 else ''}"
            )
        # Ensure unlabeled always exists
        if 0 not in registry._labels:
            registry._labels[0] = LabelInfo(
                id=0,
                name="Unlabeled",
                color=UNLABELED_COLOR,
                parent_id=None,
            )
        # Make sure _next_id doesn't alias an existing label and stays in-range.
        while registry._next_id in registry._labels and registry._next_id <= MAX_LABELS:
            registry._next_id += 1
        return registry

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, label_id: int) -> bool:
        return label_id in self._labels


def locked_mask(labels: np.ndarray, registry: LabelRegistry) -> np.ndarray:
    """Return a boolean mask of points whose current label is locked.

    Used to prevent selection tools and apply operations from overwriting
    points that the user has explicitly locked.
    """
    locked_ids = [info.id for info in registry.all_labels() if info.locked]
    if not locked_ids:
        return np.zeros(len(labels), dtype=bool)
    return np.isin(labels, locked_ids)


class LabelCountCache:
    """Per-cloud per-label point counts with explicit invalidation.

    Without this, the label panel rebuilds L*N bool sums every frame.
    Keyed by (id(cloud), label_id). Invalidate on any operation that
    mutates a cloud's labels (apply, undo/redo, labelmap, project load,
    sequence seek).

    Label id ``-1`` is a synthetic key meaning "any non-zero label"
    (the total number of labelled points). The train tab's SELECT
    CLOUDS panel needs that per cloud per frame, so we memoise it
    alongside the per-label counts.
    """

    _ANY_LABEL_ID = -1

    def __init__(self):
        self._counts: dict[tuple[object, int], int] = {}
        self._unique_ids: dict[object, tuple[int, ...]] = {}

    def _cloud_key(self, cloud):
        return getattr(cloud, 'file_path', None) or id(cloud)

    def get(self, cloud, label_id: int) -> int:
        if cloud is None:
            return 0
        key = (self._cloud_key(cloud), int(label_id))
        cached = self._counts.get(key)
        if cached is not None:
            return cached
        count = int((cloud.labels == label_id).sum()) if cloud.labels is not None else 0
        self._counts[key] = count
        return count

    def labeled_count(self, cloud) -> int:
        """Total number of points with any non-zero label."""
        if cloud is None:
            return 0
        key = (self._cloud_key(cloud), self._ANY_LABEL_ID)
        cached = self._counts.get(key)
        if cached is not None:
            return cached
        count = int((cloud.labels != 0).sum()) if cloud.labels is not None else 0
        self._counts[key] = count
        return count

    def any_labeled(self, cloud) -> bool:
        return self.labeled_count(cloud) > 0

    def unique_label_ids(self, cloud) -> tuple[int, ...]:
        """Sorted tuple of distinct non-zero label IDs present in the cloud.

        Cached per cloud; invalidated alongside the per-label counts on
        any label mutation. ``np.unique`` on millions of points is too
        expensive to run per-frame uncached.
        """
        if cloud is None or cloud.labels is None:
            return ()
        key = self._cloud_key(cloud)
        cached = self._unique_ids.get(key)
        if cached is not None:
            return cached
        ids = np.unique(cloud.labels)
        ids = ids[ids != 0]
        out = tuple(int(x) for x in ids)
        self._unique_ids[key] = out
        return out

    def invalidate_cloud(self, cloud):
        if cloud is None:
            return
        cloud_key = self._cloud_key(cloud)
        for k in list(self._counts.keys()):
            if k[0] == cloud_key:
                del self._counts[k]
        self._unique_ids.pop(cloud_key, None)

    def clear(self):
        self._counts.clear()
        self._unique_ids.clear()


# ---------------------------------------------------------------------------
# Standard ontologies — preset label registries for common datasets
# ---------------------------------------------------------------------------

def _lerp_color(a: tuple, b: tuple, t: float) -> tuple:
    """Linear interpolate between two RGBA colors."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        1.0,
    )


def build_verse_ontology() -> LabelRegistry:
    """Build a LabelRegistry pre-populated with the VERSE vertebral
    segmentation convention.

    VERSE label IDs:
        1-7    C1-C7       (cervical)
        8-19   T1-T12      (thoracic)
        20-24  L1-L5       (lumbar)
        25     L6           (anatomical variant)
        26     Sacrum
        27     S2
        28     Coccyx

    Additional IDs beyond VERSE for spinal metrics:
        50-55  Disc T12-L1 through Disc L5-S1
        60-61  Femoral Head L / R

    Hierarchy groups use IDs 240-244 (visible=False, not paintable).
    """
    reg = LabelRegistry()

    # --- Color ramps by region ---
    # Cervical: steel blues
    c_lo = (0.45, 0.60, 0.80, 1.0)
    c_hi = (0.25, 0.42, 0.70, 1.0)
    # Thoracic: sky → deep blue
    t_lo = (0.40, 0.72, 0.92, 1.0)
    t_hi = (0.10, 0.38, 0.72, 1.0)
    # Lumbar: teal → green
    l_lo = (0.20, 0.85, 0.65, 1.0)
    l_hi = (0.08, 0.55, 0.38, 1.0)
    # Discs: amber → orange
    d_lo = (1.00, 0.80, 0.30, 1.0)
    d_hi = (0.92, 0.55, 0.15, 1.0)
    # Sacropelvic: magenta → pink
    s_col = (0.85, 0.35, 0.55, 1.0)
    fem_col = (0.90, 0.45, 0.65, 1.0)

    # --- Group headers (not paintable, just for tree organization) ---
    groups = [
        (240, "Cervical",     c_lo),
        (241, "Thoracic",     t_lo),
        (242, "Lumbar",       l_lo),
        (243, "Discs",        d_lo),
        (244, "Sacropelvic",  s_col),
    ]
    for gid, gname, gcol in groups:
        reg.add_label_at(gid, gname, gcol)
        reg.set_visible(gid, False)

    # --- Cervical C1-C7 (IDs 1-7) ---
    for i in range(7):
        t = i / 6.0 if i < 6 else 1.0
        reg.add_label_at(i + 1, f"C{i + 1}", _lerp_color(c_lo, c_hi, t),
                         parent_id=240)

    # --- Thoracic T1-T12 (IDs 8-19) ---
    for i in range(12):
        t = i / 11.0
        reg.add_label_at(i + 8, f"T{i + 1}", _lerp_color(t_lo, t_hi, t),
                         parent_id=241)

    # --- Lumbar L1-L5 (IDs 20-24), L6 variant (ID 25) ---
    for i in range(5):
        t = i / 4.0
        reg.add_label_at(i + 20, f"L{i + 1}", _lerp_color(l_lo, l_hi, t),
                         parent_id=242)
    reg.add_label_at(25, "L6", _lerp_color(l_lo, l_hi, 1.0), parent_id=242)

    # --- Sacropelvic (IDs 26-28) ---
    reg.add_label_at(26, "Sacrum", s_col, parent_id=244)
    reg.add_label_at(27, "S2", (0.75, 0.32, 0.50, 1.0), parent_id=244)
    reg.add_label_at(28, "Coccyx", (0.65, 0.30, 0.45, 1.0), parent_id=244)

    # --- Disc labels (IDs 50-55, beyond VERSE range) ---
    disc_names = [
        "Disc T12-L1", "Disc L1-L2", "Disc L2-L3",
        "Disc L3-L4", "Disc L4-L5", "Disc L5-S1",
    ]
    for i, dname in enumerate(disc_names):
        t = i / max(1, len(disc_names) - 1)
        reg.add_label_at(50 + i, dname, _lerp_color(d_lo, d_hi, t),
                         parent_id=243)

    # --- Femoral heads (IDs 60-61) ---
    reg.add_label_at(60, "Femoral Head L", fem_col, parent_id=244)
    reg.add_label_at(61, "Femoral Head R",
                     (0.88, 0.50, 0.70, 1.0), parent_id=244)

    return reg


def build_endplate_ontology() -> LabelRegistry:
    """Vertebral sub-structure ontology for endplate / pedicle / facet
    segmentation.

    Six classes — bilateral symmetry uses matched hue pairs so left/right
    are visually related but distinguishable.

    IDs:
        1  Superior Endplate
        2  Inferior Endplate
        3  Left Pedicle
        4  Right Pedicle
        5  Left Facet
        6  Right Facet
    """
    reg = LabelRegistry()
    reg.add_label_at(1, "Sup Endplate",  (0.90, 0.35, 0.05, 1.0))  # deep orange
    reg.add_label_at(2, "Inf Endplate",  (0.10, 0.45, 0.85, 1.0))  # rich blue
    reg.add_label_at(3, "Left Pedicle",  (0.05, 0.70, 0.30, 1.0))  # vivid green
    reg.add_label_at(4, "Right Pedicle", (0.50, 0.75, 0.05, 1.0))  # dark lime
    reg.add_label_at(5, "Left Facet",    (0.60, 0.15, 0.75, 1.0))  # deep purple
    reg.add_label_at(6, "Right Facet",   (0.85, 0.15, 0.40, 1.0))  # hot magenta
    return reg


def build_clinical_endplate_ontology() -> LabelRegistry:
    """Minimum-viable clinical-measurement ontology for spine analysis.

    Five classes designed to cover ~13 of the 16 measurements in the
    SpineLab project's measurement_dependency_matrix using only surface
    segmentation + post-process clustering. Pairs with a downstream
    ``landmark_extraction`` module that turns predicted "Process Tip"
    points into named anatomical landmarks (SP, L-TP, R-TP) by
    DBSCAN-clustering the predictions and identifying each cluster's
    role from its position in the local vertebral frame (which itself
    derives from the endplates).

    IDs:
        1  Superior Endplate          paint to the rim, not just the central
                                       flat region — corners are derived from
                                       the rim downstream.
        2  Inferior Endplate          same.
        3  S1 Superior Endplate       sacral; geometrically distinct, drives
                                       PI / PT / SS independently.
        4  Process Tip                combined SP + TP + facet tips. Paint a
                                       small cluster (~30-50 points) at each
                                       visible tip; downstream clustering
                                       splits them back out.

    Annotation cost per vertebra: ~3-5 minutes (vs. ~25-40 for full
    process-surface labeling). Severe tip imbalance is expected
    (~0.25% of points per cluster); the launch path's auto class_weights
    of (0.05, 1.0, 1.0, 1.0, 5.0) handles it.
    """
    reg = LabelRegistry()
    reg.add_label_at(1, "Sup Endplate",   (0.90, 0.35, 0.05, 1.0))  # deep orange
    reg.add_label_at(2, "Inf Endplate",   (0.10, 0.45, 0.85, 1.0))  # rich blue
    reg.add_label_at(3, "S1 Sup Endplate", (0.95, 0.75, 0.10, 1.0)) # warm gold
    reg.add_label_at(4, "Process Tip",     (0.20, 0.85, 0.55, 1.0)) # vivid mint
    return reg


# Registry of available ontology presets — name → builder function
ONTOLOGY_PRESETS: dict[str, callable] = {
    "VERSE Spine": build_verse_ontology,
    "Endplate": build_endplate_ontology,
    "Clinical Endplate + Tips": build_clinical_endplate_ontology,
}
