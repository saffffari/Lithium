"""Hierarchy flattening modes for dataset export.

PTv3 consumes one flat int label per point, so any hierarchical label
registry has to be projected down to a flat id space at export time.
How the projection is done matters — this module implements the four
modes called out in the design doc:

  - "leaves":   each leaf registry id keeps its own class. Any non-leaf
                registry id (a parent with children) is merged into
                its own class too — the walk only collapses UPWARDS
                when explicitly requested by another mode.
  - "depth":    roll every label deeper than ``depth`` up to its
                depth-``depth`` ancestor. Labels at or above ``depth``
                are unchanged. Balances class counts when the user
                has gone too deep with sub-labels.
  - "subtree":  binary mode. Every label inside the chosen subtree
                (including the root) collapses to one "positive"
                class. Everything else becomes IGNORE (not class 0 —
                zero is the positive class, there is no negative
                class to learn). Use when the user wants a
                "vertebra vs not" model.
  - "custom":   user-supplied ``{registry_id_or_path: class_id}``
                mapping. Registry ids not present in the mapping
                become IGNORE. Class ids may be non-contiguous on
                input; ``build_class_map_from_flatten`` will pack
                them down.

All functions here return ``{old_registry_id: effective_registry_id}``
tables. The effective id is whatever registry id should be treated as
"the class" for the purposes of the final remap. Two registry ids that
map to the same effective id end up in the same PTv3 class. Unlabeled
(registry id 0) is never included — it's handled separately as
IGNORE_INDEX by ``build_class_map``.
"""

from src.data.labels import LabelRegistry


def _depth_of(registry: LabelRegistry, label_id: int) -> int:
    """Walk parent_id chain to find this label's depth (root = 0)."""
    depth = 0
    cur = registry.get(label_id)
    # Safety cap at 256 — the registry caps label count at 255 and a
    # cycle would infinite-loop this otherwise. The cap doubles as a
    # sanity check on malformed parent_id graphs.
    for _ in range(256):
        if cur is None or cur.parent_id is None:
            return depth
        depth += 1
        cur = registry.get(cur.parent_id)
    return depth


def _ancestor_at_depth(registry: LabelRegistry, label_id: int,
                       target_depth: int) -> int:
    """Return the ancestor id of ``label_id`` at depth ``target_depth``.

    If the label itself is at or shallower than target_depth, returns
    it unchanged. If we can't reach target_depth (broken chain),
    returns the current id.
    """
    cur_id = label_id
    cur_depth = _depth_of(registry, cur_id)
    while cur_depth > target_depth:
        info = registry.get(cur_id)
        if info is None or info.parent_id is None:
            break
        cur_id = info.parent_id
        cur_depth -= 1
    return cur_id


def _descendants(registry: LabelRegistry, root_id: int) -> set[int]:
    """Return the set of ids in the subtree rooted at ``root_id``.

    Includes ``root_id`` itself. Uses BFS over the parent_id graph
    (inverted on the fly so we don't have to cache children).
    """
    ids: set[int] = {root_id}
    frontier = [root_id]
    while frontier:
        parent = frontier.pop()
        for info in registry.all_labels():
            if info.parent_id == parent and info.id not in ids:
                ids.add(info.id)
                frontier.append(info.id)
    return ids


def flatten_leaves(registry: LabelRegistry) -> dict[int, int]:
    """Identity mapping — every non-unlabeled label keeps its own id."""
    return {info.id: info.id for info in registry.all_labels() if info.id != 0}


def flatten_depth(registry: LabelRegistry, depth: int) -> dict[int, int]:
    """Map each label to its depth-``depth`` ancestor.

    Labels at depth <= ``depth`` are unchanged. Labels deeper than
    ``depth`` walk their parent chain up to the target depth. A label
    at depth 4 with ``depth=2`` ends up mapped to its grandparent.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    out: dict[int, int] = {}
    for info in registry.all_labels():
        if info.id == 0:
            continue
        out[info.id] = _ancestor_at_depth(registry, info.id, depth)
    return out


def flatten_subtree(registry: LabelRegistry,
                    root_id: int) -> dict[int, int]:
    """Binary mode — collapse one subtree into a single class.

    Every label in the subtree rooted at ``root_id`` (inclusive) maps
    to ``root_id``. Labels outside the subtree are NOT in the returned
    mapping, which means ``build_class_map`` will treat them as IGNORE.
    This is the "vertebra vs not" / "tumor vs background" mode — there
    is one positive class and everything else is masked out of the
    loss.
    """
    if registry.get(root_id) is None:
        raise ValueError(f"subtree root {root_id} not in registry")
    if root_id == 0:
        raise ValueError("cannot flatten the Unlabeled subtree")
    inside = _descendants(registry, root_id)
    return {lid: root_id for lid in inside}


def flatten_custom(registry: LabelRegistry,
                   mapping: dict) -> dict[int, int]:
    """Custom mapping from registry ids (or label names) to effective ids.

    ``mapping`` keys may be either ``int`` (registry id) or ``str``
    (label name). Values are registry ids — the "effective" id that
    this label should merge into. Names are resolved once up front;
    unknown names raise.

    Ids not present in the mapping are dropped (treated as IGNORE by
    ``build_class_map``).
    """
    by_name = {info.name: info.id for info in registry.all_labels()}
    out: dict[int, int] = {}
    for key, value in mapping.items():
        if isinstance(key, str):
            if key not in by_name:
                raise ValueError(f"custom flatten: unknown label name '{key}'")
            src_id = by_name[key]
        else:
            src_id = int(key)
        if registry.get(src_id) is None:
            raise ValueError(f"custom flatten: source id {src_id} not in registry")
        dst_id = int(value)
        if registry.get(dst_id) is None:
            raise ValueError(f"custom flatten: target id {dst_id} not in registry")
        if src_id == 0 or dst_id == 0:
            raise ValueError("custom flatten: cannot use Unlabeled (id 0)")
        out[src_id] = dst_id
    return out


def resolve_flatten(registry: LabelRegistry,
                    mode: str = "leaves",
                    depth: int | None = None,
                    subtree_root: int | None = None,
                    custom: dict | None = None) -> dict[int, int]:
    """Dispatch helper. Returns ``{old_id: effective_id}`` for the mode."""
    if mode == "leaves":
        return flatten_leaves(registry)
    if mode == "depth":
        if depth is None:
            raise ValueError("flatten mode 'depth' requires depth=")
        return flatten_depth(registry, depth)
    if mode == "subtree":
        if subtree_root is None:
            raise ValueError("flatten mode 'subtree' requires subtree_root=")
        return flatten_subtree(registry, subtree_root)
    if mode == "custom":
        if custom is None:
            raise ValueError("flatten mode 'custom' requires custom=")
        return flatten_custom(registry, custom)
    raise ValueError(f"unknown flatten mode: {mode}")
