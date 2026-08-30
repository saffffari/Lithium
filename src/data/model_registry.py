"""Per-project model registry: track trained models, checkpoints, and lineage.

Each project can have multiple trained models. Models are stored in
``~/.lithium/library/models/{project_id}.json`` — one file per project,
lazy-loaded when the TRAIN tab opens.

Persistence follows the same atomic-write pattern as project.py.
"""

import os
import json
import time
import uuid
from dataclasses import dataclass, field, asdict


@dataclass
class TrainedModel:
    """A single trained model associated with a project."""

    model_id: str = ""              # uuid4 hex, unique within project
    name: str = ""                  # user-facing name, e.g. "PTv3 base 200ep"
    architecture: str = "PT-v3m1"   # extensible for future architectures
    status: str = "pending"         # pending | training | completed | failed | stopped
    created: float = 0.0            # time.time() at creation
    finished: float = 0.0           # time.time() at completion (0 = not finished)

    # Training config
    epochs: int = 200
    batch_size: int = 32
    device: str = "cuda"
    num_classes: int = 2

    # Results
    best_miou: float = 0.0
    best_checkpoint: str = ""       # absolute path to model_best.pth
    last_checkpoint: str = ""       # absolute path to last checkpoint

    # Paths
    work_dir: str = ""              # absolute path to the run directory

    # Config frozen at launch time
    config_snapshot: dict = field(default_factory=dict)

    # Class map frozen at launch: {str(label_id): name}, in training
    # class order. This is the authoritative record of what the model's
    # output channels mean — inference reads it from here first, so a
    # moved checkpoint or a changed project ontology can't silently
    # mis-map predictions (1.0 resolved classes through a fragile
    # cwd-relative classes.json fallback chain).
    class_map: dict = field(default_factory=dict)

    # Fine-tune lineage
    parent_model_id: str = ""       # empty for from-scratch, model_id for fine-tune

    error: str = ""                 # empty unless status == "failed"

    def __post_init__(self):
        if not self.model_id:
            self.model_id = uuid.uuid4().hex[:12]
        if self.created == 0.0:
            self.created = time.time()

    # -- status transitions ---------------------------------------------------

    def mark_training(self):
        self.status = "training"

    def mark_completed(self, best_miou: float = 0.0,
                       best_checkpoint: str = ""):
        self.status = "completed"
        self.finished = time.time()
        if best_miou > 0:
            self.best_miou = best_miou
        if best_checkpoint:
            self.best_checkpoint = best_checkpoint

    def mark_failed(self, error: str = ""):
        self.status = "failed"
        self.finished = time.time()
        self.error = error

    def mark_stopped(self):
        self.status = "stopped"
        self.finished = time.time()

    # -- queries --------------------------------------------------------------

    @property
    def has_checkpoint(self) -> bool:
        return bool(self.best_checkpoint) and os.path.isfile(self.best_checkpoint)

    @property
    def display_status(self) -> str:
        if self.status == "completed" and self.best_miou > 0:
            return f"completed  mIoU {self.best_miou:.3f}"
        return self.status

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TrainedModel":
        # Filter to only known fields for forward-compat
        known = {f.name for f in TrainedModel.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        m = TrainedModel(**filtered)
        return m


# ---------------------------------------------------------------------------
# Per-project model registry
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, payload) -> None:
    """Write JSON via tmp + replace to avoid corruption."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


class ProjectModelRegistry:
    """Per-project model tracking.

    Storage: ``{library_dir}/models/{project_id}.json``
    """

    def __init__(self, library_dir: str):
        self.models_dir = os.path.join(library_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self._cache: dict[str, list[TrainedModel]] = {}

    def drop_project(self, project_id: str) -> None:
        """Remove the persisted model list for ``project_id`` (SC-07).

        Called from ``LibraryCatalog.delete_project`` so a deleted
        project doesn't orphan a ``models/<id>.json`` file on disk —
        which would re-attach itself if the user later created a new
        project that hashed to the same id.
        """
        path = self._path(project_id)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"[model_registry] drop_project({project_id}) failed: {e}")
        self._cache.pop(project_id, None)

    def _path(self, project_id: str) -> str:
        # Project IDs use the form ``proj:<hash>`` (colon is part of the
        # namespace prefix). Windows refuses filenames containing
        # ``: \\ / * ? " < > |`` — without sanitising, ``os.replace`` on
        # the temp file raises WinError 87 mid-frame, which poisons
        # ImGui's frame state and locks the whole app in an assertion
        # loop. Map each forbidden char to ``_``; the inverse isn't
        # needed because the filename is only ever looked up by id.
        safe = project_id
        for ch in ':\\/*?"<>|':
            safe = safe.replace(ch, "_")
        return os.path.join(self.models_dir, f"{safe}.json")

    def load(self, project_id: str) -> list[TrainedModel]:
        """Load model list for a project. Cached after first load.

        Self-heals entries whose ``best_checkpoint`` is missing or
        points at a non-existent file by probing
        ``<work_dir>/model/model_best.pth`` (and ``model_last.pth`` as
        a last resort). A bug in the runner's log parser was capturing
        the literal string "to:" instead of the path for older runs;
        rather than ask the user to wipe and re-train, we recover the
        right path here from the run's work_dir.
        """
        if project_id in self._cache:
            return self._cache[project_id]
        path = self._path(project_id)
        models: list[TrainedModel] = []
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                for entry in data:
                    models.append(TrainedModel.from_dict(entry))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # corrupt file — start fresh
        # Heal corrupted checkpoint paths in-place. No write-back here;
        # the cache holds the fixed version and the next save() flush
        # persists it.
        healed = False
        for m in models:
            wd = getattr(m, 'work_dir', '') or ''
            cur = getattr(m, 'best_checkpoint', '') or ''
            if cur and os.path.isfile(cur):
                continue  # already valid
            if not wd or not os.path.isdir(wd):
                continue  # no work_dir to probe
            best_path = os.path.join(wd, "model", "model_best.pth")
            last_path = os.path.join(wd, "model", "model_last.pth")
            if os.path.isfile(best_path):
                m.best_checkpoint = best_path
                if not (m.last_checkpoint and os.path.isfile(m.last_checkpoint)):
                    m.last_checkpoint = (last_path
                                          if os.path.isfile(last_path) else best_path)
                healed = True
            elif os.path.isfile(last_path):
                m.best_checkpoint = last_path
                m.last_checkpoint = last_path
                healed = True
        self._cache[project_id] = models
        if healed:
            # Persist the healed paths so the next session doesn't pay
            # the probe cost.
            self.save(project_id)
        return models

    def save(self, project_id: str):
        """Persist the cached model list for a project."""
        models = self._cache.get(project_id, [])
        payload = [m.to_dict() for m in models]
        _atomic_write_json(self._path(project_id), payload)

    def add_model(self, project_id: str, model: TrainedModel):
        """Add a model and persist immediately."""
        models = self.load(project_id)
        models.append(model)
        self.save(project_id)

    def update_model(self, project_id: str, model_id: str, **updates):
        """Update fields on a model by id and persist."""
        models = self.load(project_id)
        for m in models:
            if m.model_id == model_id:
                for k, v in updates.items():
                    if hasattr(m, k):
                        setattr(m, k, v)
                break
        self.save(project_id)

    def get_model(self, project_id: str, model_id: str):
        """Return a specific model or None."""
        for m in self.load(project_id):
            if m.model_id == model_id:
                return m
        return None

    def all_models(self, project_ids) -> list[tuple[str, TrainedModel]]:
        """``(project_id, model)`` for every model across ``project_ids``,
        newest-first within each project, de-duplicated by ``model_id``
        (duplicated projects share model records by reference)."""
        out: list[tuple[str, TrainedModel]] = []
        seen: set[str] = set()
        for pid in project_ids:
            for m in self.list_models(pid):
                if m.model_id in seen:
                    continue
                seen.add(m.model_id)
                out.append((pid, m))
        return out

    def list_models(self, project_id: str) -> list[TrainedModel]:
        """Return all models for a project (most recent first)."""
        models = self.load(project_id)
        return sorted(models, key=lambda m: m.created, reverse=True)

    def delete_model(self, project_id: str, model_id: str,
                     delete_artifacts: bool = False) -> int:
        """Remove a model entry; optionally delete its run directory.

        With ``delete_artifacts=True`` the model's ``work_dir``
        (checkpoints, logs, config) is removed from disk — UNLESS any
        other registry entry in ANY cached/persisted project still
        references the same ``work_dir`` (duplicated projects share
        checkpoints by reference). Returns the number of bytes freed
        (0 when artifacts were kept).
        """
        models = self.load(project_id)
        target = next((m for m in models if m.model_id == model_id), None)
        self._cache[project_id] = [
            m for m in models if m.model_id != model_id
        ]
        self.save(project_id)
        if not (delete_artifacts and target is not None and target.work_dir):
            return 0
        if self._work_dir_referenced(target.work_dir):
            print(f"[model_registry] keeping {target.work_dir} — "
                  f"referenced by another model entry")
            return 0
        freed = self.artifact_size(target.work_dir)
        try:
            import shutil
            shutil.rmtree(target.work_dir)
        except OSError as e:
            print(f"[model_registry] artifact delete failed: {e}")
            return 0
        return freed

    def _work_dir_referenced(self, work_dir: str) -> bool:
        """True if any model entry across ALL projects references work_dir."""
        try:
            project_ids = [os.path.splitext(f)[0] for f in
                           os.listdir(self.models_dir) if f.endswith(".json")]
        except OSError:
            project_ids = list(self._cache.keys())
        for fname in project_ids:
            path = os.path.join(self.models_dir, f"{fname}.json")
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for entry in data:
                if entry.get("work_dir") == work_dir:
                    return True
        return False

    @staticmethod
    def artifact_size(work_dir: str) -> int:
        """Total bytes under a model's run directory (0 if missing)."""
        total = 0
        try:
            for root, _dirs, files in os.walk(work_dir):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def duplicate_from(self, source_project_id: str,
                       dest_project_id: str) -> int:
        """Copy a source project's model registry entries onto a fresh
        destination project, sharing checkpoint paths.

        Each duplicate gets a brand-new ``model_id`` so the registries
        stay independent for future edits (rename, delete, fine-tune),
        but ``best_checkpoint``, ``last_checkpoint``, and ``work_dir``
        point at the same files on disk. Inference in the duplicate
        loads the same .pth the source does — zero file copies, ~10 KB
        of registry JSON.

        Caveat: if the source project's training_runs directory is
        ever deleted, the duplicate's checkpoint references become
        dangling. Callers can detect this via the standard
        ``load()`` self-heal path (it falls back to ``last_checkpoint``
        or scans ``work_dir``).

        Returns the number of models copied.
        """
        import copy
        src_models = self.load(source_project_id)
        if not src_models:
            return 0
        dest_models = self.load(dest_project_id)
        # Reset uuid generation per duplicate so the dest project
        # gets brand-new ids even if it already had its own models.
        for src in src_models:
            # Round-trip through dict for a clean deep copy of all
            # nested fields (config_snapshot is a nested dict).
            clone_dict = copy.deepcopy(src.to_dict())
            clone_dict["model_id"] = ""  # force re-uuid in __post_init__
            clone = TrainedModel.from_dict(clone_dict)
            clone.model_id = uuid.uuid4().hex[:12]
            # Preserve the source's parent_model_id (lineage stays
            # accurate). The dest project's registry just has more
            # entries pointing at the same shared checkpoint files.
            dest_models.append(clone)
        self._cache[dest_project_id] = dest_models
        self.save(dest_project_id)
        return len(src_models)

    def invalidate(self, project_id: str):
        """Drop cache for a project so next load re-reads from disk."""
        self._cache.pop(project_id, None)

    def scan_orphan_runs(self, project_id: str,
                         data_dir: str) -> list[dict]:
        """Discover training_runs/ directories that have no registry entry.

        Returns dicts with {work_dir, name, has_best_ckpt} so the UI can
        offer to import them.
        """
        runs_dir = os.path.join(data_dir, "training_runs")
        if not os.path.isdir(runs_dir):
            return []
        known_dirs = {m.work_dir for m in self.load(project_id)}
        orphans = []
        for name in sorted(os.listdir(runs_dir)):
            run_path = os.path.join(runs_dir, name)
            if not os.path.isdir(run_path):
                continue
            if run_path in known_dirs:
                continue
            # Check for checkpoint files (Pointcept convention)
            best_ckpt = ""
            for candidate in ("model/model_best.pth", "model_best.pth"):
                p = os.path.join(run_path, candidate)
                if os.path.isfile(p):
                    best_ckpt = p
                    break
            orphans.append({
                "work_dir": run_path,
                "name": name,
                "has_best_ckpt": bool(best_ckpt),
                "best_ckpt_path": best_ckpt,
            })
        return orphans
