"""CPU-only contracts shared by single-cloud inference and its GUI caller."""
from pathlib import Path
import numpy as np


class FragmentLogitAccumulator:
    """Mean raw logits for every original point; incomplete results are errors."""
    def __init__(self, point_count):
        if point_count <= 0:
            raise ValueError("Inference requires a nonempty cloud")
        self.coverage = np.zeros(point_count, dtype=np.uint32)
        self.logits_sum = None
        self.fragment_count = 0

    def add(self, index, logits):
        index = np.asarray(index)
        logits = np.asarray(logits, dtype=np.float32)
        if (index.ndim != 1 or not np.issubdtype(index.dtype, np.integer)
                or len(index) == 0):
            raise ValueError("Fragment index must be a nonempty integer vector")
        if (index.min() < 0 or index.max() >= len(self.coverage)
                or len(np.unique(index)) != len(index)):
            raise ValueError("Fragment index contains duplicate or out-of-range points")
        if (logits.ndim != 2 or logits.shape[0] != len(index)
                or logits.shape[1] == 0 or not np.isfinite(logits).all()):
            raise ValueError("Fragment logits must have finite aligned class rows")
        if self.logits_sum is None:
            self.logits_sum = np.zeros((len(self.coverage), logits.shape[1]), dtype=np.float32)
        if logits.shape[1] != self.logits_sum.shape[1]:
            raise ValueError("Fragment class count changed")
        self.logits_sum[index] += logits
        self.coverage[index] += 1
        self.fragment_count += 1

    def finish(self):
        missing = int(np.count_nonzero(self.coverage == 0))
        if missing:
            raise ValueError(f"Fragments omitted {missing} of {len(self.coverage)} original points")
        mean = self.logits_sum / self.coverage[:, None].astype(np.float32)
        if not np.isfinite(mean).all():
            raise ValueError("Aggregated logits are non-finite")
        return mean.argmax(axis=1).astype(np.int32)

    def summary(self):
        return dict(aggregation_policy="mean_raw_logits_all_fragments_v1",
                    point_count=len(self.coverage), fragment_count=self.fragment_count,
                    class_count=0 if self.logits_sum is None else self.logits_sum.shape[1],
                    coverage_min=int(self.coverage.min()), coverage_max=int(self.coverage.max()),
                    covered_point_count=int(np.count_nonzero(self.coverage)),
                    fragment_point_visits=int(self.coverage.sum()))


def registered_config_path(model):
    """Honor a model's explicit source config; never guess if it went missing."""
    snapshot = getattr(model, "config_snapshot", None) or {}
    value = snapshot.get("source_config")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Registered model configuration is missing: {path}")
    return str(path.resolve())
