"""Generate Pointcept PT-v3m1 semseg configs from our exported datasets.

Pointcept uses OpenMMLab-style Python configs, not YAML — they're
actually importable modules that build a dict of nested dicts. This
module emits those configs as plain Python source strings tuned for
our setup:

  - small point clouds (16k) → whole-cloud training, no sphere crop
  - our class id convention (IGNORE_INDEX=255, contiguous 0..K-1)
  - our exported directory layout (``train/val/test/scene_*/``)
  - our custom dataset class registered in
    ``src/training/pointcept_ext/three_photon_dataset.py``

The config is written into the run's work_dir as ``config.py`` and
also contains a header block recording every input knob so a user
reopening the run file six months later knows exactly what produced it.

Intentionally not a template engine — generated configs are short
enough that a few Python f-strings are clearer than Jinja.
"""

import os
import json
import textwrap
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# User-facing knobs
# ---------------------------------------------------------------------------

@dataclass
class PTv3TrainParams:
    """Tuning knobs exposed to the UI. Defaults biased for 16k point
    single-vertebra clouds on a 4090 — override as the dataset grows.
    """
    # --- dataset ---
    data_root: str = ""           # absolute path to export dir with train/val/test
    num_classes: int = 0          # read from classes.json if zero
    ignore_index: int = 255       # matches our export contract
    has_color: bool = True        # export always writes color.npy
    has_normal: bool = True       # export writes normal.npy when requested
    grid_size: float = 0.5        # same units as coord.npy. VerSe / CT-derived
                                   # vertebrae are stored in millimetres so 0.5
                                   # = 0.5 mm — fine enough to resolve endplate
                                   # detail, coarse enough that grid_coord.max()
                                   # stays well under Pointcept's 2^16 cap
                                   # (asserted in models/utils/structure.py:82).
                                   # The previous default of 0.001 was based on
                                   # the wrong unit assumption (meters), and a
                                   # vertebra spans ~80 mm → 80 / 0.001 = 80,000
                                   # cells → fails depth <= 16.

    # --- model (PT-v3m1) ---
    # Field names verified against
    # training/pointcept/configs/scannet/semseg-pt-v3m1-0-base.py
    # on 2026-04-11. If Pointcept's backbone constructor changes field
    # names again, update both the dataclass and the emit template.
    order: tuple = ("z", "z-trans", "hilbert", "hilbert-trans")
    stride: tuple = (2, 2, 2, 2)
    enc_depths: tuple = (2, 2, 2, 6, 2)
    enc_channels: tuple = (32, 64, 128, 256, 512)
    enc_num_head: tuple = (2, 4, 8, 16, 32)
    enc_patch_size: tuple = (1024, 1024, 1024, 1024, 1024)
    dec_depths: tuple = (2, 2, 2, 2)
    dec_channels: tuple = (64, 64, 128, 256)
    dec_num_head: tuple = (4, 4, 8, 16)
    dec_patch_size: tuple = (1024, 1024, 1024, 1024)
    mlp_ratio: int = 4
    qkv_bias: bool = True
    drop_path: float = 0.3
    enable_flash: bool = False    # flash-attn is optional; off by default
    enc_mode: bool = False        # renamed from cls_mode in Pointcept main
    # PDNorm fields drive multi-dataset prompt conditioning. Off by
    # default for single-dataset training but the PT-v3m1 constructor
    # requires them to be present even when disabled.
    pdnorm_bn: bool = False
    pdnorm_ln: bool = False
    pdnorm_decouple: bool = True
    pdnorm_adaptive: bool = False
    pdnorm_affine: bool = True
    pdnorm_conditions: tuple = ("Vertebrae",)

    # --- optimizer + schedule ---
    batch_size: int = 2           # PT-v3m1 (46M params) on point cloud
                                   # scenes is memory-heavy: each sample
                                   # voxelises into thousands of attention
                                   # tokens, and the 5-level enc/dec stack
                                   # holds intermediate activations across
                                   # all of them. Batch 32 hits CUDA OOM
                                   # on a 24GB 4090 right at the first
                                   # training step. The Apr 16 working
                                   # run used 2; bump up only after
                                   # checking nvidia-smi headroom.
    num_worker: int = 4           # The runner monkey-patches Pointcept's
                                   # worker_init_fn to re-import our
                                   # ThreePhotonDataset class, so DataLoader
                                   # workers (spawn-mode on Windows) end up
                                   # with the class registered just like
                                   # the main process. See ptv3_runner.py.
    epochs: int = 200
    base_lr: float = 0.006
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    enable_amp: bool = True
    mix_prob: float = 0.0         # PointMix on single-object clouds is noisy
    empty_cache: bool = False

    # --- augmentation ---
    random_rotate_so3: bool = True  # vertebrae have no canonical up axis
    random_scale: tuple = (0.9, 1.1)
    random_jitter_sigma: float = 0.005
    random_flip: bool = True
    color_jitter: bool = False     # default off — color is often synthetic

    # --- fine-tuning ---
    weight: str = ""                # path to .pth checkpoint for fine-tuning
                                    # empty = train from scratch

    # --- class weighting (severe imbalance mitigation) ---
    # When the dataset includes "Unlabeled" as class 0 (the new default),
    # ~80% of every cloud is class 0 and only ~10% each is sup / inf
    # endplate. Plain CrossEntropyLoss collapses to predicting Unlabeled
    # everywhere. ``class_weights`` is passed verbatim to
    # ``CrossEntropyLoss(weight=...)`` so the loss penalises foreground
    # mistakes more heavily. None = uniform weighting (same as before).
    # Length must match num_classes; the launch path computes a sensible
    # default of (0.1, 1.0, 1.0, ...) when num_classes >= 2.
    class_weights: tuple = ()

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def load_num_classes(dataset_dir: str) -> int:
    """Read ``num_classes`` from ``classes.json`` emitted by export."""
    path = os.path.join(dataset_dir, "classes.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"classes.json not found at {path} — did you run export_dataset?")
    with open(path) as f:
        data = json.load(f)
    if "num_classes" in data:
        return int(data["num_classes"])
    if "class_names" in data:
        return len(data["class_names"])
    raise ValueError(f"{path} missing num_classes / class_names")


def generate_ptv3_config(params: PTv3TrainParams,
                         pointcept_ext_dir: str,
                         output_path: str) -> str:
    """Write a Pointcept PT-v3m1 semseg config to ``output_path``.

    ``pointcept_ext_dir`` is the absolute path to our custom dataset
    module directory. The generated config will insert it at the top
    of ``sys.path`` before importing ``three_photon_dataset`` so
    Pointcept's DATASETS registry picks up our class.

    Returns the generated config source for callers that want to log
    or display it.
    """
    if params.num_classes <= 0:
        params = PTv3TrainParams(
            **{**params.to_dict(),
               "num_classes": load_num_classes(params.data_root)})

    # Count the non-geometry input channels. Coord is always fed;
    # color and normal flow in when present. Keep it explicit so the
    # backbone's ``in_channels`` matches what the dataset produces.
    in_channels = 3  # xyz
    feat_keys = ["coord"]
    if params.has_color:
        in_channels += 3
        feat_keys.append("color")
    if params.has_normal:
        in_channels += 3
        feat_keys.append("normal")

    # Normalize paths for cross-platform config files. Forward slashes
    # work on Windows too and don't need Python string escaping.
    data_root = params.data_root.replace("\\", "/")
    pointcept_ext_dir = pointcept_ext_dir.replace("\\", "/")

    transforms_train = _build_transform_pipeline(params, training=True,
                                                  feat_keys=feat_keys)
    transforms_val = _build_transform_pipeline(params, training=False,
                                                feat_keys=feat_keys)

    source = f'''"""Generated 3Photon PTv3 training config.

DO NOT EDIT BY HAND - regenerate with src/training/config_gen.py.
Parameters at generation time:
{_pretty_params(params)}
"""

# Pointcept's Config.fromfile captures every top-level name in the
# module dict and tries to yapf-format it back on dump. That means
# we cannot have ``import sys`` or ``import three_photon_dataset``
# as top-level statements - yapf can't serialize a module object.
# Instead we use bare __import__ expressions which fire the side
# effect (sys.path mutation, dataset registry registration) without
# binding any name at module scope.
__import__("sys").path.insert(0, r"{pointcept_ext_dir}")
__import__("three_photon_dataset")

# ---------------------------------------------------------------------------
# Run (matches pointcept/configs/_base_/default_runtime.py fields)
# ---------------------------------------------------------------------------
weight = {repr(params.weight.replace(chr(92), '/')) if params.weight else None}
resume = False
evaluate = True
test_only = False
seed = None
save_path = "exp/default"
batch_size = {params.batch_size}
num_worker = {params.num_worker}
batch_size_val = None
batch_size_test = None
gradient_accumulation_steps = 1
clip_grad = None
sync_bn = False
enable_amp = {params.enable_amp}
amp_dtype = "float16"
empty_cache = {params.empty_cache}
# Per-epoch CUDA cache reset. PT-v3m1 with attention on variable-size
# point clouds fragments the caching allocator badly after ~30-40 epochs
# at 97% VRAM utilisation (see test3_1778436770 — per-step time went
# 100x slower mid-run while GPU power draw collapsed to ~100W). Clearing
# at epoch boundaries defrags without the per-step throughput hit that
# empty_cache=True would incur.
empty_cache_per_epoch = True
find_unused_parameters = False
enable_wandb = False
wandb_project = "3photon"
wandb_key = None
mix_prob = {params.mix_prob}
epoch = {params.epochs}
eval_epoch = {params.epochs}

# ---------------------------------------------------------------------------
# Model - PT-v3m1 semseg
# ---------------------------------------------------------------------------
model = dict(
    type="DefaultSegmentorV2",
    num_classes={params.num_classes},
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels={in_channels},
        order={tuple(params.order)!r},
        stride={tuple(params.stride)!r},
        enc_depths={tuple(params.enc_depths)!r},
        enc_channels={tuple(params.enc_channels)!r},
        enc_num_head={tuple(params.enc_num_head)!r},
        enc_patch_size={tuple(params.enc_patch_size)!r},
        dec_depths={tuple(params.dec_depths)!r},
        dec_channels={tuple(params.dec_channels)!r},
        dec_num_head={tuple(params.dec_num_head)!r},
        dec_patch_size={tuple(params.dec_patch_size)!r},
        mlp_ratio={params.mlp_ratio},
        qkv_bias={params.qkv_bias},
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path={params.drop_path},
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash={params.enable_flash},
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode={params.enc_mode},
        pdnorm_bn={params.pdnorm_bn},
        pdnorm_ln={params.pdnorm_ln},
        pdnorm_decouple={params.pdnorm_decouple},
        pdnorm_adaptive={params.pdnorm_adaptive},
        pdnorm_affine={params.pdnorm_affine},
        pdnorm_conditions={tuple(params.pdnorm_conditions)!r},
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             ignore_index={params.ignore_index},{_class_weight_kwarg(params)}),
        dict(type="LovaszLoss",
             mode="multiclass",
             loss_weight=1.0,
             ignore_index={params.ignore_index}),
    ],
)

# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr={params.base_lr}, weight_decay={params.weight_decay})
scheduler = dict(
    type="OneCycleLR",
    max_lr=[{params.base_lr}, {params.base_lr} * 0.1],
    pct_start={_warmup_pct(params)},
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr={params.base_lr} * 0.1)]

# ---------------------------------------------------------------------------
# Dataset - 3Photon export format
# ---------------------------------------------------------------------------
dataset_type = "ThreePhotonDataset"
data_root = r"{data_root}"

data = dict(
    num_classes={params.num_classes},
    ignore_index={params.ignore_index},
    names={_class_names_repr(params)},
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform={transforms_train},
        test_mode=False,
        ignore_index={params.ignore_index},
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform={transforms_val},
        test_mode=False,
        ignore_index={params.ignore_index},
    ),
)

# Hooks - full list matching default_runtime.py so the config is
# self-contained and doesn't need to inherit from a _base_ path that
# depends on where the config file lives on disk.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # PreciseEvaluator deliberately omitted. Its after_train hook builds
    # a SemSegTester that reads cfg.data.test, which this config does
    # not define — that triggered AttributeError at the very end of
    # run_1778447686 even though training had completed cleanly. The
    # tester block below stays so the schema is well-formed, but with
    # no hook to invoke it the run exits with returncode 0 after the
    # final checkpoint save. 3Photon has its own inference path
    # (tools/infer_3photon.py via _launch_inference) so the post-train
    # test pass was never load-bearing for us.
]

# Trainer + Tester
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)
'''
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # UTF-8 always — the generated config has unicode arrows in the
    # header and Windows cp1252 can't encode them. Pointcept doesn't
    # care about the encoding as long as it's readable.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(source)
    return source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_names_repr(params: PTv3TrainParams) -> str:
    """Return a Python repr of class names for the config file.

    Reads from classes.json if present; falls back to numbered names.
    """
    cj = os.path.join(params.data_root, "classes.json")
    if os.path.isfile(cj):
        with open(cj) as f:
            data = json.load(f)
        if "class_names" in data:
            return repr(data["class_names"])
    # Fallback: numbered class names
    return repr([f"class_{i}" for i in range(params.num_classes)])


def _warmup_pct(params: PTv3TrainParams) -> float:
    """OneCycleLR ``pct_start`` from the user's ``warmup_epochs`` setting.

    OneCycleLR drives the LR by total iteration fraction, but the user
    thinks in epochs. Since iterations scale linearly with epochs at a
    fixed dataset / batch size, ``warmup_epochs / total_epochs`` is the
    correct fraction. Clamped to (0, 0.5] so the warmup never exceeds
    half the run (which would make OneCycleLR's anneal phase trivial)
    and never collapses to 0 (which OneCycleLR rejects).
    """
    total = max(1, int(params.epochs))
    warm = max(0, int(getattr(params, "warmup_epochs", 10)))
    pct = warm / total if total > 0 else 0.05
    return max(0.001, min(0.5, pct))


def _class_weight_kwarg(params: PTv3TrainParams) -> str:
    """Render the ``weight=[...]`` kwarg for CrossEntropyLoss when the
    config provides class_weights, otherwise an empty string.

    The trailing comma in the criteria template means this either lands
    as a real kwarg or as a no-op separator — a clean way to make the
    weight optional without splitting the template.
    """
    cw = getattr(params, "class_weights", None)
    if not cw:
        return ""
    return f"\n             weight={list(cw)!r}"


def _build_transform_pipeline(params: PTv3TrainParams, *,
                              training: bool,
                              feat_keys: list[str]) -> str:
    """Return a Python source fragment describing the transform list.

    We build this as a string (rather than a Python list) so it
    embeds cleanly into the generated config file. Each entry is
    a Pointcept ``dict(type=..., ...)`` call on its own line.
    """
    t: list[str] = []
    t.append('dict(type="CenterShift", apply_z=True)')

    if training:
        if params.random_rotate_so3:
            # Three successive axis-aligned rotations give full SO(3)
            # coverage without needing a custom transform class.
            t.append('dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5)')
            t.append('dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.5)')
            t.append('dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.5)')
        else:
            t.append('dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5)')

        lo, hi = params.random_scale
        t.append(f'dict(type="RandomScale", scale=[{lo}, {hi}])')
        if params.random_flip:
            t.append('dict(type="RandomFlip", p=0.5)')
        if params.random_jitter_sigma > 0:
            t.append(f'dict(type="RandomJitter", sigma={params.random_jitter_sigma}, clip=0.02)')
        if params.color_jitter and params.has_color:
            t.append('dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None)')
            t.append('dict(type="ChromaticTranslation", p=0.95, ratio=0.05)')
            t.append('dict(type="ChromaticJitter", p=0.95, std=0.05)')

    if params.has_color:
        t.append('dict(type="NormalizeColor")')

    # Grid-sample into voxels so PTv3's serialization has a sparse grid
    # to work on. Our vertebra clouds are small enough that this is
    # usually a no-op — most points survive.
    # IMPORTANT: val uses mode="train" too — mode="test" returns a list
    # of fragments which breaks the dataloader collation. Only use
    # mode="test" in the actual test config (TTA inference).
    gs_mode = "train"
    t.append(
        f'dict(type="GridSample", grid_size={params.grid_size}, '
        f'hash_type="fnv", mode="{gs_mode}", return_grid_coord=True)'
    )
    t.append(f'dict(type="ToTensor")')
    t.append(
        'dict(type="Collect", keys=("coord", "grid_coord", "segment"), '
        f'feat_keys={feat_keys!r})'
    )

    joined = ",\n            ".join(t)
    return "[\n            " + joined + ",\n        ]"


def _pretty_params(params: PTv3TrainParams) -> str:
    lines = []
    for k, v in params.to_dict().items():
        lines.append(f"#   {k} = {v!r}")
    return "\n".join(lines)
