"""Conservative viewer cache limits; annotation always keeps full point resolution.

These cap retained cache payloads, not total process/driver memory. One active
cloud and the visible gallery working set are kept even if they exceed a limit.
"""
from math import isqrt
from dataclasses import dataclass

MIB = 1024 * 1024
FULL_CACHE_BYTES = 128 * MIB
FULL_CACHE_COUNT = 3
PREVIEW_CACHE_BYTES = 48 * MIB
PREVIEW_CACHE_COUNT = 96
THUMBNAIL_CACHE_BYTES = 64 * MIB
THUMBNAIL_MAX_SIDE = 256
GALLERY_UPLOADS_PER_FRAME = 2
GALLERY_DIRTY_PER_FRAME = 4


@dataclass(frozen=True)
class ResourceBudget:
    name: str = 'Lean'
    full_bytes: int = FULL_CACHE_BYTES
    full_count: int = FULL_CACHE_COUNT
    preview_bytes: int = PREVIEW_CACHE_BYTES
    preview_count: int = PREVIEW_CACHE_COUNT
    thumbnail_bytes: int = THUMBNAIL_CACHE_BYTES
    thumbnail_side: int = THUMBNAIL_MAX_SIDE
    uploads_per_frame: int = GALLERY_UPLOADS_PER_FRAME
    dirty_per_frame: int = GALLERY_DIRTY_PER_FRAME


LEAN = ResourceBudget()
BALANCED = ResourceBudget('Balanced', 512*MIB, 16, 96*MIB, 192, 128*MIB, 384, 4, 12)
WORKSTATION = ResourceBudget('Workstation', 2048*MIB, 512, 256*MIB, 768, 384*MIB, 1024, 8, 24)


def choose_budget(preference='auto', ram_bytes=0, vram_bytes=0, renderer=''):
    """Unknown/integrated GPUs stay conservative; explicit overrides always win."""
    if preference == 'lean':
        return LEAN
    if preference == 'workstation':
        return WORKSTATION
    software = any(name in renderer.lower() for name in ('llvmpipe', 'softpipe', 'software'))
    if software:
        return LEAN
    if ram_bytes >= 32*1024*MIB and vram_bytes >= 8*1024*MIB:
        return WORKSTATION
    if ram_bytes >= 16*1024*MIB and vram_bytes >= 4*1024*MIB:
        return BALANCED
    return LEAN


def physical_memory_bytes():
    """Total physical RAM without adding a runtime dependency."""
    import os, sys
    try:
        if sys.platform == 'win32':
            import ctypes as ct
            class MemoryStatus(ct.Structure):
                _fields_ = [('length', ct.c_ulong), ('load', ct.c_ulong)] + [
                    (name, ct.c_ulonglong) for name in ('total', 'available', 'page_total',
                        'page_available', 'virtual_total', 'virtual_available', 'extended')]
            status = MemoryStatus(); status.length = ct.sizeof(status)
            if ct.windll.kernel32.GlobalMemoryStatusEx(ct.byref(status)):
                return status.total
        return int(os.sysconf('SC_PHYS_PAGES')) * int(os.sysconf('SC_PAGE_SIZE'))
    except (AttributeError, ValueError, OSError):
        return 0


def dedicated_memory_bytes(ctx):
    """NVIDIA's cross-platform GL memory extension, or unknown (0).

    Spec: registry.khronos.org/OpenGL/extensions/NVX/NVX_gpu_memory_info.txt
    Other drivers use conservative auto defaults or the explicit override.
    """
    if 'GL_NVX_gpu_memory_info' not in ctx.extensions:
        return 0
    try:
        from OpenGL.GL import glGetIntegerv
        return max(0, int(glGetIntegerv(0x9047))) * 1024
    except Exception:
        return 0


def thumbnail_side(cell_side: int, visible_count: int,
                   budget: int = THUMBNAIL_CACHE_BYTES, max_side: int = THUMBNAIL_MAX_SIDE) -> int:
    """Fit visible RGBA16F thumbnails in the budget, independently of UI size."""
    fitting_side = isqrt(max(1, budget // (8 * max(1, visible_count))))
    return max(1, min(int(cell_side), max_side, fitting_side))


def cloud_payload_bytes(gpu) -> int:
    """CPU arrays + split GPU buffers, counting numpy aliases only once."""
    cloud = gpu.cloud_data
    arrays = [cloud.positions, cloud.colors, cloud.labels,
              getattr(gpu, '_sel_staging', None), getattr(gpu, '_lbl_staging', None)]
    arrays.extend(getattr(cloud, 'scalars', {}).values())
    seen = set()
    total = int(gpu.point_count) * 32  # positions/RGB/labels/selection on GPU
    for array in arrays:
        if array is None:
            continue
        base = array
        while getattr(base, 'base', None) is not None:
            base = base.base
        identity = id(base)
        if identity not in seen:
            seen.add(identity)
            total += int(getattr(base, 'nbytes', array.nbytes))
    return total


def eviction_candidates(entries, attribute, protected_ids, max_count, max_bytes):
    """Oldest retained resources first, excluding the active/visible working set."""
    resident = [e for e in entries if getattr(e, attribute, None) is not None]
    total = sum(cloud_payload_bytes(getattr(e, attribute)) for e in resident)
    count = len(resident)
    for entry in sorted(resident, key=lambda e: getattr(e, '_resident_touch', 0.0)):
        if count <= max_count and total <= max_bytes:
            break
        if id(entry) in protected_ids:
            continue
        payload = cloud_payload_bytes(getattr(entry, attribute))
        count -= 1
        total -= payload
        yield entry
