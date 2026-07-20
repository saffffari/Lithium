"""File hashing for catalog cache keys."""

import hashlib
import os


def compute_file_key(path: str) -> str:
    """Compute a cache key from file path, mtime, and size.

    Returns a 16-character hex string. Changes when the file is modified.
    """
    abs_path = os.path.abspath(path)
    stat = os.stat(abs_path)
    raw = f"{abs_path}|{stat.st_mtime_ns}|{stat.st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
