from __future__ import annotations

import time
import os


def add_one(x: int) -> int:
    return x + 1


def sleep_api(x: int) -> int:
    time.sleep(5)
    return x


def int_only(x: int) -> int:
    if not isinstance(x, int) or isinstance(x, bool):
        raise ValueError("x must be int")
    return x


def abort_on_two(x: int) -> int:
    if x == 2:
        os.abort()
    return x


def padding_api(padding: str) -> str:
    normalized = str(padding).upper()
    if normalized not in {"VALID", "SAME", "SAME_LOWER"}:
        raise ValueError(f"Unknown padding type: {padding}")
    return normalized


def positional_only_api(x, /, y=1):
    return x + y


def keyword_only_api(*, x):
    return x + 1


def axis_api(x, axes):
    import numpy as np

    if len(set(axes)) != len(tuple(axes)):
        raise ValueError("duplicate axes")
    if any(axis < 0 or axis >= np.asarray(x).ndim for axis in axes):
        raise ValueError("axis out of range")
    return np.sum(x, axis=axes)


def index_api(x, indices):
    import numpy as np

    if not np.issubdtype(np.asarray(indices).dtype, np.integer):
        raise ValueError("indices must be integer")
    return np.take(x, indices)


def uint8_api(a):
    import numpy as np

    arr = np.asarray(a)
    if arr.dtype != np.uint8:
        raise ValueError("expected uint8")
    return arr
