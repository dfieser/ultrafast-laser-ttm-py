"""Config-struct handling, mirroring the MATLAB toolbox's cfg semantics.

Solvers accept a plain dict (the analogue of the MATLAB cfg struct). A field
that is missing, ``None``, or empty falls back to the solver default, exactly
like the MATLAB ``getCfgField`` helper treats absent/empty struct fields.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def get_cfg_field(cfg: Mapping[str, Any] | None, name: str, default: Any) -> Any:
    """Return ``cfg[name]`` when present and non-empty, else ``default``."""
    if not isinstance(cfg, Mapping):
        return default
    if name not in cfg:
        return default
    value = cfg[name]
    if value is None:
        return default
    # MATLAB isempty(): '' and [] fall back to the default.
    if isinstance(value, (str, list, tuple, dict)) and len(value) == 0:
        return default
    return value


_TAG_UNSAFE = re.compile(r"[^\w.\-]+")


def safe_tag(tag: object) -> str:
    """Filename-safe form of a case tag.

    Tags are spliced into output filenames, so a path separator or other
    reserved character would otherwise crash the run at write time, after
    all the computing is done. Word characters, dots and hyphens pass
    through; everything else collapses to an underscore.
    """
    return _TAG_UNSAFE.sub("_", str(tag)).strip("_")
