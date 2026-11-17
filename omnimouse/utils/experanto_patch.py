"""Expose extra per-trial metadata keys to experanto's `valid_condition` filter. OPT-IN.

`ChunkDataset.get_condition_mask_from_meta_conditions` (experanto/datasets.py:583) is a
generic sum-of-products filter over `self.meta_conditions`, and its own docstring advertises
`{'tier': 'test', 'stim_type': 'stimulus.Frame'}`. But `_read_trials`
(experanto/datasets.py:408-416) only ever populates three keys::

    for k in ["modality", "valid_trial", "tier"]:

so any `valid_condition` mentioning `stim_type` raises `KeyError`. That blocks the
out-of-distribution stimulus experiment, where four of the five held-out classes
(GaborGridSequence, RdkGridSequence, DotSequence, Monet2) are all `modality: video` and can
only be told apart by `stim_type`.

experanto is pinned non-editable from git (pyproject.toml, rev `foundation_model`), so editing
`.venv` would be reverted by the next `uv sync`. This module monkeypatches the method instead
and lives in the omnimouse repo.

**This is opt-in and off by default**: `omnimouse.tasks.task_utils.setup_common` calls `apply()`
only when `data.expose_extra_meta_conditions: true`, so no existing run changes behaviour. If a
config filters on an unexposed key without setting the flag, `setup_common` raises a clear error
rather than letting the run die later with a bare `KeyError`. `apply()` is idempotent.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# What experanto populates on its own (experanto/datasets.py:415).
EXPERANTO_BUILTIN_KEYS = ("modality", "valid_trial", "tier")

# Keys this patch adds, all read from each session's screen/combined_meta.json:
#   stim_type      e.g. stimulus.Clip / stimulus.Frame / stimulus.GaborGridSequence
#                  / stimulus.RdkGridSequence / stimulus.DotSequence / stimulus.Monet2 / stimulus.Trippy
#   image_class    e.g. imagenet_v2_gray  (only set on stimulus.Frame trials)
#   condition_hash one id per distinct stimulus condition, e.g. per movie clip
EXTRA_META_KEYS = ("stim_type", "image_class", "condition_hash")

_PATCH_FLAG = "_omnimouse_extra_meta_patched"


def apply() -> bool:
    """Patch `ChunkDataset._read_trials` to also expose EXTRA_META_KEYS.

    Returns True if the patch was applied by this call, False if it was already in place.
    """
    from experanto.datasets import ChunkDataset

    original = ChunkDataset._read_trials
    if getattr(original, _PATCH_FLAG, False):
        return False

    def _read_trials_with_extra_meta(self) -> None:
        original(self)
        for key in EXTRA_META_KEYS:
            # Mirror experanto's own convention: a missing value becomes "blank", which is what
            # the interleaved blank trials carry for `modality`.
            self.meta_conditions[key] = [
                t.get_meta(key) if t.get_meta(key) is not None else "blank"
                for t in self._trials
            ]

    setattr(_read_trials_with_extra_meta, _PATCH_FLAG, True)
    ChunkDataset._read_trials = _read_trials_with_extra_meta
    log.debug("experanto ChunkDataset._read_trials patched to expose %s", EXTRA_META_KEYS)
    return True


def _iter_valid_condition_keys(node: Any):
    """Yield every key used inside any `valid_condition` anywhere in a config tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "valid_condition":
                # A valid_condition is a list of dicts (sum of products), but tolerate a
                # bare dict, which some configs use.
                entries = v if isinstance(v, (list, tuple)) else [v]
                for entry in entries:
                    if isinstance(entry, dict):
                        yield from entry.keys()
            else:
                yield from _iter_valid_condition_keys(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _iter_valid_condition_keys(v)


def config_needs_extra_meta(cfg) -> bool:
    """True if any `valid_condition` in `cfg` filters on a key experanto does not expose."""
    try:
        from omegaconf import OmegaConf

        tree = OmegaConf.to_container(cfg, resolve=False)
    except Exception:  # pragma: no cover - fall back to the raw object
        tree = cfg
    return any(
        k not in EXPERANTO_BUILTIN_KEYS for k in _iter_valid_condition_keys(tree)
    )
