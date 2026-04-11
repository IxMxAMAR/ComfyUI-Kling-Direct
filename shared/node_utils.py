"""Shared node utilities and mixins for ComfyUI nodes."""


class AlwaysExecuteMixin:
    """Mixin that forces re-execution on every queue (no stale cache)."""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
