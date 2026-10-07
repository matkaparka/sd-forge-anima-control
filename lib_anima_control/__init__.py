"""Anima Control: control-LoRA support for Anima in Forge Neo (no model weights included)."""


class ControlError(RuntimeError):
    """A problem the user can fix (no control image, unknown LoRA...). Reported as one clear line, without a traceback."""
