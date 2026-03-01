"""
exceptions.py
Exception hierarchy for the vision package.
All exceptions inherit from VisionError so callers can catch the base type.
"""


class VisionError(Exception):
    """Base exception for all vision package errors."""


class ElementNotFoundError(VisionError):
    """
    Raised when a UI element cannot be located after all pipeline stages
    and the timeout has elapsed.

    Attributes:
        template_path: Path of the template that was searched for.
        timeout:       Total seconds waited.
    """

    def __init__(self, message: str, template_path: str = "", timeout: float = 0.0):
        super().__init__(message)
        self.template_path = template_path
        self.timeout = timeout

    def __str__(self) -> str:
        base = super().__str__()
        if self.template_path:
            return f"{base} [template={self.template_path}, timeout={self.timeout}s]"
        return base


class ScreenCaptureError(VisionError):
    """Raised when mss fails to grab the screen."""


class TemplateLoadError(VisionError):
    """Raised when a template image cannot be read from disk."""


class PreprocessingError(VisionError):
    """Raised when an image preprocessing step produces an unusable result."""
