"""Error hierarchy."""


class NightrunnerError(Exception):
    """Base class for every error raised by nightrunner."""


class FormatError(NightrunnerError):
    """The input bytes do not match the documented structure."""


class ValidationError(NightrunnerError):
    """A structurally readable file violates an engine contract."""


class BuildError(NightrunnerError):
    """A build request cannot be satisfied (refused rather than guessed)."""


class UnsupportedError(NightrunnerError):
    """A known-but-unimplemented case (compressed storage, secondary streams, ...)."""
