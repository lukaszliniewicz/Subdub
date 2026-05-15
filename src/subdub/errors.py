"""Application-specific exception hierarchy."""


class SubdubError(Exception):
    """Base class for all application-specific errors."""


class ConfigurationError(SubdubError):
    """Raised when user configuration or runtime setup is invalid."""


class ExternalToolError(SubdubError):
    """Raised when an external executable or API call fails unexpectedly."""
