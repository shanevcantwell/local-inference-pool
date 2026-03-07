"""Custom exceptions for local-inference-pool."""


class NoModelsAvailableError(RuntimeError):
    """All server manifests are empty — auth, connectivity, or no models loaded."""


class ModelNotAvailableError(ValueError):
    """Requested model not found on any server with a populated manifest."""


class DispatcherTimeoutError(TimeoutError):
    """Timed out waiting for a server slot."""
