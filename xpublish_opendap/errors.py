"""DAP protocol error types."""


class DapError(Exception):
    """Base class for DAP protocol errors."""

    def __init__(self, code: int, message: str):  # noqa: D107
        self.code = code
        self.message = message
        super().__init__(message)


class ConstraintSyntaxError(DapError):
    """Raised when a constraint expression is malformed."""

    def __init__(self, message: str):  # noqa: D107
        super().__init__(code=1000, message=message)


class ConstraintNotSupportedError(DapError):
    """Raised for valid but unsupported constraint features."""

    def __init__(self, message: str):  # noqa: D107
        super().__init__(code=1001, message=message)


class VariableNotFoundError(DapError):
    """Raised when a constraint references a non-existent variable."""

    def __init__(self, name: str):  # noqa: D107
        super().__init__(code=1002, message=f"Variable '{name}' not found")


class IndexOutOfRangeError(DapError):
    """Raised when a hyperslab index exceeds array bounds."""

    def __init__(self, message: str):  # noqa: D107
        super().__init__(code=1003, message=message)


class RequestTooLargeError(DapError):
    """Raised when estimated memory exceeds configured threshold."""

    def __init__(self, estimated_bytes: int, max_bytes: int):  # noqa: D107
        super().__init__(
            code=1004,
            message=(
                f"Estimated response size {estimated_bytes} bytes "
                f"exceeds maximum allowed {max_bytes} bytes"
            ),
        )
        self.estimated_bytes = estimated_bytes
        self.max_bytes = max_bytes
