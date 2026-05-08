"""Custom exception types for the Travel Planning API."""


class GeminiServiceError(Exception):
    """Raised when the Gemini API returns an error or is unavailable.

    Attributes:
        status_code: HTTP status code to surface to the caller (default 502).
    """

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class ValidationError(Exception):
    """Raised when request data fails business-rule validation beyond Pydantic.

    Attributes:
        field: Optional field name that triggered the error.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class RateLimitError(Exception):
    """Raised when a caller has exceeded the allowed request rate."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message)
