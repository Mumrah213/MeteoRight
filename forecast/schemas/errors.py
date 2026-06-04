"""Error taxonomy for weather API interactions.

HTTP 200 does NOT mean valid data. This module defines structured error
types that classify failures by origin and semantics, not just HTTP codes.
"""


from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, Field


class ProviderErrorKind(Enum):
    """Classification of errors by origin and semantics.

    Errors are classified along two axes:
    - WHERE: transport, schema, validation, data, provider
    - WHAT: specific failure mode
    """

    # Transport layer failures
    CONNECTION_TIMEOUT = auto()
    CONNECTION_REFUSED = auto()
    DNS_RESOLUTION_FAILED = auto()
    HTTP_ERROR = auto()
    RESPONSE_TOO_LARGE = auto()

    # Schema / parsing failures
    INVALID_JSON = auto()
    MISSING_REQUIRED_FIELD = auto()
    TYPE_MISMATCH = auto()
    STRUCTURAL_INTEGRITY = auto()

    # Validation failures (data is parseable but semantically wrong)
    OUTSIDE_HORIZON = auto()
    IMPOSSIBLE_VALUE = auto()
    NON_MONOTONIC_TIMESTAMPS = auto()
    ARRAY_LENGTH_MISMATCH = auto()
    MISSING_UNITS = auto()

    # Provider-specific data issues (HTTP 200 but bad data)
    EMPTY_RESPONSE = auto()
    ALL_NULL_RESPONSE = auto()
    UNSUPPORTED_MODEL = auto()
    UNSUPPORTED_VARIABLE = auto()
    PARTIAL_NULLS = auto()

    # Configuration / capability issues
    FEATURE_NOT_SUPPORTED = auto()
    RATE_LIMITED = auto()
    AUTHENTICATION_FAILED = auto()

    # Unknown / unexpected
    UNKNOWN = auto()
    MALFORMED_RESPONSE = auto()


class NullClassification(Enum):
    """Classify why data arrays contain nulls.

    Distinguishing null sources is critical for forecast verification:
    an unsupported model returns all-null, but so does asking for variables
    that don't exist for a given model. We must detect both cases.
    """

    UNSUPPORTED_MODEL = auto()
    UNSUPPORTED_VARIABLE = auto()
    ALL_NULL_RESPONSE = auto()
    EMPTY_RESPONSE = auto()
    PARTIAL_NULLS = auto()
    OUTSIDE_HORIZON = auto()
    MALFORMED_RESPONSE = auto()
    UNKNOWN = auto()


class ProviderError(BaseModel):
    """Structured error from a provider interaction.

    This is a data model — use MeteoProviderError to raise it.
    """

    kind: ProviderErrorKind
    message: str
    provider: str = "unknown"
    endpoint: str = "unknown"
    parameters: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] | None = None
    null_classification: NullClassification | None = None
    http_status_code: int | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"[{self.kind.name}] {self.message}"]
        if self.provider != "unknown":
            parts.append(f"provider={self.provider}")
        if self.endpoint != "unknown":
            parts.append(f"endpoint={self.endpoint}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"ProviderError(kind={self.kind.name}, message={self.message!r})"


class MeteoProviderError(Exception):
    """Raiseable exception wrapping a ProviderError model.

    Use this to raise structured provider errors in try/except blocks.
    """

    error: ProviderError  # The underlying ProviderError model

    def __init__(self, error: ProviderError | None = None, message: str | None = None):
        if error is not None:
            self.error = error
            super().__init__(str(error))
        elif message:
            self.error = ProviderError(kind=ProviderErrorKind.UNKNOWN, message=message)
            super().__init__(message)
        else:
            super().__init__("Unknown error")

    def __str__(self) -> str:
        return str(self.error)

    def __repr__(self) -> str:
        return f"MeteoProviderError({self.error!r})"
