"""Validation result schemas.

Every validation pass produces a ValidationResult with typed issues.
Issues are classified as structural or semantic, making them easy to
filter and handle programmatically.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, Field


class IssueSeverity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


class StructuralIssue(BaseModel):
    """Structural validation issue — the data structure is wrong."""

    rule: str  # Which validation rule failed
    field: str  # Which field/variable triggered it
    message: str  # Human-readable description
    raw_value: Any = None
    severity: IssueSeverity = IssueSeverity.ERROR

    def __str__(self) -> str:
        return f"STRUCTURAL [{self.severity.name}]: {self.message}"


class SemanticIssue(BaseModel):
    """Semantic validation issue — the structure is fine but values are wrong."""

    rule: str  # Which semantic rule failed
    variable: str  # Which weather variable triggered it
    value: float | None  # The value that failed (or None if null)
    min_allowed: float | None = None
    max_allowed: float | None = None
    message: str  # Human-readable description
    severity: IssueSeverity = IssueSeverity.WARNING

    def __str__(self) -> str:
        return f"SEMANTIC [{self.severity.name}]: {self.message}"


class ValidationResult(BaseModel):
    """Result of validating a raw forecast response.

    A response with issues is NOT automatically invalid. The caller
    decides how to handle issues based on severity and type.
    """

    valid: bool
    issues: list[StructuralIssue | SemanticIssue] = Field(default_factory=list)
    null_classification: str | None = None  # From NullClassification enum string
    provider: str = "unknown"
    endpoint: str = "unknown"
    rules_checked: int = 0
    rules_passed: int = 0

    @property
    def error_count(self) -> int:
        return sum(
            1
            for i in self.issues
            if isinstance(i, StructuralIssue) and i.severity == IssueSeverity.ERROR
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1
            for i in self.issues
            if (isinstance(i, StructuralIssue) and i.severity == IssueSeverity.WARNING)
            or isinstance(i, SemanticIssue)
        )

    @property
    def has_critical_issues(self) -> bool:
        """Does this response have errors that make it unusable?"""
        return self.error_count > 0

    def add_issue(self, issue: StructuralIssue | SemanticIssue) -> None:
        self.issues.append(issue)

    def mark_passed(self) -> None:
        self.rules_passed += 1

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        # Convert issues to simple dicts for JSON serialization
        serialized_issues = []
        for issue in self.issues:
            if isinstance(issue, StructuralIssue):
                serialized_issues.append({**issue.model_dump(), "type": "structural"})
            else:
                serialized_issues.append({**issue.model_dump(), "type": "semantic"})
        data["issues"] = serialized_issues
        return data
