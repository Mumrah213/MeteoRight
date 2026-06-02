"""Abstract validator base class.

Every validator implements:
- check(response) -> list[Issue]
- rule_name: unique identifier
- severity: default severity level

Validators are composable — chain them in a pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas.forecast_raw import RawForecastResponse
from ..schemas.validation import IssueSeverity, SemanticIssue, StructuralIssue


class BaseValidator(ABC):
    """Abstract validator — all validators implement this interface."""

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Unique identifier for this validation rule."""

    @abstractmethod
    def check(self, response: RawForecastResponse) -> list[StructuralIssue | SemanticIssue]:
        """Run validation and return list of issues.

        Returns empty list if the response passes.
        """

    @property
    def severity(self) -> IssueSeverity:
        """Default severity for issues from this validator."""
        return IssueSeverity.ERROR


class ValidatorPipeline:
    """Chain multiple validators into a single pipeline.

    Usage:
        pipeline = ValidatorPipeline()
        pipeline.add(StructuralValidator())
        pipeline.add(SemanticValidator())
        issues = pipeline.check(response)
    """

    def __init__(self) -> None:
        self._validators: list[BaseValidator] = []

    def add(self, validator: BaseValidator) -> ValidatorPipeline:
        """Add a validator to the pipeline."""
        self._validators.append(validator)
        return self

    def check(self, response: RawForecastResponse) -> list[StructuralIssue | SemanticIssue]:
        """Run all validators in order and return combined issues."""
        issues: list[StructuralIssue | SemanticIssue] = []
        for validator in self._validators:
            issues.extend(validator.check(response))
        return issues

    @property
    def rule_names(self) -> list[str]:
        return [v.rule_name for v in self._validators]
