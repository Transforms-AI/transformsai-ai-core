"""
Validation results: one report type for every check, core or project.

Validation never stops at the first problem; every issue is collected into a
``ValidationReport``. Only ``load_config`` raises, and it raises ``ConfigError``
carrying the full report.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Issue:
    """One problem found in a config.

    ``path`` is dotted (``cameras.surveillance.stream.bitrate``; ``""`` = the root),
    ``suggestion`` an optional "did you mean" key or value.
    """

    path: str
    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        text = f"{self.path or '<root>'}: {self.message}"
        if self.suggestion:
            text += f" (did you mean '{self.suggestion}'?)"
        return text


@dataclass
class ValidationReport:
    """Errors block a config; warnings don't. ``ok`` means no errors."""

    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, path: str, message: str, suggestion: str | None = None) -> None:
        self.errors.append(Issue(path, message, suggestion))

    def warning(self, path: str, message: str, suggestion: str | None = None) -> None:
        self.warnings.append(Issue(path, message, suggestion))

    def add(self, issue: Issue, *, as_error: bool) -> None:
        (self.errors if as_error else self.warnings).append(issue)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ConfigError(self)

    def __str__(self) -> str:
        if not self.errors and not self.warnings:
            return "config OK"
        lines = [f"error: {issue}" for issue in self.errors]
        lines += [f"warning: {issue}" for issue in self.warnings]
        return "\n".join(lines)


class ConfigError(ValueError):
    """A config that cannot be used. ``.report`` holds every issue found, not just the first."""

    def __init__(self, report: ValidationReport):
        self.report = report
        count = len(report.errors)
        super().__init__(f"invalid config ({count} error{'s' if count != 1 else ''}):\n{report}")
