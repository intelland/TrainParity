"""Import helpers for process-safe case specifications."""

from __future__ import annotations

from importlib import import_module

from trainparity.protocols import AccumulationCase, ProcessResumeCase, ResumeCase


class CaseImportError(ValueError):
    """Raised when an import specification cannot produce a resume case."""


def load_case(spec: str) -> ResumeCase:
    """Load a zero-argument case class from a ``module:attribute`` spec."""
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise CaseImportError("case must use the form 'package.module:ClassName'")
    try:
        target = getattr(import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as error:
        raise CaseImportError(f"cannot import case {spec!r}: {error}") from error
    if not isinstance(target, type):
        raise CaseImportError(f"case target {spec!r} must be a class")
    try:
        instance = target()
    except TypeError as error:
        raise CaseImportError(f"case class {spec!r} must have a zero-argument constructor") from error
    if not isinstance(instance, ResumeCase):
        raise CaseImportError(f"case {spec!r} does not implement ResumeCase")
    return instance


def load_process_case(spec: str) -> ProcessResumeCase:
    """Load a zero-argument command-oriented case from an import spec."""
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise CaseImportError("case must use the form 'package.module:ClassName'")
    try:
        target = getattr(import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as error:
        raise CaseImportError(f"cannot import case {spec!r}: {error}") from error
    if not isinstance(target, type):
        raise CaseImportError(f"case target {spec!r} must be a class")
    try:
        instance = target()
    except TypeError as error:
        raise CaseImportError(f"case class {spec!r} must have a zero-argument constructor") from error
    if not isinstance(instance, ProcessResumeCase):
        raise CaseImportError(f"case {spec!r} does not implement ProcessResumeCase")
    return instance


def load_accumulation_case(spec: str) -> AccumulationCase:
    """Load a zero-argument accumulation case from an import spec."""
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise CaseImportError("case must use the form 'package.module:ClassName'")
    try:
        target = getattr(import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as error:
        raise CaseImportError(f"cannot import case {spec!r}: {error}") from error
    if not isinstance(target, type):
        raise CaseImportError(f"case target {spec!r} must be a class")
    try:
        instance = target()
    except TypeError as error:
        raise CaseImportError(f"case class {spec!r} must have a zero-argument constructor") from error
    if not isinstance(instance, AccumulationCase):
        raise CaseImportError(f"case {spec!r} does not implement AccumulationCase")
    return instance
