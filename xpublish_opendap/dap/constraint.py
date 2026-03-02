"""Recursive-descent parser for DAP2 constraint expressions.

Grammar (simplified):
    CE              = projection *("&" selection)
    projection      = proj_item *("," proj_item)
    proj_item       = identifier *("[" hyperslab "]")
    hyperslab       = index [":" index [":" index]]
    selection        = identifier rel_op value
    rel_op          = "<" | "<=" | ">" | ">=" | "=" | "!=" | "=~"
    identifier      = name *("." name)
    value           = number | quoted_string
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import unquote

from xpublish_opendap.errors import ConstraintNotSupportedError, ConstraintSyntaxError


@dataclass(frozen=True)
class HyperSlab:
    """A single dimension slice: [start:stride:stop] (DAP-style, inclusive stop)."""

    start: int
    stop: int
    stride: int = 1


@dataclass(frozen=True)
class ProjectionItem:
    """A single item in a projection clause."""

    name: str
    slices: tuple[HyperSlab, ...] | None = None


@dataclass(frozen=True)
class SelectionClause:
    """A single selection sub-expression: field op value."""

    field: str
    operator: str
    value: str | float | int


@dataclass
class Constraint:
    """The fully parsed constraint expression."""

    projections: list[ProjectionItem] = field(default_factory=list)
    selections: list[SelectionClause] = field(default_factory=list)


class _Parser:
    """Recursive-descent parser for DAP2 constraint expressions."""

    def __init__(self, raw: str):
        self.raw = raw
        self.pos = 0

    def at_end(self) -> bool:
        return self.pos >= len(self.raw)

    def parse(self) -> Constraint:
        """Parse the full constraint expression."""
        constraint = Constraint()

        if self.at_end():
            return constraint

        # Split on unquoted '&' at the top level
        # First part is the projection, rest are selections
        parts = self._split_on_ampersand()

        if parts:
            # First part is always the projection clause (may be empty)
            proj_str = parts[0]
            if proj_str:
                constraint.projections = self._parse_projections(proj_str)

            # Remaining parts are selection clauses
            for sel_str in parts[1:]:
                self._parse_selection(sel_str)

        return constraint

    def _split_on_ampersand(self) -> list[str]:
        """Split the raw string on '&' respecting quoted strings."""
        parts = []
        current: list[str] = []
        in_quotes = False
        i = 0

        while i < len(self.raw):
            ch = self.raw[i]
            if ch == '"':
                in_quotes = not in_quotes
                current.append(ch)
            elif ch == "&" and not in_quotes:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
            i += 1

        parts.append("".join(current))
        return parts

    def _parse_projections(self, text: str) -> list[ProjectionItem]:
        """Parse a comma-separated projection clause."""
        items = []
        # Split on commas not inside brackets
        parts = self._split_projection_items(text)
        for part in parts:
            part = part.strip()  # noqa: PLW2901
            if part:
                items.append(self._parse_projection_item(part))
        return items

    def _split_projection_items(self, text: str) -> list[str]:
        """Split projection text on commas, respecting brackets and parentheses."""
        parts = []
        current: list[str] = []
        bracket_depth = 0
        paren_depth = 0

        for ch in text:
            if ch == "(":
                paren_depth += 1
                current.append(ch)
            elif ch == ")":
                paren_depth -= 1
                current.append(ch)
            elif ch == "[":
                bracket_depth += 1
                current.append(ch)
            elif ch == "]":
                bracket_depth -= 1
                current.append(ch)
            elif ch == "," and bracket_depth == 0 and paren_depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)

        parts.append("".join(current))
        return parts

    def _parse_projection_item(self, text: str) -> ProjectionItem:
        """Parse a single projection item like 'air[0:10][0:5]'."""
        # Check for function call syntax: name(args)
        if "(" in text and ")" in text:
            raise ConstraintNotSupportedError(
                f"Server-side functions are not supported: {text!r}",
            )

        # Extract name and slices
        bracket_start = text.find("[")
        if bracket_start == -1:
            return ProjectionItem(name=text.strip())

        name = text[:bracket_start].strip()
        if not name:
            raise ConstraintSyntaxError(f"Empty variable name in projection: {text!r}")

        slices_text = text[bracket_start:]
        slices = self._parse_hyperslabs(slices_text)

        return ProjectionItem(name=name, slices=tuple(slices))

    def _parse_hyperslabs(self, text: str) -> list[HyperSlab]:
        """Parse a sequence of [start:stride:stop] hyperslabs."""
        slabs = []
        i = 0

        while i < len(text):
            if text[i] == "[":
                end = text.index("]", i)
                slab_text = text[i + 1 : end]
                slabs.append(self._parse_single_hyperslab(slab_text))
                i = end + 1
            else:
                i += 1

        return slabs

    def _parse_single_hyperslab(self, text: str) -> HyperSlab:
        """Parse a single hyperslab like '0:2:10' or '0:10' or '5'.

        DAP2 hyperslab format: [start:stride:stop] or [start:stop] or [start]
        All indices inclusive.
        """
        parts = text.split(":")

        try:
            if len(parts) == 1:  # noqa: PLR2004
                # [index] - single element
                idx = int(parts[0])
                return HyperSlab(start=idx, stop=idx, stride=1)
            elif len(parts) == 2:  # noqa: PLR2004
                # [start:stop]
                start = int(parts[0])
                stop = int(parts[1])
                return HyperSlab(start=start, stop=stop, stride=1)
            elif len(parts) == 3:  # noqa: PLR2004
                # [start:stride:stop]
                start = int(parts[0])
                stride = int(parts[1])
                stop = int(parts[2])
                if stride <= 0:
                    raise ConstraintSyntaxError(
                        f"Stride must be positive, got {stride}",
                    )
                return HyperSlab(start=start, stop=stop, stride=stride)
            else:
                raise ConstraintSyntaxError(f"Invalid hyperslab: [{text}]")
        except ValueError as e:
            raise ConstraintSyntaxError(
                f"Invalid hyperslab index in [{text}]: {e}",
            ) from e

    def _parse_selection(self, text: str) -> SelectionClause:
        """Parse a single selection clause like 'field>value'."""
        raise ConstraintNotSupportedError(
            f"Selection constraints are not yet supported: {text!r}",
        )


def parse_dap2_constraint(raw: str) -> Constraint:
    """Parse a DAP2 constraint expression string.

    Args:
        raw: The raw constraint expression (URL query string, possibly percent-encoded).

    Returns:
        A parsed Constraint object.

    Raises:
        ConstraintSyntaxError: If the expression is malformed.
        ConstraintNotSupportedError: If the expression uses unsupported features.
    """
    decoded = unquote(raw)
    parser = _Parser(decoded)
    return parser.parse()


def parse_dap4_constraint(raw: str) -> Constraint:
    """Parse a DAP4 constraint expression string.

    DAP4 CE differences from DAP2:
    - Semicolons separate projections (not commas/ampersands)
    - Variable names may have a leading '/' which is stripped
    - The expression is typically extracted from a ``dap4.ce`` query parameter

    This normalizes DAP4 syntax to reuse the existing _Parser internals.

    Args:
        raw: The raw DAP4 constraint expression.

    Returns:
        A parsed Constraint object.

    Raises:
        ConstraintSyntaxError: If the expression is malformed.
        ConstraintNotSupportedError: If the expression uses unsupported features.
    """
    decoded = unquote(raw)

    if not decoded.strip():
        return Constraint()

    # DAP4 uses semicolons to separate projections; normalize to commas
    # Also strip leading '/' from variable names
    normalized = _normalize_dap4_ce(decoded)

    parser = _Parser(normalized)
    return parser.parse()


def _normalize_dap4_ce(text: str) -> str:
    """Normalize DAP4 constraint expression to DAP2-compatible syntax.

    - Replace semicolons (projection separator) with commas
    - Strip leading '/' from variable names
    - Filters (DAP4 selections) use '|' separator — not yet supported
    """
    # Split on '|' to separate projections from filters
    parts = text.split("|", 1)
    projection_part = parts[0]

    if len(parts) > 1 and parts[1].strip():
        raise ConstraintNotSupportedError(
            f"DAP4 filter expressions are not yet supported: {parts[1]!r}",
        )

    # Replace semicolons with commas for projection items
    normalized = projection_part.replace(";", ",")

    # Strip leading '/' from variable names (but not from inside brackets)
    result: list[str] = []
    i = 0
    in_bracket = False
    strip_next_slash = True

    while i < len(normalized):
        ch = normalized[i]
        if ch == "[":
            in_bracket = True
            result.append(ch)
            strip_next_slash = False
        elif ch == "]":
            in_bracket = False
            result.append(ch)
        elif ch == ",":
            result.append(ch)
            strip_next_slash = True
        elif ch == "/" and not in_bracket and strip_next_slash:
            # Skip leading slash on variable names
            pass
        else:
            result.append(ch)
            strip_next_slash = False
        i += 1

    return "".join(result)
