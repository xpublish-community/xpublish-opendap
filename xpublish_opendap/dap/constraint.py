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

    @property
    def remaining(self) -> str:
        return self.raw[self.pos :]

    def at_end(self) -> bool:
        return self.pos >= len(self.raw)

    def peek(self) -> str:
        if self.at_end():
            return ''
        return self.raw[self.pos]

    def advance(self) -> str:
        ch = self.raw[self.pos]
        self.pos += 1
        return ch

    def expect(self, ch: str) -> None:
        if self.at_end() or self.raw[self.pos] != ch:
            raise ConstraintSyntaxError(
                f'Expected {ch!r} at position {self.pos}, got {self.peek()!r}'
            )
        self.pos += 1

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
                clause = self._parse_selection(sel_str)
                constraint.selections.append(clause)

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
            elif ch == '&' and not in_quotes:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
            i += 1

        parts.append(''.join(current))
        return parts

    def _parse_projections(self, text: str) -> list[ProjectionItem]:
        """Parse a comma-separated projection clause."""
        items = []
        # Split on commas not inside brackets
        parts = self._split_projection_items(text)
        for part in parts:
            part = part.strip()
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
            if ch == '(':
                paren_depth += 1
                current.append(ch)
            elif ch == ')':
                paren_depth -= 1
                current.append(ch)
            elif ch == '[':
                bracket_depth += 1
                current.append(ch)
            elif ch == ']':
                bracket_depth -= 1
                current.append(ch)
            elif ch == ',' and bracket_depth == 0 and paren_depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)

        parts.append(''.join(current))
        return parts

    def _parse_projection_item(self, text: str) -> ProjectionItem:
        """Parse a single projection item like 'air[0:10][0:5]'."""
        # Check for function call syntax: name(args)
        if '(' in text and ')' in text:
            raise ConstraintNotSupportedError(
                f'Server-side functions are not supported: {text!r}'
            )

        # Extract name and slices
        bracket_start = text.find('[')
        if bracket_start == -1:
            return ProjectionItem(name=text.strip())

        name = text[:bracket_start].strip()
        if not name:
            raise ConstraintSyntaxError(f'Empty variable name in projection: {text!r}')

        slices_text = text[bracket_start:]
        slices = self._parse_hyperslabs(slices_text)

        return ProjectionItem(name=name, slices=tuple(slices))

    def _parse_hyperslabs(self, text: str) -> list[HyperSlab]:
        """Parse a sequence of [start:stride:stop] hyperslabs."""
        slabs = []
        i = 0

        while i < len(text):
            if text[i] == '[':
                end = text.index(']', i)
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
        parts = text.split(':')

        try:
            if len(parts) == 1:
                # [index] - single element
                idx = int(parts[0])
                return HyperSlab(start=idx, stop=idx, stride=1)
            elif len(parts) == 2:
                # [start:stop]
                start = int(parts[0])
                stop = int(parts[1])
                return HyperSlab(start=start, stop=stop, stride=1)
            elif len(parts) == 3:
                # [start:stride:stop]
                start = int(parts[0])
                stride = int(parts[1])
                stop = int(parts[2])
                if stride <= 0:
                    raise ConstraintSyntaxError(f'Stride must be positive, got {stride}')
                return HyperSlab(start=start, stop=stop, stride=stride)
            else:
                raise ConstraintSyntaxError(f'Invalid hyperslab: [{text}]')
        except ValueError as e:
            raise ConstraintSyntaxError(f'Invalid hyperslab index in [{text}]: {e}') from e

    def _parse_selection(self, text: str) -> SelectionClause:
        """Parse a single selection clause like 'field>value'."""
        raise ConstraintNotSupportedError(
            f'Selection constraints are not yet supported: {text!r}'
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
