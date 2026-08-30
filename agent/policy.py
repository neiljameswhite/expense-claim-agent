"""
Policy loader — parses the expense policy into clause-level chunks and
supplies each check with only the sections it needs.

Two design decisions live here.

Per-check retrieval rather than whole-document prompting. Narrowing the
context to the decision is cheaper, faster and more accurate: a model asked
to find the subsistence limit in thirty pages will occasionally read across
clauses and apply the wrong one, and that failure is quiet.

Deterministic mapping rather than vector search. The policy is small and
structured, so a fixed check-to-clause mapping is more reliable than semantic
retrieval and is exactly reproducible. The seam is the same either way —
swap resolve_sections() for a vector query and nothing downstream changes.

The cost of that precision is a coupling: the mapping encodes knowledge about
the policy's structure, and a renumbered policy breaks it. That coupling is
made detectable by validate(), which runs at load and fails loudly rather
than letting a stale mapping retrieve the wrong text at inference.

See solution design v0.4 section 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.assessment import (
    CHECK_CATEGORY_CORRECT,
    CHECK_COST_EXPLANATION,
    CHECK_MILEAGE_JOURNEY,
    CHECK_OTHER_DESCRIPTION,
    CHECK_SUBSISTENCE_ELIGIBLE,
    CHECK_WITHIN_LIMIT,
)

# --------------------------------------------------------------------------
# Which sections each check is entitled to see.
#
# A check receives its sections and nothing else. Keep these narrow: every
# extra section is context the model must sift, and a source of cross-clause
# error.
# --------------------------------------------------------------------------

CHECK_SECTIONS: dict[int, list[str]] = {
    CHECK_CATEGORY_CORRECT: ["2", "9"],             # categories and their detail
    CHECK_SUBSISTENCE_ELIGIBLE: ["5", "9"],         # eligibility and which limit
    CHECK_MILEAGE_JOURNEY: ["6"],                   # required journey elements
    CHECK_WITHIN_LIMIT: ["2", "5", "6", "9"],       # limits, subsistence, mileage
    CHECK_COST_EXPLANATION: ["4"],                  # exceeding a limit
    CHECK_OTHER_DESCRIPTION: ["3", "2"],            # the Other category, and 2.3
}


class PolicyStructureError(Exception):
    """The loaded policy does not contain the sections the mapping expects.

    Raised at load time. A policy whose structure has changed must not be
    used with a mapping written against the previous version — the failure
    mode is a confident wrong answer citing the wrong clause.
    """


@dataclass(frozen=True)
class Clause:
    """A numbered clause, e.g. 4.2, with its text."""

    ref: str          # "4.2"
    section: str      # "4"
    text: str

    def cited(self) -> str:
        return f"{self.ref} {self.text}"


@dataclass(frozen=True)
class Section:
    """A top-level section, e.g. 4, with its heading and clauses."""

    number: str       # "4"
    heading: str      # "Exceeding a category limit"
    clauses: list[Clause] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        parts = [f"## {self.number}. {self.heading}"]
        parts.extend(c.cited() for c in self.clauses)
        parts.extend(self.tables)
        return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")

# Two clause styles appear in the policy and both must parse:
#   **4.2** A rationale is supported where...
#   **7.1 Subsistence.** Meals and refreshments purchased...
# The second carries a name inside the bold markers.
_CLAUSE_RE = re.compile(r"^\*\*(\d+\.\d+)([^*]*)\*\*\s*(.*)$")

_TABLE_RE = re.compile(r"^\|")


class Policy:
    """A parsed policy document."""

    def __init__(self, version: str, sections: dict[str, Section], source: str):
        self.version = version
        self.sections = sections
        self.source = source

    # -- retrieval ---------------------------------------------------------

    def resolve_sections(self, check_id: int) -> list[Section]:
        """The sections this check is entitled to see."""
        refs = CHECK_SECTIONS.get(check_id)
        if refs is None:
            raise KeyError(f"No section mapping for check {check_id}")
        return [self.sections[r] for r in refs]

    def context_for(self, check_id: int) -> str:
        """The policy text to place in this check's prompt."""
        return "\n\n---\n\n".join(s.as_text() for s in self.resolve_sections(check_id))

    def clause(self, ref: str) -> Clause:
        """A single clause by reference, e.g. '4.2'."""
        section_no = ref.split(".")[0]
        section = self.sections.get(section_no)
        if section is None:
            raise KeyError(f"No section {section_no}")
        for c in section.clauses:
            if c.ref == ref:
                return c
        raise KeyError(f"No clause {ref}")

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Confirm the document still contains what the mapping expects.

        Called at load. Failing here is the point: a mapping written against
        a previous policy version must not silently retrieve the wrong text.
        """
        problems: list[str] = []

        expected = sorted({s for refs in CHECK_SECTIONS.values() for s in refs}, key=int)
        for ref in expected:
            section = self.sections.get(ref)
            if section is None:
                problems.append(f"section {ref} is referenced by the check mapping but absent")
            elif not section.clauses and not section.tables:
                problems.append(f"section {ref} ({section.heading}) is empty")

        # Anchors: specific clauses the checks reason about by name. If these
        # move, the mapping is stale even though the sections still exist.
        anchors = {
            "2.3": "Other must not be used to avoid a category limit",
            "4.2": "the grounds on which an excess is supported",
            "5.2": "the two subsistence limits",
            "6.4": "the elements a mileage journey description must state",
        }
        for ref, why in anchors.items():
            try:
                self.clause(ref)
            except KeyError:
                problems.append(f"clause {ref} is absent — expected to carry {why}")

        if problems:
            raise PolicyStructureError(
                "Policy structure does not match the check mapping:\n  - "
                + "\n  - ".join(problems)
            )

    def __repr__(self) -> str:
        return f"<Policy v{self.version}, {len(self.sections)} sections>"


def parse(text: str, version: str = "unknown", source: str = "") -> Policy:
    """Parse policy markdown into sections and clauses."""
    sections: dict[str, Section] = {}
    current: Section | None = None

    for raw in text.splitlines():
        line = raw.rstrip()

        m = _SECTION_RE.match(line)
        if m:
            current = Section(number=m.group(1), heading=m.group(2), clauses=[], tables=[])
            sections[current.number] = current
            continue

        if current is None:
            continue

        m = _CLAUSE_RE.match(line)
        if m:
            ref = m.group(1)
            inside = m.group(2).strip()   # a name carried inside the bold, if any
            after = m.group(3).strip()
            body = f"{inside} {after}".strip() if inside else after
            current.clauses.append(Clause(ref=ref, section=current.number, text=body))
            continue

        if _TABLE_RE.match(line):
            current.tables.append(line)
            continue

        # Continuation of the preceding clause: bullets and wrapped lines.
        if line.strip() and current.clauses:
            last = current.clauses[-1]
            current.clauses[-1] = Clause(
                ref=last.ref,
                section=last.section,
                text=(last.text + "\n" + line.strip()).strip(),
            )

    # Collapse each section's table lines into one block.
    for s in sections.values():
        if s.tables:
            block = "\n".join(s.tables)
            s.tables.clear()
            s.tables.append(block)

    return Policy(version=version, sections=sections, source=source)


_VERSION_RE = re.compile(r"\*\*Version\s+([\d.]+)\*\*")


def load(path: str | Path, validate: bool = True) -> Policy:
    """Load and parse a policy file, validating its structure by default."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    m = _VERSION_RE.search(text)
    version = m.group(1) if m else "unknown"

    policy = parse(text, version=version, source=path.name)
    if validate:
        policy.validate()
    return policy
