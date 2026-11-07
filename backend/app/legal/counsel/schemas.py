"""The trusted UI component contract, v1.0 (docs/echomind-legal/03_INTERACTIVE_COMPONENTS.md).

The browser renders a chat turn only from these shapes. The model never writes UI: it writes text, which lands in a
length-limited plain-text field; the server chooses the component, fills it from records it just authorized, and
validates the whole envelope here before it is stored or sent. Anything that does not validate is not shown.

Rules enforced by the models:
- every model forbids unknown fields (`extra="forbid"`), so nothing can smuggle HTML, CSS, scripts or handlers in;
- every list and string is bounded, so a hostile document cannot inflate a payload;
- identifiers match a strict pattern; in-app paths match an allow-list of the workspace's own pages; there are no URLs;
- an envelope holds at most three components (one primary, two supporting) and four suggested actions.

A conversation may cover several of the person's matters (2026-10-28): the envelope lists them, and every citation, event,
item and component that comes from one names its matter, so nothing from two matters is ever presented as one source.
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

SCHEMA_VERSION = "1.0"
MAX_COMPONENTS = 3
MAX_PRIORITY_ITEMS = 3
MAX_MATTERS = 5                                           # matters in one conversation

Ident = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,80}$")]
MatterNumber = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._~ -]{1,40}$")]
CitationId = Annotated[str, StringConstraints(pattern=r"^cit_[0-9a-f]{32}$")]
ComponentId = Annotated[str, StringConstraints(pattern=r"^cmp_[0-9a-f]{32}$")]
DiscrepancyId = Annotated[str, StringConstraints(pattern=r"^dsc_[0-9a-f]{32}$")]


def text(limit: int):
    return Annotated[str, StringConstraints(max_length=limit)]


# The workspace's own pages an action may open, and nothing else (no scheme, host, "..", query other than v/at).
_PAGE = re.compile(
    r"^/m/[A-Za-z0-9._~%-]{1,96}/(?:overview|documents|compare|obligations|drafting|meetings|exports|activity"
    r"|documents/[A-Za-z0-9_-]{1,64}/read(?:\?v=[A-Za-z0-9_-]{1,64}(?:&at=\d{1,9}-\d{1,9})?)?|documents/[A-Za-z0-9_-]{1,64}/definitions"
    r"|notes/[A-Za-z0-9_-]{1,64}|drafting/[A-Za-z0-9_-]{1,64}|meetings/[A-Za-z0-9_-]{1,64}(?:\?tab=minutes)?"
    r"|redlines/[A-Za-z0-9_-]{1,64}|summaries/[A-Za-z0-9_-]{1,64}|exports/[A-Za-z0-9_-]{1,64}|panel(?:/[A-Za-z0-9_-]{1,64})?)$"
    r"|^/(?:reviews|matters|drafts|activity|pipeline)$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Actions ──────────────────────────────────────────────────────────────────────
ActionType = Literal["ask", "open_source", "open_discrepancy", "open_page"]


class Action(Strict):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
    label: text(48)
    action_type: ActionType
    target_id: Optional[Ident] = None                     # a citation or discrepancy id, resolved by the server
    prompt: Optional[text(300)] = None                    # for "ask": the question the browser sends as the next turn
    document_ids: list[Ident] = Field(default_factory=list, max_length=10)   # for "ask": answer only from these
    path: Optional[text(300)] = None                      # for "open_page": one of the workspace's own pages

    @model_validator(mode="after")
    def _shape(self):
        if self.action_type == "ask" and not (self.prompt or "").strip():
            raise ValueError("an ask action needs a prompt")
        if self.action_type in ("open_source", "open_discrepancy") and not self.target_id:
            raise ValueError(f"{self.action_type} needs a target")
        if self.action_type == "open_page" and not (self.path and _PAGE.match(self.path)):
            raise ValueError("open_page may only open the workspace's own pages")
        if self.action_type != "open_page" and self.path is not None:
            raise ValueError("only open_page carries a path")
        return self


# ── Citations ────────────────────────────────────────────────────────────────────
class CitationRef(Strict):
    id: CitationId
    n: int = Field(ge=1, le=50)
    document_id: Ident
    document_title: text(300)
    version_number: Optional[int] = Field(default=None, ge=1)
    version_status: Optional[text(32)] = None
    page: Optional[int] = Field(default=None, ge=1)
    clause: Optional[text(40)] = None
    verified: bool                                        # true only when the quote was found in the cited passage
    removed: bool = False                                 # the document was purged after the answer
    sentences: list[text(600)] = Field(default_factory=list, max_length=8)   # the answer's sentences this span supports
    matter: Optional[MatterNumber] = None                 # in a conversation about several matters: the source's matter


# ── Component props ──────────────────────────────────────────────────────────────
AnswerState = Literal["answered", "partial", "unverified", "abstained"]


class CalculationFact(Strict):
    label: text(60)
    value: text(80)
    quote: text(300)                                      # the document's own words, as stored
    clause: Optional[text(40)] = None
    path: text(300)                                       # opens those words in the reader

    @model_validator(mode="after")
    def _own_page(self):
        if not _PAGE.match(self.path):
            raise ValueError("a fact opens the workspace's own reader")
        return self


class DerivedDate(Strict):
    label: text(80)
    display: text(80)


class CalculationProps(Strict):
    """Dates calculated from what one document states, never presented as stated (2026-10-29 QA RAG-003)."""
    document_title: text(300)
    facts: list[CalculationFact] = Field(default_factory=list, max_length=6)
    derived: list[DerivedDate] = Field(default_factory=list, max_length=4)
    assumptions: list[text(400)] = Field(default_factory=list, max_length=4)
    not_calculated: list[text(300)] = Field(default_factory=list, max_length=3)


class WorkingFigure(Strict):
    label: text(80)
    value: text(80)
    quote: text(400)                                      # the document's own words, as stored
    document_title: text(300)
    clause: Optional[text(40)] = None
    path: text(300)                                       # opens those words in the reader

    @model_validator(mode="after")
    def _own_page(self):
        if not _PAGE.match(self.path):
            raise ValueError("a figure opens the workspace's own reader")
        return self


class WorkingStep(Strict):
    label: text(120)
    working: text(600)                                    # the formula with its figures: "10% × USD 18,500"
    result: text(80)
    check: bool = False                                   # a condition: its result is "holds" or "does not hold"

    @model_validator(mode="after")
    def _check_result(self):
        if self.check and self.result not in ("holds", "does not hold"):
            raise ValueError("a check holds or does not hold")
        return self


class WorkingProps(Strict):
    """Figures worked out by rule from what the documents state, never presented as stated (research/working.py)."""
    steps: list[WorkingStep] = Field(min_length=1, max_length=10)
    figures: list[WorkingFigure] = Field(default_factory=list, max_length=16)
    assumptions: list[text(400)] = Field(default_factory=list, max_length=3)


class SourceAnswerProps(Strict):
    answer: text(8000)
    state: AnswerState
    citations: list[CitationRef] = Field(default_factory=list, max_length=30)
    unsupported: list[text(500)] = Field(default_factory=list, max_length=10)
    notes: list[text(400)] = Field(default_factory=list, max_length=3)
    understood: Optional[text(400)] = None                # what a follow-up was taken to ask
    absence: Optional[text(400)] = None                   # a decline: what the documents were found not to state (QA UX-001)
    calculation: Optional[CalculationProps] = None        # dates calculated from a document's words (QA RAG-003)
    working: Optional[WorkingProps] = None                # figures worked out from the documents' figures (research/working.py)
    # The answer opens by saying the documents do not state what was asked, then gives related cited text (EV19's narrower
    # option, 2026-11-03): the card says so instead of "partly supported", which describes neither.
    not_stated: bool = False

    @model_validator(mode="after")
    def _abstained_has_no_sources(self):
        if self.state == "abstained" and self.citations:
            raise ValueError("an abstention cites nothing")
        if self.absence is not None and self.state != "abstained":
            raise ValueError("only a decline says what is missing")
        if (self.calculation is not None or self.working is not None) and self.state == "abstained":
            raise ValueError("a decline calculates nothing")
        if self.not_stated and self.state == "abstained":
            raise ValueError("a decline is shown as a decline")
        return self


class PriorityItem(Strict):
    id: Ident
    kind: Literal["discrepancy", "new_documents", "review", "obligation", "scan", "council"]
    title: text(160)
    detail: text(400) = ""
    action: Action
    matter: Optional[MatterNumber] = None


class PriorityListProps(Strict):
    summary: text(400)
    items: list[PriorityItem] = Field(default_factory=list, max_length=MAX_PRIORITY_ITEMS)
    more: int = Field(default=0, ge=0)
    more_action: Optional[Action] = None
    empty: Optional[text(300)] = None


DiscrepancyStatus = Literal["potential", "in_review", "explained", "confirmed", "dismissed"]


class ComparisonSide(Strict):
    document_id: Ident
    document_title: text(300)
    version_id: Ident
    version_number: Optional[int] = Field(default=None, ge=1)
    version_status: Optional[text(32)] = None
    clause: Optional[text(40)] = None
    page: Optional[int] = Field(default=None, ge=1)
    value: text(300)
    context: text(1500)
    value_start: int = Field(ge=0)                        # the value's characters within `context`
    value_end: int = Field(ge=0)
    reader_path: Optional[text(300)] = None
    stale: bool = False                                   # a newer version of this document exists

    @model_validator(mode="after")
    def _value_inside_context(self):
        if not (0 <= self.value_start <= self.value_end <= len(self.context)):
            raise ValueError("the highlighted value must lie inside the context")
        if self.reader_path is not None and not _PAGE.match(self.reader_path):
            raise ValueError("reader_path must be the workspace's document reader")
        return self


class ReviewEntry(Strict):
    to_status: DiscrepancyStatus
    by: text(80)
    at: text(40)
    note: Optional[text(1000)] = None


class EvidenceComparisonProps(Strict):
    discrepancy_id: DiscrepancyId
    kind: text(24)
    title: text(160)
    status: DiscrepancyStatus
    a: ComparisonSide
    b: ComparisonSide
    explanation: text(1500) = ""
    caution: text(400)
    method: Literal["model", "rule"]
    history: list[ReviewEntry] = Field(default_factory=list, max_length=20)
    can_review: bool


IsoDate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class TimelineEvent(Strict):
    id: Ident
    date: Optional[IsoDate] = None                        # as the document states it; None when it does not
    date_text: text(80)                                   # the words of the date, as written
    certainty: Literal["stated", "uncertain", "undated"]
    kind: Literal["event", "deadline", "document_date", "obligation"]
    reported: bool = False                                # someone's recollection ("Witness A recalls …")
    superseded: bool = False                              # in a clause a signed amendment replaced
    text: text(400)
    status: Optional[text(80)] = None                     # an obligation's review state
    notes: list[text(300)] = Field(default_factory=list, max_length=3)
    source: ComparisonSide
    matter: Optional[MatterNumber] = None

    @model_validator(mode="after")
    def _undated_has_no_date(self):
        if (self.certainty == "undated") != (self.date is None):
            raise ValueError("an event has a date exactly when its date is stated")
        return self


class TimelineProps(Strict):
    summary: text(400)
    events: list[TimelineEvent] = Field(default_factory=list, max_length=60)
    more: int = Field(default=0, ge=0)
    caution: text(400)


class CaseRow(Strict):
    """One matter, as its records stand (counsel/cases.py): nothing in a row is generated."""
    matter_id: Ident
    number: MatterNumber
    title: text(200)
    client: text(200)
    status: Literal["open", "closed", "archived"]
    sensitivity: Literal["standard", "restricted", "highly_restricted"]
    documents: int = Field(ge=0)                          # documents the person may read
    open_inconsistencies: Optional[int] = Field(default=None, ge=0)
    next_deadline: Optional[IsoDate] = None
    next_deadline_text: Optional[text(200)] = None
    deadline_state: Optional[text(60)] = None             # "Confirmed by a lawyer" or "Not confirmed", never decided here
    last_activity: Optional[text(40)] = None
    path: text(140)

    @model_validator(mode="after")
    def _own_page(self):
        if not _PAGE.match(self.path):
            raise ValueError("a row opens the workspace's own matter page")
        return self


class CaseTableProps(Strict):
    summary: text(400)
    rows: list[CaseRow] = Field(min_length=1, max_length=MAX_MATTERS * 2)
    caution: text(400)


class TaskProgressProps(Strict):
    label: text(200)
    done: int = Field(ge=0)
    total: int = Field(ge=0)
    state: Literal["running", "done", "canceled", "failed"]


class NoticeProps(Strict):
    tone: Literal["info", "caution"]
    text: text(800)


class UnavailableProps(Strict):
    """What a stored component resolves to when its records are gone or no longer visible to the reader."""
    text: text(300)


def _own_page(path: Optional[str], what: str) -> None:
    if path is not None and not _PAGE.match(path):
        raise ValueError(f"{what} opens only the workspace's own pages")


# ── checklist: the matter's open obligations (counsel/tasks.py) ──────────────────
class ChecklistItem(Strict):
    id: Ident                                             # the obligation's id
    title: text(300)                                      # what is to be done, as the register words it
    party: Optional[text(80)] = None
    when: Optional[text(300)] = None                      # the document's words for when
    due_date: Optional[IsoDate] = None
    status: Literal["proposed", "confirmed", "done", "dismissed"]
    overdue: bool = False
    owner: Optional[text(80)] = None
    document_title: text(300)
    clause: Optional[text(40)] = None
    path: text(300)                                       # the document's words, in the reader
    version: int = Field(ge=1)                            # the obligation's version (each change is checked against it)

    @model_validator(mode="after")
    def _paths(self):
        _own_page(self.path, "an item")
        return self


class ChecklistProps(Strict):
    summary: text(400)
    items: list[ChecklistItem] = Field(default_factory=list, max_length=12)
    more: int = Field(default=0, ge=0)
    can_done: bool                                        # may tick items done or reopen them
    can_confirm: bool                                     # may confirm a deadline read from a document
    caution: text(400)
    register_path: text(140)

    @model_validator(mode="after")
    def _paths(self):
        _own_page(self.register_path, "the register link")
        return self


# ── case_status: where a matter stands, what happened lately, what is next (counsel/casestatus.py) ─
class CaseUpdateItem(Strict):
    kind: Literal["note", "call", "meeting", "filing", "hearing", "order", "instruction", "milestone", "stage"]
    kind_label: text(30)
    occurred_on: IsoDate
    author: text(80)
    text: text(600)                                       # what the person wrote (shortened), never model text
    document_title: Optional[text(300)] = None


class CaseTaskItem(Strict):
    title: text(300)
    kind: Literal["task", "key_date", "hearing", "filing"]
    due_on: Optional[IsoDate] = None
    overdue: bool = False
    assignee: Optional[text(80)] = None


class CaseStatusProps(Strict):
    summary: text(400)
    number: MatterNumber
    title: text(200)
    track_label: text(40)
    stages: list[text(60)] = Field(min_length=1, max_length=8)
    stage_index: int = Field(ge=0, le=7)
    days_in_stage: Optional[int] = Field(default=None, ge=0)
    updates: list[CaseUpdateItem] = Field(default_factory=list, max_length=5)
    tasks: list[CaseTaskItem] = Field(default_factory=list, max_length=5)
    open_tasks: int = Field(ge=0)
    overdue: int = Field(ge=0)
    as_of: text(40)
    can_post: bool
    path: text(140)                                       # the matter's overview, where updates and the plan are kept

    @model_validator(mode="after")
    def _paths(self):
        _own_page(self.path, "the case link")
        if self.stage_index >= len(self.stages):
            raise ValueError("the stage is one of the track's stages")
        return self


# ── approval_request: drafts waiting for this person's decision (counsel/approvals.py) ─
class ApprovalItem(Strict):
    artifact_id: Ident
    kind: text(40)                                        # "Legal draft", "Minutes", …
    title: text(200)
    version: int = Field(ge=1)
    author: text(80)
    created_at: text(40)
    preview: list[text(600)] = Field(default_factory=list, max_length=4)   # the first lines, as written
    gaps: int = Field(default=0, ge=0)                    # [[NEEDS INPUT]] gaps left (a legal draft needs none to be approved)
    consequence: text(400)                                # what approving does, and what it does not
    path: text(300)
    state: Literal["waiting", "approved", "changes_requested", "changed", "gone"] = "waiting"
    decided_by: Optional[text(80)] = None
    can_decide: bool

    @model_validator(mode="after")
    def _paths(self):
        _own_page(self.path, "an item")
        return self


class ApprovalRequestProps(Strict):
    summary: text(400)
    items: list[ApprovalItem] = Field(default_factory=list, max_length=MAX_PRIORITY_ITEMS)
    more: int = Field(default=0, ge=0)
    caution: text(400)


# ── guided_questions: at most three questions before something is made ────────────
class GuidedOption(Strict):
    value: Ident
    label: text(200)
    detail: Optional[text(200)] = None


class GuidedQuestion(Strict):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
    label: text(160)
    kind: Literal["choice", "text"]
    options: list[GuidedOption] = Field(default_factory=list, max_length=12)
    default: Optional[text(1000)] = None
    required: bool = True
    max_length: int = Field(default=200, ge=1, le=1000)

    @model_validator(mode="after")
    def _shape(self):
        if self.kind == "choice" and not self.options:
            raise ValueError("a choice has options")
        if self.kind == "choice" and self.default is not None and self.default not in {o.value for o in self.options}:
            raise ValueError("a choice's default is one of its options")
        if self.kind == "text" and self.options:
            raise ValueError("a text question has no options")
        return self


class GuidedQuestionsProps(Strict):
    purpose: Literal["draft", "panel"]
    title: text(160)
    intro: text(400)
    questions: list[GuidedQuestion] = Field(min_length=1, max_length=3)
    submit_label: text(48)
    note: text(400)


# ── document_preview: a draft made from the chat (counsel/drafting.py) ───────────
class PreviewField(Strict):
    label: text(120)
    value: text(300)
    status: Literal["sourced", "matter", "system", "entered", "needs_input"]


class PreviewBlock(Strict):
    type: Literal["heading", "paragraph"]
    text: text(1500)
    origin: Literal["template", "written", "edited"]


class DocumentPreviewProps(Strict):
    artifact_id: Ident
    title: text(200)
    template: text(200)
    version: int = Field(ge=1)
    status: Literal["draft", "approved", "superseded", "filed"]
    gaps: int = Field(ge=0)
    sources: int = Field(ge=0)
    fields: list[PreviewField] = Field(default_factory=list, max_length=12)
    blocks: list[PreviewBlock] = Field(default_factory=list, max_length=8)
    more_blocks: int = Field(default=0, ge=0)
    review: text(300)                                     # where it stands with the second lawyer
    path: text(300)
    can_edit: bool

    @model_validator(mode="after")
    def _paths(self):
        _own_page(self.path, "the draft")
        return self


# ── council_summary: the review panel's run (counsel/deliberation.py) ────────────
class PanelPosition(Strict):
    role: text(60)
    text: text(600)


class PanelIssue(Strict):
    issue: text(300)
    positions: list[PanelPosition] = Field(default_factory=list, max_length=5)
    disagreement: bool = False


class CouncilSummaryProps(Strict):
    review_id: Ident
    question: text(2000)
    area: text(80)
    status: Literal["queued", "running", "done", "needs_facts", "failed", "not_allowed"]
    issues: list[PanelIssue] = Field(default_factory=list, max_length=6)
    missing_facts: list[text(300)] = Field(default_factory=list, max_length=6)
    approval: Optional[text(80)] = None
    path: text(300)
    caution: text(400)
    # the Council (2026-11-05): the chair's conclusion first, then at most three points and gaps; figures in them no answer gave
    conclusion: Optional[text(400)] = None
    points: list[text(400)] = Field(default_factory=list, max_length=3)
    gaps: list[text(300)] = Field(default_factory=list, max_length=3)
    outcome: Optional[Literal["answered", "inconclusive", "stopped"]] = None
    figures_unsupported: list[text(40)] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def _paths(self):
        _own_page(self.path, "the review")
        return self


# ── numeric_chart: amounts the documents state (counsel/amounts.py) ──────────────
class ChartBar(Strict):
    id: Ident
    shown: text(80)                                       # the amount as written
    value: float = Field(ge=0)
    document_title: text(300)
    clause: Optional[text(40)] = None
    source: ComparisonSide


class NumericChartProps(Strict):
    summary: text(400)
    unit: text(12)                                        # one currency per chart: never added across currencies
    bars: list[ChartBar] = Field(default_factory=list, max_length=24)
    more: int = Field(default=0, ge=0)
    other_units: list[text(12)] = Field(default_factory=list, max_length=6)
    caution: text(400)


# ── graph_excerpt: parties and the documents that name them (counsel/parties.py) ─
class GraphNode(Strict):
    id: Ident
    label: text(160)
    kind: Literal["party", "document"]
    role: Optional[text(40)] = None                      # from the matter's records ("Client", "Adverse"), never inferred


class GraphEdge(Strict):
    id: Ident
    source: Ident                                         # a party node
    target: Ident                                         # a document node
    label: Literal["party to", "named in"]
    evidence: ComparisonSide


class GraphExcerptProps(Strict):
    summary: text(400)
    nodes: list[GraphNode] = Field(default_factory=list, max_length=18)
    edges: list[GraphEdge] = Field(default_factory=list, max_length=40)
    more: int = Field(default=0, ge=0)
    caution: text(400)

    @model_validator(mode="after")
    def _edges_join_nodes(self):
        ids = {n.id for n in self.nodes}
        if any(e.source not in ids or e.target not in ids for e in self.edges):
            raise ValueError("every edge joins two shown nodes")
        return self


# ── live_transcript: a meeting being recorded (counsel/live.py) ──────────────────
class LiveSegment(Strict):
    speaker: Optional[text(80)] = None
    text: text(1000)
    at: Optional[text(40)] = None


class LiveTranscriptProps(Strict):
    meeting_id: Ident
    title: text(200)
    state: Literal["recording", "paused", "not_recording"]
    consent: text(200)
    segments: list[LiveSegment] = Field(default_factory=list, max_length=12)
    earlier: int = Field(default=0, ge=0)
    path: text(300)
    caution: text(400)

    @model_validator(mode="after")
    def _paths(self):
        _own_page(self.path, "the meeting")
        return self


# ── Components and the envelope ──────────────────────────────────────────────────
class _Base(Strict):
    id: ComponentId
    version: int = Field(default=1, ge=1)
    matter_id: Optional[Ident] = None                    # in a conversation about several matters: the one this belongs to


class SourceAnswer(_Base):
    type: Literal["source_answer"]
    props: SourceAnswerProps


class PriorityList(_Base):
    type: Literal["priority_list"]
    props: PriorityListProps


class EvidenceComparison(_Base):
    type: Literal["evidence_comparison"]
    data_ref: ComponentId
    props: EvidenceComparisonProps


class Timeline(_Base):
    type: Literal["timeline"]
    props: TimelineProps


class CaseTable(_Base):
    type: Literal["case_table"]
    props: CaseTableProps


class TaskProgress(_Base):
    type: Literal["task_progress"]
    props: TaskProgressProps


class Notice(_Base):
    type: Literal["notice"]
    props: NoticeProps


class Unavailable(_Base):
    type: Literal["unavailable"]
    props: UnavailableProps


class Checklist(_Base):
    type: Literal["checklist"]
    data_ref: ComponentId
    props: ChecklistProps


class ApprovalRequest(_Base):
    type: Literal["approval_request"]
    data_ref: ComponentId
    props: ApprovalRequestProps


class GuidedQuestions(_Base):
    type: Literal["guided_questions"]
    props: GuidedQuestionsProps


class DocumentPreview(_Base):
    type: Literal["document_preview"]
    data_ref: ComponentId
    props: DocumentPreviewProps


class CouncilSummary(_Base):
    type: Literal["council_summary"]
    data_ref: ComponentId
    props: CouncilSummaryProps


class NumericChart(_Base):
    type: Literal["numeric_chart"]
    props: NumericChartProps


class GraphExcerpt(_Base):
    type: Literal["graph_excerpt"]
    props: GraphExcerptProps


class LiveTranscript(_Base):
    type: Literal["live_transcript"]
    data_ref: ComponentId
    props: LiveTranscriptProps


class CaseStatus(_Base):
    type: Literal["case_status"]
    props: CaseStatusProps


Component = Annotated[Union[SourceAnswer, PriorityList, EvidenceComparison, Timeline, CaseTable, TaskProgress, Notice, Unavailable,
                            Checklist, ApprovalRequest, GuidedQuestions, DocumentPreview, CouncilSummary, NumericChart, GraphExcerpt,
                            LiveTranscript, CaseStatus],
                      Field(discriminator="type")]
COMPONENT_TYPES = ("source_answer", "priority_list", "evidence_comparison", "timeline", "case_table", "task_progress", "notice",
                   "unavailable", "checklist", "approval_request", "guided_questions", "document_preview", "council_summary",
                   "numeric_chart", "graph_excerpt", "live_transcript", "case_status")
# Components drawn from records that change (an obligation's state, a draft's newest version, a meeting): only a
# reference is stored, and they are rebuilt from the records, as the reader may see them, every time they are shown.
LIVE_TYPES = ("checklist", "approval_request", "document_preview", "council_summary", "live_transcript")


class MatterRef(Strict):
    id: Ident
    number: MatterNumber
    title: text(200)


class Envelope(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    message_id: Ident
    matter_id: Ident
    matters: list[MatterRef] = Field(default_factory=list, max_length=MAX_MATTERS)   # when the conversation covers several
    summary: text(400) = ""
    components: list[Component] = Field(default_factory=list, max_length=MAX_COMPONENTS)
    suggested_actions: list[Action] = Field(default_factory=list, max_length=4)

    @field_validator("components")
    @classmethod
    def _unique_ids(cls, v):
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("component ids must be unique")
        return v


def validate_envelope(data: dict) -> dict:
    """The envelope as it may be sent: validated, then dumped back to plain JSON. Raises pydantic.ValidationError."""
    out = Envelope.model_validate(data).model_dump(mode="json", exclude_none=True)
    if not out.get("matters"):                             # a conversation about one matter: the envelope it always had
        out.pop("matters", None)
    return out


def validate_component(data: dict) -> dict:
    from pydantic import TypeAdapter
    return TypeAdapter(Component).validate_python(data).model_dump(mode="json", exclude_none=True)
