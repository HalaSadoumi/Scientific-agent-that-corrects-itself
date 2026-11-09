from enum import StrEnum


class InputKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"


class SourceTier(StrEnum):
    """Evidence trust tier. T0 is authoritative only about its own content."""

    T0_USER = "t0_user"
    T1_SCHOLARLY = "t1_scholarly"
    T2_REFERENCE = "t2_reference"
    T3_WEB = "t3_web"


class ParseStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    DONE = "done"
    FAILED = "failed"


class ChunkKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    EQUATION = "equation"
    OTHER = "other"


class VisualKind(StrEnum):
    PAGE = "page"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    IMAGE = "image"
