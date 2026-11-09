import re
from datetime import UTC, datetime
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from agent.schemas.common import BoundingBox, ObjectMention, Quantity
from agent.schemas.enums import ChunkKind, InputKind, ParseStatus, SourceTier, VisualKind

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _now() -> datetime:
    return datetime.now(UTC)


class InputItem(BaseModel):
    """One thing the user supplied: a text prompt, an image, or a document."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    kind: InputKind
    created_at: datetime = Field(default_factory=_now)

    # Text prompts carry their text; files carry a storage key instead.
    text: str | None = None
    storage_key: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, v: str | None) -> str | None:
        if v is not None and not _SHA256.match(v):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return v

    @model_validator(mode="after")
    def _check_payload(self) -> Self:
        if self.kind is InputKind.TEXT:
            if not self.text or not self.text.strip():
                raise ValueError("a text input needs non-empty text")
        elif not self.storage_key:
            raise ValueError(f"a {self.kind.value} input needs a storage_key")
        return self


class Document(BaseModel):
    """A parsed document: an uploaded file now, a fetched paper later."""

    id: UUID = Field(default_factory=uuid4)
    input_id: UUID | None = None
    title: str | None = None
    source_tier: SourceTier = SourceTier.T0_USER
    parse_status: ParseStatus = ParseStatus.PENDING
    parser: str | None = Field(default=None, description="e.g. 'docling', 'pymupdf'")
    page_count: int | None = Field(default=None, ge=0)
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)


class Chunk(BaseModel):
    """A retrievable piece of text that always knows where it came from."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    order: int = Field(ge=0, description="Position within the document")
    kind: ChunkKind = ChunkKind.PARAGRAPH
    text: str = Field(min_length=1)

    page: int | None = Field(default=None, ge=1)
    bbox: BoundingBox | None = None
    section_path: list[str] = Field(default_factory=list)

    quantities: list[Quantity] = Field(default_factory=list)
    object_mentions: list[ObjectMention] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_page(self) -> Self:
        if self.bbox is not None:
            if self.page is None:
                self.page = self.bbox.page
            elif self.page != self.bbox.page:
                raise ValueError("chunk page does not match bbox page")
        return self


class VisualObject(BaseModel):
    """An image or region (figure, table, equation, page) that claims can point to."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID | None = None
    input_id: UUID | None = None
    kind: VisualKind
    storage_key: str = Field(min_length=1, description="Where the image file is stored")

    page: int | None = Field(default=None, ge=1)
    bbox: BoundingBox | None = None
    caption: str | None = None
    extracted_text: str | None = None
    extracted_data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_owner(self) -> Self:
        if self.document_id is None and self.input_id is None:
            raise ValueError("a visual object needs a document_id or an input_id")
        return self


class ContextBundle(BaseModel):
    """Everything ingestion hands to the Retrieval stage for one request."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    created_at: datetime = Field(default_factory=_now)

    prompt: str | None = None
    inputs: list[InputItem] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    visual_objects: list[VisualObject] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_not_empty(self) -> Self:
        if not (self.prompt or self.inputs):
            raise ValueError("a context bundle needs a prompt or at least one input")
        return self

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        document_ids = {d.id for d in self.documents}
        for chunk in self.chunks:
            if chunk.document_id not in document_ids:
                raise ValueError(f"chunk {chunk.id} references an unknown document")
        return self
