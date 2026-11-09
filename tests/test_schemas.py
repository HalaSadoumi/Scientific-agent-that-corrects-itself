from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent.schemas import (
    BoundingBox,
    Chunk,
    ContextBundle,
    Document,
    InputItem,
    InputKind,
    ObjectMention,
    Quantity,
    SourceTier,
    VisualKind,
    VisualObject,
)

SESSION = uuid4()


def _bbox(page: int = 1) -> BoundingBox:
    return BoundingBox(page=page, x0=0.1, y0=0.1, x1=0.5, y1=0.3)


def test_text_input_requires_text() -> None:
    with pytest.raises(ValidationError):
        InputItem(session_id=SESSION, kind=InputKind.TEXT, text="   ")


def test_text_input_ok() -> None:
    item = InputItem(session_id=SESSION, kind=InputKind.TEXT, text="How far is Betelgeuse?")
    assert item.text == "How far is Betelgeuse?"


def test_file_input_requires_storage_key() -> None:
    with pytest.raises(ValidationError):
        InputItem(session_id=SESSION, kind=InputKind.DOCUMENT, filename="paper.pdf")


def test_sha256_must_be_hex() -> None:
    with pytest.raises(ValidationError):
        InputItem(
            session_id=SESSION,
            kind=InputKind.IMAGE,
            storage_key="uploads/a.png",
            sha256="not-a-hash",
        )


def test_bbox_must_be_ordered_and_normalized() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(page=1, x0=0.5, y0=0.1, x1=0.2, y1=0.3)
    with pytest.raises(ValidationError):
        BoundingBox(page=1, x0=0.1, y0=0.1, x1=1.5, y1=0.3)


def test_chunk_takes_page_from_bbox() -> None:
    chunk = Chunk(document_id=uuid4(), order=0, text="The limit is 1.4 M☉.", bbox=_bbox(page=3))
    assert chunk.page == 3


def test_chunk_page_must_match_bbox() -> None:
    with pytest.raises(ValidationError):
        Chunk(document_id=uuid4(), order=0, text="x", page=2, bbox=_bbox(page=3))


def test_chunk_carries_astronomy_fields() -> None:
    chunk = Chunk(
        document_id=uuid4(),
        order=1,
        text="Betelgeuse has about 16.5 M☉.",
        quantities=[Quantity(raw_text="16.5 M☉", value=16.5, unit="solMass")],
        object_mentions=[ObjectMention(raw_name="Betelgeuse")],
    )
    assert chunk.quantities[0].unit == "solMass"
    assert chunk.object_mentions[0].resolved_id is None


def test_visual_object_needs_an_owner() -> None:
    with pytest.raises(ValidationError):
        VisualObject(kind=VisualKind.FIGURE, storage_key="crops/f1.png")


def test_document_defaults_to_user_tier() -> None:
    assert Document().source_tier is SourceTier.T0_USER


def test_bundle_needs_prompt_or_input() -> None:
    with pytest.raises(ValidationError):
        ContextBundle(session_id=SESSION)


def test_bundle_rejects_unknown_document_reference() -> None:
    chunk = Chunk(document_id=uuid4(), order=0, text="orphan")
    with pytest.raises(ValidationError):
        ContextBundle(session_id=SESSION, prompt="hi", chunks=[chunk])


def test_bundle_round_trips_through_json() -> None:
    doc = Document(title="Paper")
    chunk = Chunk(document_id=doc.id, order=0, text="Hello", bbox=_bbox())
    bundle = ContextBundle(session_id=SESSION, prompt="Explain", documents=[doc], chunks=[chunk])
    restored = ContextBundle.model_validate_json(bundle.model_dump_json())
    assert restored == bundle
