from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    """A rectangle on a page, in normalized page coordinates (0.0 to 1.0).

    Origin is the top-left corner of the page. Normalized values stay valid
    when a page is re-rendered at a different resolution.
    """

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1, description="1-based page number")
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have x1 > x0 and y1 > y0")
        return self


class Quantity(BaseModel):
    """A number with a unit found in text, kept alongside the original wording.

    `unit` is a canonical astropy-style unit string (for example "solMass").
    Normalization happens in a later step; until then `unit` may be None.
    """

    raw_text: str = Field(min_length=1, description="Exact text, e.g. '1.4 M☉'")
    value: float | None = None
    unit: str | None = None


class ObjectMention(BaseModel):
    """An astronomical object named in text, e.g. 'Betelgeuse' or 'HD 39801'.

    `resolved_id` is filled later by a catalog lookup (for example SIMBAD).
    """

    raw_name: str = Field(min_length=1)
    resolved_id: str | None = None
