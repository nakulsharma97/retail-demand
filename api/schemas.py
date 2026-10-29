from pydantic import BaseModel, Field
from typing import Optional


class ForecastRequest(BaseModel):
    store_id:    str = Field(..., example="S001")
    sku_id:      str = Field(..., example="SKU0001")
    horizon_weeks: int = Field(default=4, ge=1, le=52)
    include_confidence: bool = False


class ForecastResponse(BaseModel):
    store_id:      str
    sku_id:        str
    forecasts:     list[dict]   # [{week_start, predicted_qty, lower, upper}]
    model_used:    str
    generated_at:  str
