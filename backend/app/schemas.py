"""Pydantic request / response models. These are the API contract the React
frontend (frontend/src/types.ts) mirrors."""
from __future__ import annotations

from datetime import date as Date
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from . import config


# ------------------------------------------------------------------ requests
class KnownLocation(BaseModel):
    mode: Literal["known"] = "known"
    state: str = Field(min_length=1)
    district: str = Field(min_length=1)


class CustomLocation(BaseModel):
    mode: Literal["custom"] = "custom"
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    elevation: float = Field(ge=-500, le=9000)
    distance_to_coast_km: float = Field(ge=0, le=3000)


LocationInput = Annotated[Union[KnownLocation, CustomLocation], Field(discriminator="mode")]


class KnnrOptions(BaseModel):
    enabled: bool = True
    draws: int = Field(default=50, ge=10, le=200)
    top_k: int = Field(default=10, ge=3, le=50)


class PredictRequest(BaseModel):
    date: Date
    location: LocationInput
    knnr: KnnrOptions = KnnrOptions()

    @field_validator("date")
    @classmethod
    def _date_in_range(cls, v: Date) -> Date:
        lo, hi = Date.fromisoformat(config.MIN_DATE), Date.fromisoformat(config.MAX_DATE)
        if not lo <= v <= hi:
            raise ValueError(f"date must be between {lo} and {hi}")
        return v


class ValidationRequest(BaseModel):
    window: Literal[3, 5, 7] = 3
    scope: Literal["district", "all"] = "district"
    location: Optional[KnownLocation] = None   # required when scope == "district"
    n_days: int = Field(default=200, ge=20, le=1000)
    draws: int = Field(default=20, ge=5, le=100)
    top_k: int = Field(default=10, ge=3, le=50)


# ----------------------------------------------------------------- responses
class DistrictInfo(BaseModel):
    name: str
    latitude: float
    longitude: float


class StateInfo(BaseModel):
    name: str
    districts: list[DistrictInfo]


class MetaResponse(BaseModel):
    states: list[StateInfo]
    min_date: str
    max_date: str
    monsoon_months: list[int]
    knnr_windows: list[int]
    severity_order: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    districts: int = 0
    climate_models: int = 0
    knnr_windows: list[int] = []
    detail: Optional[str] = None


class MatchedFrom(BaseModel):
    district_name: str
    state_name: str
    similarity_distance: float


class LocationOut(BaseModel):
    state_name: str
    district_name: str
    district_code: str
    latitude: float
    longitude: float
    elevation: float
    distance_to_coast_km: float
    is_unlisted_location: bool
    matched_from: Optional[MatchedFrom] = None


class SeverityProb(BaseModel):
    key: str
    probability: float


class MostLikely(BaseModel):
    bin: str
    probability: float
    p25: float
    p75: float


class ExpectedPrecip(BaseModel):
    weighted_mean: float
    weighted_median: float
    most_likely: MostLikely


class ClimateValue(BaseModel):
    key: str
    label: str
    unit: str
    value: float
    p10: Optional[float] = None   # from the quantile model, when available
    p90: Optional[float] = None


class PredictionOut(BaseModel):
    date: str
    rain_probability: float
    severity_probabilities: list[SeverityProb]
    expected_precipitation_mm: ExpectedPrecip
    climate: list[ClimateValue]


class HourPoint(BaseModel):
    hour: int
    mean: float
    p10: float
    p25: float
    p75: float
    p90: float


class MatchInfo(BaseModel):
    window: int
    matched_on: str


class HourlyOut(BaseModel):
    hours: list[HourPoint]
    daily_total: float
    peak_hour: int
    peak_value: float
    simulations: int
    windows: list[int]
    any_pattern_matching: bool
    target_in_library: bool
    match_info: list[MatchInfo]
    analogue_dates: list[str]


HourlyStatus = Literal["ok", "disabled", "library_missing", "no_rain", "no_analogue", "error"]


class PredictResponse(BaseModel):
    location: LocationOut
    prediction: PredictionOut
    hourly: Optional[HourlyOut] = None
    hourly_status: HourlyStatus
    hourly_message: Optional[str] = None


class ValidationMetrics(BaseModel):
    mean_mae: float
    median_mae: float
    peak_hit_rate: float
    mean_nse: Optional[float] = None
    median_nse: Optional[float] = None


class ValidationSummary(BaseModel):
    n_days: int
    knnr: ValidationMetrics
    naive: ValidationMetrics
    beats_naive: bool
    match_counts: dict[str, int]


class ValidationRow(BaseModel):
    date: str
    daily_total_mm: float
    mae: float
    peak_hour_hit: bool
    nse: Optional[float] = None
    naive_mae: float
    naive_peak_hour_hit: bool
    naive_nse: Optional[float] = None
    match_type: str


class ValidationDetail(BaseModel):
    date: str
    daily_total_mm: float
    actual: list[float]
    predicted: list[float]
    p10: list[float]
    p90: list[float]


class ValidationResponse(BaseModel):
    window: int
    summary: ValidationSummary
    rows: list[ValidationRow]
    detail: list[ValidationDetail]
