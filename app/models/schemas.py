"""
Pydantic Request/Response Schemas
=================================
The wire contract between the Fastify backend and this AI service.

Request models validate inbound JSON on the FastAPI routes (app/api/routes/v1.py).
Response models document and shape outbound JSON. Field names/shapes mirror exactly
what the routes, Celery tasks, and DB repositories produce — keep them in sync.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================================ #
# E-Consultation requests                                                       #
# ============================================================================ #

class TranscribeRequest(BaseModel):
    session_id: str
    # Provide exactly ONE audio source:
    audio_base64: Optional[str] = Field(None, description="Base64-encoded audio (short clips)")
    audio_url: Optional[str] = Field(None, description="Fetchable URL, e.g. a presigned "
                                     "object-storage link (preferred for long consultations)")
    audio_format: str = Field("wav", description="Container/codec, e.g. wav, mp3, m4a, ogg, webm")
    language: Optional[str] = Field(None, description="ISO code; None = auto-detect")
    speaker: str = Field("patient", description="Speaker label, e.g. patient or doctor")
    diarize_stereo: bool = Field(False, description="Split a stereo recording per channel "
                                 "and label by speaker (left=channel 0, right=channel 1)")
    channel_roles: Optional[list[str]] = Field(None, description="Labels for [left, right] "
                                               "channels, e.g. [\"doctor\", \"patient\"]")
    callback_url: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self) -> "TranscribeRequest":
        if bool(self.audio_base64) == bool(self.audio_url):
            raise ValueError("provide exactly one of audio_base64 or audio_url")
        if self.channel_roles is not None and len(self.channel_roles) != 2:
            raise ValueError("channel_roles must have exactly 2 entries: [left, right]")
        return self


class SOAPRequest(BaseModel):
    session_id: str
    transcript: str
    patient_id: str
    doctor_id: str
    specialty: Optional[str] = None
    callback_url: Optional[str] = None


class SummaryRequest(BaseModel):
    session_id: str
    transcript: str
    callback_url: Optional[str] = None


class TriageRequest(BaseModel):
    patient_id: str
    symptoms: str
    age: Optional[int] = Field(None, ge=0, le=130)
    vital_signs: Optional[dict] = None
    medical_history: Optional[list[str]] = None
    callback_url: Optional[str] = None


# ============================================================================ #
# Health Social Media requests                                                  #
# ============================================================================ #

class ModerationRequest(BaseModel):
    content_id: str
    content_type: str = Field("post", description="post, comment, message, etc.")
    text: str
    author_id: str
    callback_url: Optional[str] = None


class RecommendationRequest(BaseModel):
    user_id: str
    context: str = Field("", description="Free-text context for the recommendation")
    user_interests: list[str] = Field(default_factory=list)
    user_conditions: list[str] = Field(default_factory=list)
    limit: int = Field(10, ge=1, le=100)
    callback_url: Optional[str] = None


class SentimentRequest(BaseModel):
    content_id: str
    text: str
    callback_url: Optional[str] = None


class SupportAssistRequest(BaseModel):
    ticket_id: str
    subject: str = ""
    message: str
    category_hint: Optional[str] = Field(None, description="Customer-selected category, if any")
    callback_url: Optional[str] = None


class MisinfoCheckRequest(BaseModel):
    text: str = Field(..., description="The health claim to fact-check")
    entity_id: Optional[str] = Field(None, description="For your correlation")
    k: int = Field(4, ge=1, le=10, description="Number of evidence passages to retrieve")
    callback_url: Optional[str] = None


class Citation(BaseModel):
    source: str
    url: Optional[str] = None
    snippet: str
    score: Optional[float] = None


class MisinfoResult(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "unsupported", "not_health_claim", "unverified"]
    confidence: float
    rationale: str
    flag: bool = Field(..., description="True if it should go to human review (advisory, never auto-remove)")
    needs_human_review: bool
    citations: list[Citation] = Field(default_factory=list)
    llm_used: bool = False
    model_version: str


# ============================================================================ #
# Tier-1 SYNCHRONOUS safety endpoints (gate a user action inline, < ~400ms)     #
# These run rules first (deterministic, GPU-independent) and escalate ambiguous #
# cases to the local LLM, failing safe if the LLM is unavailable.               #
# ============================================================================ #

class TriageSyncRequest(BaseModel):
    symptoms: str
    age: Optional[int] = Field(None, ge=0, le=130)
    medical_history: Optional[list[str]] = None
    patient_id: Optional[str] = Field(None, description="For audit correlation only")
    use_llm: bool = Field(True, description="Escalate non-red-flag cases to the LLM for an urgency opinion")


class TriageResult(BaseModel):
    urgency_level: Literal["emergency", "urgent", "semi_urgent", "non_urgent", "self_care"]
    urgency_score: float
    red_flag_symptoms: list[str] = Field(default_factory=list,
        description="Deterministic safety net — if non-empty, treat as emergency")
    recommended_specialty: str = "General Practitioner"
    reasoning: str
    self_care_advice: Optional[str] = None
    llm_used: bool = False
    model_version: str


class ModerateTextRequest(BaseModel):
    text: str
    context: Literal["chat", "post", "comment", "review", "report"] = "post"
    entity_id: Optional[str] = Field(None, description="For your correlation")
    deep_scan: bool = Field(False,
        description="Also run the LLM for nuanced toxicity (slower; use for reports/borderline)")


class ToxicitySignal(BaseModel):
    score: float
    label: Literal["clean", "toxic", "harassment"]
    matched: list[str] = Field(default_factory=list)


class DistressSignal(BaseModel):
    detected: bool
    severity: Literal["none", "low", "high"]
    escalate_to_human: bool = Field(...,
        description="If true, route to the human crisis/clinical workflow regardless of `action`")
    matched: list[str] = Field(default_factory=list)


class ModerateResult(BaseModel):
    action: Literal["allow", "flag", "block"]
    toxicity: ToxicitySignal
    distress: DistressSignal
    llm_used: bool = False
    model_version: str


# ============================================================================ #
# Semantic search & recommendations (sync, embedding-backed content store)      #
# ============================================================================ #

class ContentItem(BaseModel):
    id: str
    text: str = Field(..., description="Searchable text — title + summary works well")
    type: str = Field("content", description="article, guide, post, etc.")
    metadata: dict = Field(default_factory=dict)


class IndexContentRequest(BaseModel):
    items: list[ContentItem]


class RemoveContentRequest(BaseModel):
    ids: list[str]


class IndexResponse(BaseModel):
    indexed: Optional[int] = None
    removed: Optional[int] = None
    total: int


class SearchRequest(BaseModel):
    query: str
    k: int = Field(10, ge=1, le=50)
    content_type: Optional[str] = Field(None, description="Filter to a single type")


class RecommendQuery(BaseModel):
    user_interests: list[str] = Field(default_factory=list)
    user_conditions: list[str] = Field(default_factory=list)
    seed_content_ids: list[str] = Field(default_factory=list,
        description="content_ids the user recently engaged with — drives cold-start "
                    "recs ('more like this') when interests aren't known")
    context: str = Field("", description="e.g. 'after_consultation'")
    k: int = Field(10, ge=1, le=50)
    exclude: list[str] = Field(default_factory=list, description="content_ids already seen")


class SearchHit(BaseModel):
    content_id: str
    type: str
    title: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    count: int
    model_version: str


class RecommendResponse(BaseModel):
    recommendations: list[SearchHit]
    strategy: Literal["content_based", "similar_to_recent", "trending"]
    model_version: str


# ============================================================================ #
# Generic async responses                                                       #
# ============================================================================ #

class EnqueueResponse(BaseModel):
    """Returned immediately when a task is queued (or served from cache)."""
    task_id: str
    status: Literal["queued", "complete"]
    result: Optional[dict] = None
    priority: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Returned when Fastify polls GET /api/v1/task/{task_id}."""
    task_id: str
    status: Literal["pending", "processing", "complete", "failed"]
    result: Optional[Any] = None
    error: Optional[str] = None


# ============================================================================ #
# Database query responses (PostgreSQL-backed history endpoints)                #
# ============================================================================ #

class TaskHistoryItem(BaseModel):
    task_id: str
    task_type: str
    status: str
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    urgency_level: Optional[str] = None
    verdict: Optional[str] = None
    sentiment: Optional[str] = None
    duration_ms: Optional[int] = None
    result_summary: Optional[dict] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class InferenceMetricItem(BaseModel):
    # `model_name` would otherwise trip Pydantic's protected "model_" namespace.
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    total_calls: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_top_score: float


class AuditEventItem(BaseModel):
    audit_id: str
    action: str
    module: str
    entity_id: Optional[str] = None
    data: Optional[dict] = None
    created_at: Optional[str] = None


# ============================================================================ #
# Tier-3 — drug-interaction check (#12)                                         #
# ============================================================================ #

class DrugInteractionRequest(BaseModel):
    medications: list[str] = Field(..., min_length=1,
        description="Medication names (brand or generic; dosage text is tolerated)")
    patient_id: Optional[str] = Field(None, description="For your correlation")
    use_llm: bool = Field(False,
        description="Add an advisory LLM pass for additional candidate interactions (slower)")


class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: Literal["contraindicated", "major", "moderate", "minor"]
    effect: str
    management: Optional[str] = None
    source: str


class DrugInteractionResult(BaseModel):
    medications_checked: list[str]
    unrecognized: list[str] = Field(default_factory=list,
        description="Could not be matched to the reference set — NOT cleared as safe")
    interactions: list[DrugInteraction] = Field(default_factory=list)
    interaction_count: int
    highest_severity: Literal["contraindicated", "major", "moderate", "minor", "none"]
    has_contraindication: bool
    llm_advisory: list[DrugInteraction] = Field(default_factory=list,
        description="Unverified additional candidates from the LLM — for human review only")
    advisory: str
    needs_human_review: bool = True
    llm_used: bool = False
    model_version: str


# ============================================================================ #
# Tier-3 — pre-consultation symptom intake (#13)                                #
# ============================================================================ #

class SymptomIntakeRequest(BaseModel):
    symptoms: str = Field(..., min_length=1, description="Patient's free-text complaint")
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = None
    duration: Optional[str] = Field(None, description="e.g. '3 days', 'since this morning'")
    existing_conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    patient_id: Optional[str] = Field(None, description="For your correlation")
    use_llm: bool = True


class SymptomIntakeResult(BaseModel):
    chief_complaint: str
    structured_summary: str
    clarifying_questions: list[str] = Field(default_factory=list)
    urgency_level: Literal["emergency", "urgent", "semi_urgent", "non_urgent", "self_care"]
    red_flag_symptoms: list[str] = Field(default_factory=list,
        description="Deterministic safety net — if non-empty, treat as emergency")
    recommended_specialty: str = "General Practitioner"
    emergency: bool
    patient_guidance: str
    disclaimer: str
    degraded: bool = False
    needs_human_review: bool = True
    llm_used: bool = False
    model_version: str
