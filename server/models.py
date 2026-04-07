from pydantic import BaseModel
from typing import Literal, Optional
import uuid


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def improvement_pct(baseline: float, score: float) -> float:
    if baseline <= 0:
        return 0.0
    return round(((baseline - score) / baseline) * 100, 2)


# ── Request models ──

class RegisterRequest(BaseModel):
    client_version: str = "1.0"


class HeartbeatRequest(BaseModel):
    status: Literal["idle", "working"] = "working"
    current_hypothesis_id: Optional[str] = None


class HypothesisCreate(BaseModel):
    agent_id: str
    title: str
    description: str
    strategy_tag: Literal[
        "construction",
        "local_search",
        "metaheuristic",
        "constraint_relaxation",
        "decomposition",
        "hybrid",
        "data_structure",
        "other",
    ]
    parent_hypothesis_id: Optional[str] = None
    auto_claim: bool = True


class ExperimentCreate(BaseModel):
    agent_id: str
    hypothesis_id: Optional[str] = None
    algorithm_diff: str = ""
    score: float
    feasible: bool = True
    num_vehicles: int = 0
    total_distance: float = 0.0
    runtime_seconds: float = 0.0
    notes: str = ""
    route_data: Optional[dict] = None


class AdminAuth(BaseModel):
    admin_key: str


class AdminBroadcast(AdminAuth):
    message: str
    priority: Literal["normal", "high"] = "normal"


class MessageCreate(BaseModel):
    agent_id: Optional[str] = None
    agent_name: str
    content: str
    msg_type: Literal["agent", "synthesis", "milestone"] = "agent"


class KnowledgeUpdate(BaseModel):
    content: str
    updated_by: str = ""


# ── Response models ──

class AgentResponse(BaseModel):
    agent_id: str
    agent_name: str
    registered_at: str
    config: dict


class HypothesisResponse(BaseModel):
    hypothesis_id: str
    status: str
    fingerprint: str


class DuplicateResponse(BaseModel):
    error: str = "duplicate_hypothesis"
    similar_hypothesis_id: str
    similar_title: str
    similar_status: str
    suggestion: str = "Consider a different angle or combine with another approach."


class ExperimentResponse(BaseModel):
    experiment_id: str
    is_new_best: bool
    rank: int
    improvement_over_baseline_pct: float
    hypothesis_status_updated_to: Optional[str] = None
