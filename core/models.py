"""Pydantic data models enforcing strict separation of facts, summaries, and interpretations."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ReliabilityState(str, Enum):
    """Categorical reliability states grounded in evidence, without arbitrary numerical scores."""
    OFFICIAL = "verified/official"      # Directly from official company announcement/repo/docs
    CROSS_CHECKED = "cross-checked"    # Confirmed by 2+ independent reputable secondary sources
    REPORTED = "reported"              # Reported by a single reputable secondary source
    UNCERTAIN = "uncertain"            # Conflicting reports or unverified speculation


class SourceFact(BaseModel):
    """Short, strictly grounded factual claim supported by the source (Change 1)."""
    claim: str = Field(..., description="Short factual statement directly supported by source")
    source_name: str = Field(..., description="Name of the reporting publication or official domain")
    source_url: str = Field(..., description="Original article or documentation URL")


class ConflictInfo(BaseModel):
    """Preserves conflicting claims across sources rather than silently merging them."""
    topic: str = Field(..., description="Attribute or subject of conflict (e.g. 'pricing', 'release date')")
    conflicting_claims: List[str] = Field(..., description="The differing claims reported by different sources")
    sources_involved: List[str] = Field(..., description="Sources providing the differing claims")
    explanation: str = Field(..., description="Explanation of the discrepancy")


class ToolMetadata(BaseModel):
    """Verified tool details preferred from official documentation/repositories."""
    name: str
    official_url: str
    repo_url: Optional[str] = None
    is_free: Optional[bool] = None
    pricing_details: Optional[str] = None
    install_guide: Optional[List[str]] = None


class NewsFactSheet(BaseModel):
    """Core data representation of a news event with strict claim and interpretation boundaries."""
    event_id: str = Field(..., description="Unique deterministic identifier for the event cluster")
    headline: str = Field(..., description="Clear, concise headline")
    primary_source_name: str = Field(..., description="Primary publication or official entity name")
    primary_source_url: str = Field(..., description="Canonical or primary article URL")
    supporting_sources: List[str] = Field(default_factory=list, description="URLs of cross-checking sources")

    # Reliability and Official Status
    reliability: ReliabilityState = Field(..., description="Reliability state of this item")
    is_official_source: bool = Field(default=False, description="True if primary source is official domain/repo")

    # Tri-Partite Content Separation
    source_reported_facts: List[SourceFact] = Field(
        default_factory=list,
        description="Short, strictly grounded factual claims supported by the source"
    )
    ai_summary: str = Field(..., description="Objective, concise summary for the morning digest")
    ai_interpretation: Optional[str] = Field(
        default=None,
        description="Village-level analogy and practical learning insight (clearly labelled)"
    )

    # Conflict Information
    conflicts: List[ConflictInfo] = Field(default_factory=list, description="Documented disagreements across sources")

    # Optional Tool Launch Details
    tool_metadata: Optional[ToolMetadata] = None

    category: str = Field(default="ai_around_the_world", description="'ai_around_the_world' or 'what_people_are_doing'")
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    retrieved_at: Optional[str] = None
    event_date: Optional[str] = None


class ChatMessage(BaseModel):
    """Representation of an interactive message scoped to a specific authorized user."""
    message_id: Optional[int] = None
    user_id: int = Field(..., description="Telegram user ID owning this message")
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Text content of the message")
    timestamp: str = Field(..., description="ISO 8601 timestamp")

