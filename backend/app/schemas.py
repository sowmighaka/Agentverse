from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class JobInputType(str, Enum):
    text = "text"
    url = "url"


class TailorRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    job_input: str = Field(..., min_length=1)
    job_input_type: JobInputType
    blog_urls: list[str] = Field(default_factory=list)
    github_org: str | None = None

    @field_validator("resume_text", "job_input")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("blog_urls")
    @classmethod
    def drop_blank_blog_urls(cls, value: list[str]) -> list[str]:
        return [url.strip() for url in value if url and url.strip()]

    @field_validator("github_org")
    @classmethod
    def normalize_github_org(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().strip("/")
        return normalized or None


class TailorResponse(BaseModel):
    session_id: str
    rewritten_resume: dict[str, Any]
    cover_letter: str
    diff_explanation: str
    verification_status: str
    confidence_notes: list[str]
    sources_checked: list[dict[str, Any]]
    agent_trace: list[dict[str, Any]]
    validation_report: dict[str, Any] | None = None


class RenderResumePdfRequest(BaseModel):
    resume: dict[str, Any]
