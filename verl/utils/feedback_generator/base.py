from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeedbackRequest:
    question: str
    student_answer: str
    ground_truth: Optional[str] = None
    data_source: Optional[str] = None


@dataclass
class FeedbackResponse:
    feedback_text: str
    completion_tokens: int = 0
    api_attempts: int = 1
    success: bool = True


class AbstractFeedbackGenerator(ABC):
    @abstractmethod
    def generate(self, requests: list[FeedbackRequest]) -> list[FeedbackResponse]:
        """Generate feedback for a batch of requests."""

    def get_last_generation_metrics(self) -> dict[str, float]:
        """Return backend-specific metrics for the most recent generate() call."""
        return {}

    def get_step_relative_metrics(self, step_duration_seconds: float) -> dict[str, float]:
        """Return optional metric measuring impact of feedback generation on step duration."""
        return {}

    def close(self) -> None:
        """Optional cleanup hook for implementations with network/session resources."""
