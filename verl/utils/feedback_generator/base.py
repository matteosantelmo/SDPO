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


class AbstractFeedbackGenerator(ABC):
    @abstractmethod
    def generate(self, requests: list[FeedbackRequest]) -> list[FeedbackResponse]:
        """Generate feedback for a batch of requests."""

    def close(self) -> None:
        """Optional cleanup hook for implementations with network/session resources."""
