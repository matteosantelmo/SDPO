from __future__ import annotations

import time

from .base import AbstractFeedbackGenerator, FeedbackRequest, FeedbackResponse


class ReferenceBasedFeedbackGenerator(AbstractFeedbackGenerator):
    """Mock feedback generator that returns the provided reference solution text as a feedback.

    This backend is intended for debugging/ablation runs and does not call any external API.
    It returns the `ground_truth` field from each FeedbackRequest as feedback text.
    """

    def __init__(self, fail_on_missing_reference: bool = False) -> None:
        self.fail_on_missing_reference = fail_on_missing_reference
        self._last_generation_metrics: dict[str, float] = {}

    @staticmethod
    def from_config(cfg) -> "ReferenceBasedFeedbackGenerator":
        return ReferenceBasedFeedbackGenerator(
            fail_on_missing_reference=cfg.get("fail_on_missing_reference", False),
        )

    def generate(self, requests: list[FeedbackRequest]) -> list[FeedbackResponse]:
        started_at = time.perf_counter()
        responses: list[FeedbackResponse] = []

        for req in requests:
            reference_solution = (req.ground_truth or "").strip()
            if not reference_solution and self.fail_on_missing_reference:
                raise ValueError(
                    "ReferenceBasedFeedbackGenerator requires FeedbackRequest.ground_truth, "
                    "but none was provided for at least one sample"
                )
            feedback = ""
            if len(reference_solution) > 0:
                feedback += "Your solution is incorrect. The correct solution is:\n\n" 
                feedback += reference_solution

            responses.append(
                FeedbackResponse(
                    feedback_text=feedback,
                    completion_tokens=max(len(feedback.split()), 0),
                    api_attempts=1,
                    success=bool(feedback),
                )
            )

        duration_seconds = time.perf_counter() - started_at
        generated_feedback_count = sum(1 for item in responses if item.feedback_text)
        failed_feedback_count = len(responses) - generated_feedback_count
        generated_tokens = sum(max(item.completion_tokens, 0) for item in responses)

        self._last_generation_metrics = {
            "self_distillation/feedback_generator/generated_tokens": float(generated_tokens),
            "self_distillation/feedback_generator/tokens_per_second": float(
                generated_tokens / max(duration_seconds, 1e-8)
            ),
            "self_distillation/feedback_generator/total_duration_s": float(duration_seconds),
            "self_distillation/feedback_generator/requests_requested": float(len(responses)),
            "self_distillation/feedback_generator/api_requests_sent": 0.0,
            "self_distillation/feedback_generator/generated_feedback_count": float(generated_feedback_count),
            "self_distillation/feedback_generator/failed_feedback_count": float(failed_feedback_count),
            "self_distillation/feedback_generator/retry_count": 0.0,
        }

        return responses

    def get_last_generation_metrics(self) -> dict[str, float]:
        return dict(self._last_generation_metrics)
