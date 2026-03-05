from __future__ import annotations

from typing import Any

from .base import AbstractFeedbackGenerator
from .mock_generator import ReferenceBasedFeedbackGenerator
from .openai_generator import OpenAIAPIFeedbackGenerator


def build_feedback_generator(cfg: Any) -> AbstractFeedbackGenerator:    
    print(f"Building feedback generator with config: {cfg}")
    
    backend = cfg.get("backend", "openai")
    if backend == "openai":
        return OpenAIAPIFeedbackGenerator.from_config(cfg)
    if backend == "reference_based":
        return ReferenceBasedFeedbackGenerator.from_config(cfg)

    raise ValueError(f"Unsupported feedback generator backend: {backend}")
