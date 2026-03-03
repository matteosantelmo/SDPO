from __future__ import annotations

import asyncio
import importlib
import os
from typing import Any

from .base import AbstractFeedbackGenerator, FeedbackRequest, FeedbackResponse


class OpenAIAPIFeedbackGenerator(AbstractFeedbackGenerator):
    """Feedback generator using an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str,
        endpoint: str = "https://api.swissai.cscs.ch/v1",
        request_timeout_seconds: float = 60.0,
        max_concurrent_requests: int = 8,
        max_retries: int = 3,
        initial_retry_delay_seconds: float = 1.0,
        max_retry_delay_seconds: float = 30.0,
        fail_on_error: bool = False,
        prompt_template: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a feedback generator that calls an OpenAI-compatible chat completions API.
        Args:
            model: The model name to use for generating feedback.
            api_key: The API key for authenticating with the OpenAI API.
            endpoint: The base URL for the OpenAI API (default to swiss-ai: "https://api.swissai.cscs.ch/v1").
            request_timeout_seconds: Timeout for individual API requests in seconds (default: 60.0).
            max_concurrent_requests: Maximum number of concurrent API requests (default: 8).
            max_retries: Maximum number of retries for failed API requests (default: 3).
            initial_retry_delay_seconds: Initial delay before retrying a failed request, in seconds (default: 1.0).
            max_retry_delay_seconds: Maximum delay between retries in seconds (default: 30.0).
            fail_on_error: If True, raise an exception if feedback generation fails for any request.
            prompt_template: Optional custom prompt template string. If not provided, a default template will be used.
            generation_kwargs: Optional dictionary for additional parameters (e.g. temperature, max_tokens).
        """
        if not model:
            raise ValueError("feedback_generator.model is required for OpenAI backend")
        if not api_key:
            raise ValueError("feedback_generator.api_key is required for OpenAI backend")

        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        self.max_concurrent_requests = max_concurrent_requests
        self.max_retries = max_retries
        self.initial_retry_delay_seconds = initial_retry_delay_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.fail_on_error = fail_on_error
        self.generation_kwargs = generation_kwargs or {}
        self.prompt_template = prompt_template or (
            "You are an expert teacher. Given a question and a student's answer, provide concise, actionable feedback "
            "to help the student improve.\n\n"
            "Question:\n{question}\n\n"
            "Student answer:\n{student_answer}\n\n"
            "{ground_truth_block}"
            "Feedback:"
        )

        if self.max_concurrent_requests <= 0:
            raise ValueError("feedback_generator.max_concurrent_requests must be positive")
        if self.max_retries < 0:
            raise ValueError("feedback_generator.max_retries must be >= 0")
        if self.initial_retry_delay_seconds <= 0:
            raise ValueError("feedback_generator.initial_retry_delay_seconds must be positive")
        if self.max_retry_delay_seconds <= 0:
            raise ValueError("feedback_generator.max_retry_delay_seconds must be positive")
        if self.max_retry_delay_seconds < self.initial_retry_delay_seconds:
            raise ValueError("feedback_generator.max_retry_delay_seconds must be >= initial_retry_delay_seconds")

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for feedback_generator.backend=openai. "
                "Install it in your environment (e.g. `pip install openai`)."
            ) from e
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.endpoint,
            timeout=self.request_timeout_seconds,
        )


    @staticmethod
    def from_config(cfg) -> "OpenAIAPIFeedbackGenerator":
        api_key = cfg.get("api_key", "")
        api_key_env_var = cfg.get("api_key_env_var", "OPENAI_API_KEY")
        if not api_key:
            api_key = os.getenv(api_key_env_var, "")

        generation_kwargs = cfg.get("generation_kwargs", None)
        generation_kwargs = dict(generation_kwargs) if generation_kwargs is not None else None

        return OpenAIAPIFeedbackGenerator(
            model=cfg.get("model", ""),
            api_key=api_key,
            endpoint=cfg.get("endpoint", "https://api.swissai.cscs.ch/v1"),
            request_timeout_seconds=cfg.get("request_timeout_seconds", 60.0),
            max_concurrent_requests=cfg.get("max_concurrent_requests", 8),
            max_retries=cfg.get("max_retries", 3),
            initial_retry_delay_seconds=cfg.get("initial_retry_delay_seconds", 1.0),
            max_retry_delay_seconds=cfg.get("max_retry_delay_seconds", 30.0),
            fail_on_error=cfg.get("fail_on_error", False),
            prompt_template=cfg.get("prompt_template", None),
            generation_kwargs=generation_kwargs,
        )

    def _build_prompt(self, req: FeedbackRequest) -> str:
        ground_truth_block = ""
        if req.ground_truth is not None and req.ground_truth != "":
            ground_truth_block = f"Reference ground truth answer:\\n{req.ground_truth}\\n\\n"
        return self.prompt_template.format(
            question=req.question,
            student_answer=req.student_answer,
            ground_truth=req.ground_truth or "",
            ground_truth_block=ground_truth_block,
            data_source=req.data_source or "",
        )

    async def _call_chat_completions(self, prompt: str) -> str:
        delay = self.initial_retry_delay_seconds
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **self.generation_kwargs,
                )
                choices = response.choices
                if not choices:
                    raise RuntimeError("Feedback API returned no choices")

                content = choices[0].message.content or ""
                if isinstance(content, list):
                    content = "".join(getattr(c, "text", "") for c in content)
                print("Generated feedback")
                return str(content).strip()
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(delay, self.max_retry_delay_seconds))
                delay = min(delay * 2, self.max_retry_delay_seconds)

        raise last_error

    async def _generate_async(self, prompts: list[str]) -> list[str]:
        # Use a semaphore to limit concurrency of API calls
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def _run_one(prompt: str) -> str:
            async with semaphore:
                try:
                    return await self._call_chat_completions(prompt)
                except Exception as e:
                    if self.fail_on_error:
                        raise
                    print(f"Warning: feedback generation failed for one sample: {e}")
                    return ""

        tasks = [_run_one(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def generate(self, requests: list[FeedbackRequest]) -> list[FeedbackResponse]:
        prompts = [self._build_prompt(req) for req in requests]
        if not prompts:
            return []
        feedback_texts = asyncio.run(self._generate_async(prompts))
        return [FeedbackResponse(feedback_text=text) for text in feedback_texts]

    def close(self) -> None:
        try:
            close_fn = getattr(self.client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
