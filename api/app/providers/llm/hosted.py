"""Hosted free-tier provider (Groq, Google AI Studio, OpenRouter)."""

from __future__ import annotations

import json
import logging

import httpx
from pydantic import ValidationError

from ...config import Settings
from ...schemas import Answer, Citation, Passage

log = logging.getLogger("rahbar.provider.hosted")

FALLBACK_ANSWER = Answer(
    reasoning="System fallback invoked due to hosted API failure or schema validation error.",
    answer="The service is temporarily unable to retrieve an answer from the hosted model. Please try again shortly.",
    citations=[],
    answered_from_context=False,
    confidence="low",
)


class HostedProvider:
    name = "hosted"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _build_prompt(self, question: str, passages: list[Passage]) -> str:
        evidence_lines = []
        for idx, p in enumerate(passages, start=1):
            evidence_lines.append(
                f"[{idx}] Source: {p.source} | Section: {p.section} | Page: {p.page or 'N/A'}\n{p.text}\n"
            )
        evidence_block = "\n".join(evidence_lines)

        return (
            "You are RAHBAR, an official university academic advisor. "
            "Output MUST be valid JSON adhering strictly to schema:\n"
            "{\n"
            '  "reasoning": "step-by-step reasoning",\n'
            '  "answer": "grounded factual response",\n'
            '  "citations": [{"source": "file.pdf", "section": "1.21", "page": 1}],\n'
            '  "answered_from_context": true,\n'
            '  "confidence": "high" | "medium" | "low"\n'
            "}\n\n"
            f"=== EVIDENCE ===\n{evidence_block}\n=== END EVIDENCE ===\n\n"
            f"Question: {question}\n"
        )

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Call hosted OpenAI-compatible chat completion endpoint with JSON mode and retry."""
        if not passages:
            return Answer(
                reasoning="No passages available above the relevance floor.",
                answer="The provided documents do not contain information regarding this topic.",
                citations=[],
                answered_from_context=False,
                confidence="low",
            )

        if not self.settings.hosted_base_url or not self.settings.llm_api_key:
            log.warning("Hosted provider invoked without HOSTED_BASE_URL or LLM_API_KEY; returning fallback")
            return FALLBACK_ANSWER

        prompt = self._build_prompt(question, passages)
        url = f"{self.settings.hosted_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},
        }

        max_retries = 2
        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code != 200:
                        raise RuntimeError(f"Hosted provider returned HTTP {resp.status_code}: {resp.text[:100]}")
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]

                parsed = Answer.model_validate_json(content)
                valid_passages = {(p.source.lower(), p.section) for p in passages}
                verified_citations = [
                    c for c in parsed.citations if (c.source.lower(), c.section) in valid_passages
                ]
                return Answer(
                    reasoning=parsed.reasoning,
                    answer=parsed.answer,
                    citations=verified_citations,
                    answered_from_context=parsed.answered_from_context,
                    confidence=parsed.confidence,
                )

            except (ValidationError, Exception) as exc:
                log.warning("Hosted generation attempt %d failed (%s)", attempt, exc)
                if attempt == max_retries:
                    log.error("Exhausted retries on hosted endpoint; using safe fallback")
                    return FALLBACK_ANSWER

        return FALLBACK_ANSWER
