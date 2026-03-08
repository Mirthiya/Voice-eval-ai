from __future__ import annotations
import json
import time
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    prompt: str
    response: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    is_mock: bool = False

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "response": self.response,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "is_mock": self.is_mock,
        }


class OllamaClient:
    GENERATE_ENDPOINT = "/api/generate"
    HEALTH_ENDPOINT   = "/api/tags"

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "llama3.2:1b",
                 temperature: float = 0.0,
                 seed: int = 42,
                 timeout: int = 60):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self._available = self._health_check()

    def _health_check(self) -> bool:
        try:
            url = self.host + self.HEALTH_ENDPOINT
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            logger.warning("Ollama not reachable — using mock LLM responses")
            return False

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 512) -> LLMResponse:
        if self._available:
            return self._call_ollama(prompt, system, max_tokens)
        return self._mock_response(prompt)

    def _call_ollama(self, prompt: str, system: Optional[str],
                     max_tokens: int) -> LLMResponse:
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        payload: dict = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": max_tokens,
                "num_ctx": 2048,
            },
        }

        body = json.dumps(payload).encode("utf-8")
        url  = self.host + self.GENERATE_ENDPOINT

        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            logger.warning(f"Ollama request failed: {e} — falling back to mock")
            return self._mock_response(prompt)
        except Exception as e:
            logger.warning(f"Ollama error: {e} — falling back to mock")
            return self._mock_response(prompt)

        latency_ms = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            prompt=prompt,
            response=data.get("response", "").strip(),
            model=data.get("model", self.model),
            latency_ms=latency_ms,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            is_mock=False,
        )

    def _mock_response(self, prompt: str) -> LLMResponse:
        import hashlib
        h = hashlib.md5(prompt.encode()).hexdigest()[:6]
        templates = {
            "hallucination": '{"is_hallucination": false, "confidence": 0.12, "reasoning": "The response is grounded in the provided context."}',
            "similarity":    '{"score": 0.82, "reasoning": "Both sentences convey the same core meaning."}',
            "default":       f"Mock response [{h}]: This is a deterministic placeholder.",
        }
        if "hallucination" in prompt.lower():
            response = templates["hallucination"]
        elif "similar" in prompt.lower() or "semantic" in prompt.lower():
            response = templates["similarity"]
        else:
            response = templates["default"]

        t0 = time.perf_counter()
        time.sleep(0.005)
        latency_ms = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            prompt=prompt, response=response, model="mock-llm",
            latency_ms=latency_ms,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response.split()),
            is_mock=True,
        )