"""Program mutators: stub (no LLM) and optional LLM via API key."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

import numpy as np

from squarepack.evolve.program import SearchProgram, section7_seed_program
from squarepack.evolve.sandbox import SandboxError, validate_program_dict


class Mutator(Protocol):
    def mutate(
        self,
        parent: SearchProgram,
        *,
        generation: int,
        context: dict[str, Any],
        rng: np.random.Generator,
    ) -> SearchProgram: ...


def detect_llm_provider() -> tuple[str | None, str | None]:
    """Return (provider, env_var_name) if a usable API key is present."""
    for provider, var in (
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("gemini", "GOOGLE_API_KEY"),
    ):
        val = os.environ.get(var, "").strip()
        if val:
            return provider, var
    return None, None


class StubMutator:
    """Deterministic hyperparameter jitter — proves loop wiring without an LLM."""

    def __init__(self, name: str = "stub") -> None:
        self.name = name

    def mutate(
        self,
        parent: SearchProgram,
        *,
        generation: int,
        context: dict[str, Any],
        rng: np.random.Generator,
    ) -> SearchProgram:
        child = SearchProgram.from_dict(parent.to_dict())
        child.parent = parent.name
        child.method = "stub_mutate"
        child.generation = generation
        child.name = f"stub_gen{generation}_{int(rng.integers(0, 1_000_000))}"

        # Jitter knobs in safe ranges
        child.eta0 = float(np.clip(parent.eta0 * float(rng.uniform(0.5, 1.5)), 1e-4, 5e-2))
        child.temperature = float(
            np.clip(parent.temperature * float(rng.uniform(0.3, 3.0)), 1e-5, 5e-2)
        )
        child.cool_factor = float(np.clip(parent.cool_factor + float(rng.normal(0, 0.002)), 0.97, 0.9995))
        child.seeded_expand = float(
            np.clip(parent.seeded_expand + float(rng.normal(0, 0.001)), 1.0001, 1.02)
        )
        child.max_lbfgs_iter = int(
            np.clip(parent.max_lbfgs_iter + int(rng.integers(-20, 21)), 20, 200)
        )
        probs = np.asarray(parent.normalized_move_probs(), dtype=np.float64)
        probs = probs * rng.uniform(0.5, 1.5, size=probs.shape)
        probs = np.clip(probs, 1e-3, None)
        probs /= probs.sum()
        child.move_probs = probs.tolist()
        child.notes = f"stub mutate from {parent.name}; ctx_keys={sorted(context)[:8]}"
        validate_program_dict(child.to_dict())
        return child


class LLMMutator:
    """Call an LLM API to propose a mutated SearchProgram JSON object.

    Requires a real API key in the environment. Raises if unavailable.
    """

    def __init__(self, provider: str | None = None) -> None:
        detected, var = detect_llm_provider()
        self.provider = provider or detected
        self.env_var = var
        if self.provider is None:
            raise RuntimeError(
                "No LLM API key in environment (checked OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY)"
            )

    def mutate(
        self,
        parent: SearchProgram,
        *,
        generation: int,
        context: dict[str, Any],
        rng: np.random.Generator,
    ) -> SearchProgram:
        prompt = {
            "instruction": (
                "Mutate the search-program hyperparameters (move probabilities, "
                "eta0, temperature, cool_factor, seeded_expand, max_lbfgs_iter). "
                "Return ONLY a JSON object with those fields. Do not emit code "
                "that writes trivial grid packings; fitness already penalizes that."
            ),
            "parent": parent.to_dict(),
            "context": {
                "basin_summary": context.get("basin_summary"),
                "failure_notes": context.get("failure_notes"),
                "train_score": context.get("train_score"),
            },
            "generation": generation,
        }
        raw = self._complete(json.dumps(prompt))
        data = self._parse_json_object(raw)
        data = validate_program_dict(data)
        # Fill identity fields locally (do not trust the model for provenance).
        merged = parent.to_dict()
        for k in (
            "eta0",
            "temperature",
            "cool_factor",
            "seeded_expand",
            "max_lbfgs_iter",
            "energy_feasible",
            "move_probs",
            "notes",
        ):
            if k in data:
                merged[k] = data[k]
        child = SearchProgram.from_dict(merged)
        child.parent = parent.name
        child.method = "llm_mutate"
        child.generation = generation
        child.name = f"llm_gen{generation}_{int(rng.integers(0, 1_000_000))}"
        return child

    def _complete(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._openai(prompt)
        if self.provider == "anthropic":
            return self._anthropic(prompt)
        if self.provider == "gemini":
            return self._gemini(prompt)
        raise RuntimeError(f"unsupported provider {self.provider}")

    def _openai(self, prompt: str) -> str:
        key = os.environ["OPENAI_API_KEY"]
        body = json.dumps(
            {
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": "You mutate square-packing search programs."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc
        return payload["choices"][0]["message"]["content"]

    def _anthropic(self, prompt: str) -> str:
        key = os.environ["ANTHROPIC_API_KEY"]
        body = json.dumps(
            {
                "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc
        parts = payload.get("content") or []
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")

    def _gemini(self, prompt: str) -> str:
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}"
        )
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        return payload["candidates"][0]["content"]["parts"][0]["text"]

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [ln for ln in lines if not ln.startswith("```")]
            text = "\n".join(lines)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end < 0:
                raise SandboxError("LLM response was not JSON")
            data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise SandboxError("LLM JSON was not an object")
        return data


def make_mutator(prefer_llm: bool = True) -> tuple[Mutator, str]:
    """Factory: LLM if key present and prefer_llm, else stub."""
    provider, var = detect_llm_provider()
    if prefer_llm and provider is not None:
        return LLMMutator(provider=provider), f"llm:{provider} ({var})"
    return StubMutator(), "stub (no LLM API key)"
