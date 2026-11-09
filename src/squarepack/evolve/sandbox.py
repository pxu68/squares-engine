"""Sandbox for evaluating / loading mutated search programs (section 9).

Generated code (if any) runs in a subprocess with CPU/memory limits, a wall
timeout, and no inherited API credentials. Hyperparameter-only programs are
JSON and evaluated in-process via ``fitness.evaluate_program`` after a schema
check — still under a subprocess wrapper for isolation when requested.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


class SandboxError(RuntimeError):
    pass


def _apply_limits(cpu_s: int, mem_mb: int) -> None:
    # Best-effort; some platforms ignore RLIMIT_AS.
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except (ValueError, resource.error):
        pass
    bytes_limit = mem_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
    except (ValueError, resource.error):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (bytes_limit, bytes_limit))
    except (ValueError, resource.error):
        pass


def run_python_sandboxed(
    source: str,
    *,
    cwd: Path,
    timeout_s: float = 30.0,
    cpu_s: int = 30,
    mem_mb: int = 512,
    env_extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute ``source`` in a child Python with resource limits.

    The child must print a single JSON object to stdout on success.
    Network credentials are stripped from the environment.
    """
    strip_prefixes = (
        "OPENAI_",
        "ANTHROPIC_",
        "GEMINI_",
        "GOOGLE_",
        "AZURE_",
        "AWS_",
        "SSH_",
        "GH_",
        "GITHUB_",
        "CURSOR_",
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(strip_prefixes) and k not in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(cwd / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)

    wrapper = textwrap.dedent(
        f"""
        import json, resource, sys
        def _limits():
            try:
                resource.setrlimit(resource.RLIMIT_CPU, ({cpu_s}, {cpu_s}))
            except Exception:
                pass
            b = {mem_mb} * 1024 * 1024
            for name in ("RLIMIT_AS", "RLIMIT_DATA"):
                try:
                    resource.setrlimit(getattr(resource, name), (b, b))
                except Exception:
                    pass
        _limits()
        """
    ) + "\n" + source

    with tempfile.TemporaryDirectory(prefix="sqpack_sbx_") as td:
        script = Path(td) / "job.py"
        script.write_text(wrapper, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"sandbox timeout after {timeout_s}s") from exc

    if proc.returncode != 0:
        raise SandboxError(
            f"sandbox exit {proc.returncode}: {proc.stderr[-2000:]}"
        )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise SandboxError(f"sandbox non-JSON stdout: {proc.stdout[-1000:]}") from exc


def validate_program_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Reject obviously unsafe / malformed program payloads."""
    if not isinstance(data, dict):
        raise SandboxError("program must be a JSON object")
    allowed = {
        "name",
        "eta0",
        "temperature",
        "cool_factor",
        "seeded_expand",
        "max_lbfgs_iter",
        "energy_feasible",
        "move_probs",
        "parent",
        "method",
        "generation",
        "notes",
        "move_names",
    }
    unknown = set(data) - allowed
    if unknown:
        raise SandboxError(f"unknown program keys: {sorted(unknown)}")
    for key in ("eta0", "temperature", "cool_factor", "seeded_expand", "energy_feasible"):
        if key in data and not isinstance(data[key], (int, float)):
            raise SandboxError(f"{key} must be numeric")
    if "move_probs" in data:
        if not isinstance(data["move_probs"], list) or not data["move_probs"]:
            raise SandboxError("move_probs must be a non-empty list")
        if any(not isinstance(x, (int, float)) or x < 0 for x in data["move_probs"]):
            raise SandboxError("move_probs must be non-negative numbers")
    if "max_lbfgs_iter" in data:
        if not isinstance(data["max_lbfgs_iter"], int) or not (1 <= data["max_lbfgs_iter"] <= 500):
            raise SandboxError("max_lbfgs_iter out of range")
    return data
