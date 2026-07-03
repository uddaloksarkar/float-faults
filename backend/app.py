"""Sandboxed execution API for the NumPy Sampler Fault Lab page.

Runs each request's Python snippet as a fresh subprocess (sandbox_runner.py)
so no state leaks between requests even within a warm, reused Cloud Run
instance. The container itself should be deployed with
--execution-environment=gen1 (gVisor) for OS-level isolation; this file adds
the next layer down: per-request process isolation, resource limits, a hard
timeout ceiling, and basic rate limiting.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

RUNNER = Path(__file__).parent / "sandbox_runner.py"
MAX_TIMEOUT_S = 30       # hard ceiling regardless of what the client requests
MAX_CODE_LEN = 20_000    # reject absurdly large payloads before spawning anything

ALLOWED_ORIGINS = [
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    # TODO: add the GitHub Pages origin once the site is published, e.g.
    # "https://<username>.github.io",
]


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=get_client_ip)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    code: str
    timeout_ms: int = 15000


@app.get("/health")
def health():
    return {"version": np.__version__}


@app.post("/run")
@limiter.limit("20/minute")
def run(request: Request, body: RunRequest):
    if len(body.code) > MAX_CODE_LEN:
        return {"out": f"ERROR - snippet too large ({len(body.code)} chars, max {MAX_CODE_LEN})\nverdict: BUG (rejected)"}

    timeout_s = min(max(body.timeout_ms / 1000, 1), MAX_TIMEOUT_S)
    try:
        proc = subprocess.run(
            [sys.executable, str(RUNNER)],
            input=body.code,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {"out": proc.stdout}
    except subprocess.TimeoutExpired:
        return {"out": f"HANG - no result in {timeout_s:g}s\nverdict: BUG (non-termination)"}
