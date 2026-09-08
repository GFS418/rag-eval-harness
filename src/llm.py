"""Claude API access shared by the generator and the judge.

* One ``call()`` path for interactive use (the Streamlit app, smoke tests) and
  one ``run_batch()`` path for evaluation runs at 50% cost via the Batches API.
* Every response is cached on disk keyed by a hash of (model, system, user
  prompt, schema, effort), so re-running an evaluation never re-pays for a
  request and results are reproducible from the cache alone.
* Usage (tokens) and latency are recorded with each response for the
  cost/latency tables.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODELS = {
    "sonnet-5": "claude-sonnet-5",
    "haiku-4.5": "claude-haiku-4-5",
    "opus-5": "claude-opus-5",
}
# Haiku 4.5 rejects output_config.effort; the others accept it.
SUPPORTS_EFFORT = {"claude-sonnet-5", "claude-opus-5"}

# List prices per million tokens (input, output) as of 2026-06, used only to
# *estimate* spend in reports; the README states the date.
PRICE_PER_MTOK = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
}

CACHE_PATH = Path("data/cache/llm.sqlite")


@dataclass(frozen=True)
class Request:
    custom_id: str
    model: str            # key in MODELS
    system: str
    user: str
    schema: dict[str, Any]
    effort: str | None = "medium"
    max_tokens: int = 4096

    @property
    def model_id(self) -> str:
        return MODELS[self.model]

    def cache_key(self) -> str:
        blob = json.dumps([self.model_id, self.system, self.user, self.schema,
                           self.effort if self.model_id in SUPPORTS_EFFORT else None],
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def params(self) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": self.user}],
            "output_config": {"format": {"type": "json_schema", "schema": self.schema}},
        }
        if self.effort and self.model_id in SUPPORTS_EFFORT:
            p["output_config"]["effort"] = self.effort
        return p


@dataclass
class Response:
    custom_id: str
    data: dict[str, Any] | None     # parsed JSON, None on failure
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_s: float | None         # None for batch results
    error: str | None = None
    model_id: str = ""

    @property
    def cost_usd(self) -> float:
        pi, po = PRICE_PER_MTOK.get(self.model_id, (0.0, 0.0))
        return (self.input_tokens * pi + self.output_tokens * po) / 1e6


class Cache:
    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, custom_id TEXT, "
                          "model_id TEXT, response TEXT, created REAL)")
        self.conn.commit()

    def get(self, req: Request) -> Response | None:
        row = self.conn.execute("SELECT response FROM responses WHERE key=?", (req.cache_key(),)).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        return Response(**d)

    def put(self, req: Request, resp: Response) -> None:
        self.conn.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
                          (req.cache_key(), req.custom_id, req.model_id, json.dumps(resp.__dict__), time.time()))
        self.conn.commit()


def _parse_message(custom_id: str, msg: Any, latency: float | None) -> Response:
    stop = msg.stop_reason
    text = next((b.text for b in msg.content if b.type == "text"), "")
    data, err = None, None
    if stop == "refusal":
        err = "refusal"
    elif stop == "max_tokens":
        err = "max_tokens"
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            err = f"json: {e}"
    return Response(custom_id, data, stop, msg.usage.input_tokens, msg.usage.output_tokens,
                    latency, err, msg.model)


def _client():
    import anthropic

    return anthropic.Anthropic()


def call(req: Request, cache: Cache | None = None) -> Response:
    """Synchronous single request, cached."""
    cache = cache or Cache()
    hit = cache.get(req)
    if hit is not None:
        return hit
    t0 = time.time()
    msg = _client().messages.create(**req.params())
    resp = _parse_message(req.custom_id, msg, time.time() - t0)
    cache.put(req, resp)
    return resp


def run_batch(reqs: list[Request], cache: Cache | None = None, poll_s: int = 30,
              log=print) -> dict[str, Response]:
    """Submit uncached requests as one Message Batch, poll, cache, return all."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as BatchRequest

    cache = cache or Cache()
    out: dict[str, Response] = {}
    todo: list[Request] = []
    for r in reqs:
        hit = cache.get(r)
        if hit is not None:
            out[r.custom_id] = hit
        else:
            todo.append(r)
    log(f"batch: {len(out)} cached, {len(todo)} to submit")
    if not todo:
        return out
    by_id = {r.custom_id: r for r in todo}
    client = _client()
    batch = client.messages.batches.create(requests=[
        BatchRequest(custom_id=r.custom_id, params=MessageCreateParamsNonStreaming(**r.params()))
        for r in todo])
    log(f"batch {batch.id} submitted")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        log(f"  {b.processing_status}: {b.request_counts.processing} processing, "
            f"{b.request_counts.succeeded} done")
        time.sleep(poll_s)
    for res in client.messages.batches.results(batch.id):
        req = by_id[res.custom_id]
        if res.result.type == "succeeded":
            resp = _parse_message(res.custom_id, res.result.message, None)
        else:
            detail = getattr(getattr(res.result, "error", None), "type", res.result.type)
            resp = Response(res.custom_id, None, res.result.type, 0, 0, None, f"batch:{detail}", req.model_id)
        cache.put(req, resp)
        out[res.custom_id] = resp
    return out
