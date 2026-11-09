"""Redis Streams worker for durable, retryable file ingestion."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import uuid
from typing import cast

from redis.asyncio import Redis

from agent.config import get_settings
from agent.ingestion.jobs import process_input_job

STREAM = "ingestion:stream"
GROUP = "ingestion:workers"
MAX_ATTEMPTS = 3
logger = logging.getLogger(__name__)


async def _handle(redis: Redis, message_id: str, fields: dict[str, str]) -> None:
    payload = json.loads(fields["payload"])
    job_key = f"ingestion:job:{payload['job_id']}"
    state = await redis.hget(job_key, "status")
    if state in {"completed", "failed"}:
        await redis.xack(STREAM, GROUP, message_id)
        await redis.xdel(STREAM, message_id)
        return

    attempt = int(payload.get("attempt", 1))
    await redis.hset(job_key, mapping={"status": "running", "attempts": str(attempt)})
    try:
        result = await asyncio.to_thread(process_input_job, payload)
    except Exception as exc:
        logger.exception("Ingestion job %s failed on attempt %s", payload["job_id"], attempt)
        safe_error = f"Ingestion failed ({type(exc).__name__})"
        if attempt < MAX_ATTEMPTS:
            next_attempt = attempt + 1
            await redis.hset(
                job_key,
                mapping={"status": "queued", "error": safe_error},
            )
            await asyncio.sleep(min(2**attempt, 10))
            payload["attempt"] = next_attempt
            await redis.xadd(STREAM, {"payload": json.dumps(payload)})
        else:
            await redis.hset(
                job_key,
                mapping={"status": "failed", "error": safe_error},
            )
    else:
        await redis.hset(
            job_key,
            mapping={"status": "completed", "result": json.dumps(result), "error": ""},
        )
    finally:
        await redis.xack(STREAM, GROUP, message_id)
        await redis.xdel(STREAM, message_id)


async def run_worker() -> None:
    settings = get_settings()
    redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=10,
    )
    consumer = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    try:
        await redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise

    # Recover messages left pending if a worker process stopped mid-job.
    cursor = "0-0"
    while True:
        claim_result = cast(
            tuple[str, list[tuple[str, dict[str, str]]], list[str]],
            await redis.xautoclaim(
                STREAM, GROUP, consumer, min_idle_time=60_000, start_id=cursor, count=25
            ),
        )
        next_cursor, abandoned, _ = claim_result
        for message_id, fields in abandoned:
            await _handle(redis, message_id, fields)
        if next_cursor == "0-0":
            break
        cursor = next_cursor

    try:
        while True:
            batches = cast(
                list[tuple[str, list[tuple[str, dict[str, str]]]]],
                await redis.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=1, block=5000),
            )
            for _, messages in batches:
                for message_id, fields in messages:
                    await _handle(redis, message_id, fields)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(run_worker())
