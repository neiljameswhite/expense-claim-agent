"""
Pipeline — claiming pending work and processing it.

This is what n8n would call. Extracted from the CLI so the UI and the command
line share one code path rather than drifting apart.

The shape is deliberately the poller's shape even when a button triggers it:
a claim is written first and picked up separately, in its own transaction,
with the row claimed atomically. Nothing here assumes it was invoked by a
person rather than a schedule, so replacing the button with n8n changes the
trigger and nothing else.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Callable

from agent.db import connect, setting
from agent.policy import Policy
from agent.runner import RunOutcome, run_claim

# FOR UPDATE SKIP LOCKED is what makes this safe to run concurrently: two
# pollers, or a button press racing a scheduled run, cannot pick up the same
# claim. The row moves to 'processing' in the same statement that selects it.
CLAIM_ROWS = """
UPDATE claims
   SET status = 'processing', claimed_at = now()
 WHERE claim_id IN (
       SELECT claim_id FROM claims
        WHERE status = 'new' {filter}
        ORDER BY submitted_at
        {limit}
          FOR UPDATE SKIP LOCKED
       )
RETURNING *
"""

INSERT_RUN = """
INSERT INTO runs (
    run_id, claim_id, claim_status, current_stage, check_results,
    ai_verdict, ai_reason_detail, ai_confidence, ai_verdict_at,
    policy_version, model_version, autonomy_level, run_label
) VALUES (
    %(run_id)s, %(claim_id)s, 'awaiting_review', 'routed', %(check_results)s,
    %(ai_verdict)s, %(ai_reason_detail)s, %(ai_confidence)s, now(),
    %(policy_version)s, %(model_version)s, %(autonomy_level)s, %(run_label)s
)
"""

INSERT_EVENT = """
INSERT INTO run_events (run_id, sequence, stage, event_type, actor, detail_json)
VALUES (%(run_id)s, %(sequence)s, %(stage)s, %(event_type)s, %(actor)s, %(detail_json)s)
"""


@dataclass
class Processed:
    """One claim, processed."""

    claim_id: str
    record_id: str | None
    run_id: str | None
    outcome: RunOutcome | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def verdict(self) -> str:
        return self.outcome.verdict if self.outcome else "failed_system"


def pending_count() -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM claims WHERE status = 'new'"
        ).fetchone()["n"]


def _persist(conn, outcome: RunOutcome, claim_row: dict, policy_version: str) -> str:
    run_id = str(uuid.uuid4())

    conn.execute(
        INSERT_RUN,
        {
            "run_id": run_id,
            "claim_id": claim_row["claim_id"],
            "check_results": json.dumps(outcome.as_json()),
            "ai_verdict": outcome.verdict,
            "ai_reason_detail": outcome.narrative(),
            "ai_confidence": outcome.confidence(),
            "policy_version": policy_version,
            "model_version": setting("MODEL_VERSION", "claude-sonnet-5"),
            "autonomy_level": setting("AUTONOMY_LEVEL", "L0"),
            "run_label": setting("RUN_LABEL", "unlabelled"),
        },
    )

    seq = 1
    conn.execute(
        INSERT_EVENT,
        {
            "run_id": run_id,
            "sequence": seq,
            "stage": "received",
            "event_type": "run_created",
            "actor": "system",
            "detail_json": json.dumps({"claim_id": claim_row["claim_id"]}),
        },
    )
    for event in outcome.events:
        seq += 1
        actor = "model" if event["event_type"] == "check_completed" else "system"
        conn.execute(
            INSERT_EVENT,
            {
                "run_id": run_id,
                "sequence": seq,
                "stage": event["stage"],
                "event_type": event["event_type"],
                "actor": actor,
                "detail_json": json.dumps(event["detail_json"]),
            },
        )
    seq += 1
    conn.execute(
        INSERT_EVENT,
        {
            "run_id": run_id,
            "sequence": seq,
            "stage": "routed",
            "event_type": "suspended_for_review",
            "actor": "system",
            "detail_json": json.dumps({"verdict": outcome.verdict}),
        },
    )

    conn.execute(
        "UPDATE claims SET status = 'done' WHERE claim_id = %s", (claim_row["claim_id"],)
    )
    return run_id


def claim_pending(only: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """Take pending claims, marking them 'processing' atomically."""
    filter_sql = "AND record_id = ANY(%(ids)s)" if only else ""
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    sql = CLAIM_ROWS.format(filter=filter_sql, limit=limit_sql)

    with connect() as conn:
        rows = conn.execute(sql, {"ids": only} if only else {}).fetchall()
    return [dict(r) for r in rows]


def process_one(
    claim_row: dict,
    policy: Policy,
    *,
    on_event: Callable[[str, dict], None] | None = None,
) -> Processed:
    """Run the checks over one claim and persist the result.

    A failure is caught and recorded rather than raised: one bad claim must
    not abandon the rest of the batch, and the claim is returned to 'new' so
    the next pass picks it up.
    """
    record_id = claim_row.get("record_id")
    try:
        outcome = run_claim(claim_row, policy, on_event=on_event)
    except Exception as exc:  # noqa: BLE001 - deliberately broad
        with connect() as conn:
            conn.execute(
                "UPDATE claims SET status = 'new', claimed_at = NULL WHERE claim_id = %s",
                (claim_row["claim_id"],),
            )
        if on_event is not None:
            on_event("run_failed", {"claim_id": claim_row["claim_id"], "error": str(exc)})
        return Processed(
            claim_id=claim_row["claim_id"],
            record_id=record_id,
            run_id=None,
            outcome=None,
            error=str(exc),
        )

    with connect() as conn:
        run_id = _persist(conn, outcome, claim_row, policy.version)

    if on_event is not None:
        on_event("run_persisted", {"claim_id": claim_row["claim_id"], "run_id": run_id})

    return Processed(
        claim_id=claim_row["claim_id"],
        record_id=record_id,
        run_id=run_id,
        outcome=outcome,
    )


def process_pending(
    policy: Policy,
    *,
    only: list[str] | None = None,
    limit: int | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> list[Processed]:
    """Claim pending work and process all of it.

    on_progress is called with (done, total, claim_id) before each claim.
    on_event is passed through to the runner and fires for every check as it
    starts and completes, which is what a live log reads.

    Processing is sequential: each claim makes several model calls, and
    running them in parallel would make the traces harder to follow for no
    benefit at this scale.
    """
    rows = claim_pending(only=only, limit=limit)
    total = len(rows)
    results: list[Processed] = []

    for index, row in enumerate(rows, start=1):
        if on_progress is not None:
            on_progress(index, total, row["claim_id"])
        if on_event is not None:
            on_event(
                "claim_started",
                {
                    "claim_id": row["claim_id"],
                    "submitter": row.get("submitter"),
                    "amount": row.get("claim_amount"),
                    "currency": row.get("claim_currency"),
                    "category": row.get("claim_category"),
                    "index": index,
                    "total": total,
                },
            )
        results.append(process_one(row, policy, on_event=on_event))

    return results
