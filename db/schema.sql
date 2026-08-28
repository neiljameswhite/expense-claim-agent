-- Expense Claim Agent — schema v0.4
-- Postgres 16 / Neon

-- ---------------------------------------------------------------
-- claims: the submission store. n8n polls this table.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS claims (
    claim_id                    text PRIMARY KEY,
    record_id                   text,                       -- corpus record this came from
    status                      text NOT NULL DEFAULT 'new' -- new | processing | done
                                CHECK (status IN ('new','processing','done')),
    submitter                   text NOT NULL,
    claim_amount                numeric(10,2) NOT NULL,
    claim_currency              text NOT NULL DEFAULT 'GBP',
    claim_category              text NOT NULL,
    claim_date                  date NOT NULL,
    business_purpose            text NOT NULL,
    tax_amount                  numeric(10,2),
    cost_exception_rationale    text,
    other_category_rationale    text,
    receipt_ref                 text,                       -- pointer; unused while extraction is stubbed
    extraction                  jsonb NOT NULL,             -- what the extractor would have returned
    submitted_at                timestamptz NOT NULL DEFAULT now(),
    claimed_at                  timestamptz                 -- when n8n took the row
);

CREATE INDEX IF NOT EXISTS idx_claims_status ON claims (status);

-- ---------------------------------------------------------------
-- runs: one mutable row per processing run. Holds current state.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id                  uuid PRIMARY KEY,
    claim_id                text NOT NULL REFERENCES claims (claim_id),
    claim_status            text NOT NULL DEFAULT 'awaiting_review'
                            CHECK (claim_status IN
                                ('awaiting_review','approved','declined','failed_system')),
    current_stage           text,

    check_results           jsonb,      -- [{check_id, result, inputs}]

    ai_verdict              text CHECK (ai_verdict IN ('approve','decline')),
    ai_reason_detail        text,
    ai_confidence           numeric(4,3),
    ai_verdict_at           timestamptz,

    human_verdict           text CHECK (human_verdict IN ('approve','decline')),
    human_reason_detail     text,
    reviewer_id             text,
    decided_at              timestamptz,
    detail_opened           boolean NOT NULL DEFAULT false,

    final_reason_detail     text,
    reason_overwritten      boolean GENERATED ALWAYS AS
                            (final_reason_detail IS DISTINCT FROM ai_reason_detail) STORED,
    agreement               text GENERATED ALWAYS AS
                            (CASE
                                WHEN human_verdict IS NULL THEN NULL
                                WHEN human_verdict = ai_verdict THEN 'agreed'
                                ELSE 'overturned'
                             END) STORED,

    policy_version          text NOT NULL,
    model_version           text NOT NULL,
    autonomy_level          text NOT NULL DEFAULT 'L0',
    run_label               text NOT NULL,
    trace_url               text,
    state_json              jsonb,

    created_at              timestamptz NOT NULL DEFAULT now(),
    closed_at               timestamptz,

    -- Invariant 6: the AI's position precedes the human's
    CONSTRAINT ai_verdict_precedes_decision
        CHECK (decided_at IS NULL OR ai_verdict_at IS NULL OR ai_verdict_at < decided_at),

    -- Invariant 7: a decline requires the detail view to have been opened
    CONSTRAINT decline_requires_detail
        CHECK (human_verdict IS DISTINCT FROM 'decline' OR detail_opened = true)
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (claim_status);
CREATE INDEX IF NOT EXISTS idx_runs_label  ON runs (run_label);

-- Invariant 8: an overturn must record a reason detail.
-- Enforced as a trigger because it references a generated column.
CREATE OR REPLACE FUNCTION enforce_overturn_reason() RETURNS trigger AS $$
BEGIN
    IF NEW.human_verdict IS NOT NULL
       AND NEW.human_verdict IS DISTINCT FROM NEW.ai_verdict
       AND (NEW.human_reason_detail IS NULL OR btrim(NEW.human_reason_detail) = '') THEN
        RAISE EXCEPTION 'An overturned decision requires human_reason_detail';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_overturn_reason ON runs;
CREATE TRIGGER trg_overturn_reason
    BEFORE INSERT OR UPDATE ON runs
    FOR EACH ROW EXECUTE FUNCTION enforce_overturn_reason();

-- ---------------------------------------------------------------
-- run_events: append-only audit record. Never updated or deleted.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_events (
    event_id    bigserial PRIMARY KEY,
    run_id      uuid NOT NULL REFERENCES runs (run_id),
    sequence    integer NOT NULL,
    stage       text,
    event_type  text NOT NULL,
    actor       text NOT NULL CHECK (actor IN ('system','model','reviewer')),
    entered_at  timestamptz NOT NULL DEFAULT now(),
    detail_json jsonb,
    UNIQUE (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_events_run ON run_events (run_id, sequence);

-- ---------------------------------------------------------------
-- payment_requests: the stub endpoint records here. Idempotent on run_id.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payment_requests (
    run_id       uuid PRIMARY KEY REFERENCES runs (run_id),
    claim_id     text NOT NULL,
    amount       numeric(10,2) NOT NULL,
    currency     text NOT NULL,
    payload      jsonb NOT NULL,
    received_at  timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------
-- test_results: eval outcomes per run label.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS test_results (
    result_id        bigserial PRIMARY KEY,
    run_label        text NOT NULL,
    record_id        text NOT NULL,
    run_id           uuid,
    trial_number     integer NOT NULL DEFAULT 1,
    expected_verdict text,
    actual_verdict   text,
    expected_checks  jsonb,
    actual_checks    jsonb,
    passed           boolean NOT NULL,
    failure_detail   text,
    evaluated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_results_label ON test_results (run_label);

-- ---------------------------------------------------------------
-- Convenience view: the review queue
-- ---------------------------------------------------------------
CREATE OR REPLACE VIEW review_queue AS
SELECT r.run_id,
       r.claim_id,
       c.submitter,
       c.claim_amount,
       c.claim_currency,
       c.claim_category,
       c.claim_date,
       c.business_purpose,
       r.ai_verdict,
       r.ai_confidence,
       r.check_results,
       r.created_at
FROM runs r
JOIN claims c ON c.claim_id = r.claim_id
WHERE r.claim_status = 'awaiting_review';
