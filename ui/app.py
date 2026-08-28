"""
Expense Claim Agent — the one interface the system has.

Four tabs:

  Submit     the corpus as a table; releasing a claim writes it with status
             'new', which is what a real submission would do
  Review     the queue, one decision at a time, with the evidence behind it
  Completed  decided claims, with agreement and calibration below
  Policy     the expense policy in full, so a reviewer can verify any clause
             the system cited rather than trusting the excerpt

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.assessment import CHECK_NAMES  # noqa: E402,F401
from agent.db import connect, setting  # noqa: E402
from agent.pipeline import pending_count as unprocessed_count  # noqa: E402
from agent.pipeline import process_pending  # noqa: E402
from agent.policy import load as load_policy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "corpus_v1.json"
POLICY_FILE = ROOT / "corpus" / "expense_policy_v1.md"
RECEIPTS = Path(__file__).resolve().parent / "assets" / "receipts"
FALLBACK_RECEIPT = Path(__file__).resolve().parent / "assets" / "demo_receipt.png"

# Placement of the AI verdict in the detail view. Leading with a conclusion
# is faster but means agreement partly measures compliance rather than
# independent judgement. Flip this to compare overturn rates either way.
VERDICT_FIRST = setting("SHOW_VERDICT_FIRST", "true").lower() == "true"

# Checks 1-3 are deterministic; their detail is code-generated, not model
# output. Marked in the rationale column so the origin is never implied.
DETERMINISTIC_CHECKS = {1, 2, 3}

RESULT_ICON = {
    "pass": "✅",
    "fail": "❌",
    "inconclusive": "⚠️",
    "not_applicable": "–",
    "not_evaluated": "–",
}

st.set_page_config(page_title="Prototype Expense Claim Agent", page_icon="📋", layout="wide")

# Blue rather than Streamlit's default red: an empty required field is not
# an error state, and red styling reads as one.
st.markdown(
    """
    <style>
      div[data-testid="stTextInput"] input:focus,
      div[data-testid="stTextArea"] textarea:focus {
          border-color: #1F5FA8 !important;
          box-shadow: 0 0 0 1px #1F5FA8 !important;
      }
      div[data-testid="stTextInput"] input,
      div[data-testid="stTextArea"] textarea {
          border-color: #C3CEDC !important;
      }
      button[kind="primary"] {
          background-color: #E8F0FA !important;
          color: #143F73 !important;
          border: 1px solid #1F5FA8 !important;
          font-weight: 600 !important;
      }
      button[kind="primary"]:hover:enabled {
          background-color: #1F5FA8 !important;
          color: #FFFFFF !important;
      }
      button[kind="primary"]:disabled {
          background-color: #F2F4F6 !important;
          color: #9AA5B1 !important;
          border: 1px solid #D8DEE6 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@st.cache_data
def load_corpus() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))["records"]


@st.cache_data
def policy_text() -> str:
    return POLICY_FILE.read_text(encoding="utf-8")


@st.cache_data
def readable_policy() -> str:
    """The policy with markdown syntax stripped, for plain-text display.

    Removes heading hashes, bold markers and the table separator rows, so
    the document reads as a document rather than as source. Nothing is
    reworded — this is the same text the checks are assessed against.
    """
    out: list[str] = []
    for raw in policy_text().splitlines():
        line = raw.rstrip()

        # table separator rows: |---|---|
        if line.startswith("|") and set(line) <= set("|-: "):
            continue

        # headings
        if line.startswith("#"):
            line = line.lstrip("#").strip()
            if line:
                out.append("")
                out.append(line.upper())
                out.append("-" * len(line))
            continue

        # horizontal rules
        if line.strip() in {"---", "***", "___"}:
            out.append("")
            continue

        line = line.replace("**", "").replace("*", "")

        # table rows: render as aligned columns
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            line = "  " + "".join(c.ljust(24) for c in cells).rstrip()

        out.append(line)

    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


@st.cache_resource
def policy_object():
    return load_policy(POLICY_FILE)


@st.cache_data
def reference_map() -> dict[str, str]:
    """Record id to the reference shown in the interface.

    The corpus keeps its own ids (A1, F2, H1) because the CLI, the eval suite
    and every expected_checks block key off them. EXP-n is a display label
    only, so renumbering here changes nothing downstream.
    """
    return {r["record_id"]: r.get("reference", r["record_id"]) for r in load_corpus()}


def submitted_counts() -> dict[str, int]:
    """How many times each corpus record has been submitted."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT record_id, count(*) AS n FROM claims GROUP BY record_id"
        ).fetchall()
    return {r["record_id"]: r["n"] for r in rows if r["record_id"]}


def pending_record_ids() -> set[str]:
    """Records already written and awaiting assessment.

    Submitting one of these again would create a second claim for the same
    test case, and processing would then pick up both. A record that is
    already waiting does not need submitting; it needs assessing.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT record_id FROM claims WHERE status IN ('new', 'processing')"
        ).fetchall()
    return {r["record_id"] for r in rows if r["record_id"]}


def submit_claims(records: list[dict]) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from seed import seed  # noqa: E402

    return seed(only=[r["record_id"] for r in records])


def run_processing(only: list[str] | None = None) -> list:
    """Claim pending work and process it, logging each step as it happens.

    This is what n8n would do on a timer. The mechanism is identical — rows
    are claimed atomically from 'new' — so replacing this with a scheduled
    workflow later changes the trigger and nothing else.

    The console exists because the architecture is otherwise invisible: it
    shows three checks resolving instantly as deterministic code, four
    pausing while they call the model with a named slice of policy, and the
    verdict computed at the end with no model call at all.
    """
    policy = policy_object()
    lines: list[str] = []
    console = st.empty()
    bar = st.progress(0.0, text="Starting…")

    def draw() -> None:
        console.code("\n".join(lines[-400:]), language="text")

    def on_event(kind: str, detail: dict) -> None:
        if kind == "claim_started":
            if lines:
                lines.append("")
            lines.append(
                f"[{detail['index']}/{detail['total']}] {detail['claim_id']}  "
                f"{detail.get('submitter') or ''} · "
                f"{detail.get('currency') or ''} {detail.get('amount')} · "
                f"{detail.get('category') or ''}"
            )
        elif kind == "check_started":
            source = "claude" if detail["uses_model"] else "code"
            sections = detail.get("policy_sections") or []
            slice_text = (
                "  policy §" + ", §".join(sections) if sections else ""
            )
            lines.append(
                f"   →  {detail['check_id']}. {detail['name']:<32} {source}{slice_text}"
            )
        elif kind == "check_completed":
            tokens = detail.get("input_tokens", 0) + detail.get("output_tokens", 0)
            cost = f"  {tokens:,} tokens" if tokens else ""
            refs = detail.get("clause_refs") or []
            cited = "  cites " + ", ".join(refs) if refs else ""
            symbol = {"pass": "✓", "fail": "✗", "inconclusive": "?"}.get(
                detail["result"], "·"
            )
            lines[-1] = (
                f"   {symbol}  {detail['check_id']}. {detail['name']:<32} "
                f"{detail['result']}{cited}{cost}"
            )
        elif kind == "ai_verdict_written":
            conf = detail.get("confidence")
            conf_text = f" · confidence {conf:.2f}" if conf is not None else ""
            lines.append(
                f"   ⇒  verdict: {detail['verdict']}{conf_text} · "
                f"{detail.get('tokens', 0):,} tokens"
            )
        elif kind == "run_failed":
            lines.append(f"   !  failed: {detail['error']}")
        draw()

    def progress(done: int, total: int, claim_id: str) -> None:
        bar.progress(
            (done - 1) / max(total, 1),
            text=f"Assessing {claim_id}  ({done} of {total})",
        )

    results = process_pending(policy, only=only, on_progress=progress, on_event=on_event)
    bar.progress(1.0, text=f"Assessed {len(results)} claim(s)")

    tokens = sum(r.outcome.tokens() for r in results if r.ok and r.outcome)
    if tokens:
        lines.append("")
        lines.append(f"{len(results)} claim(s) · {tokens:,} tokens total")
        draw()

    return results


def report_processing(results: list) -> None:
    if not results:
        st.info("Nothing pending to assess.")
        return

    failed = [r for r in results if not r.ok]
    approved = sum(1 for r in results if r.ok and r.verdict == "approve")
    declined = sum(1 for r in results if r.ok and r.verdict == "decline")

    st.success(
        f"{len(results) - len(failed)} claim(s) assessed — "
        f"{approved} approve, {declined} decline. Now awaiting review."
    )
    for r in failed:
        st.error(f"{r.claim_id} failed: {r.error}. Returned to pending.")


def queue_rows(order: str) -> list[dict]:
    clause = {
        "Oldest first": "r.created_at ASC",
        "Newest first": "r.created_at DESC",
        "Highest value": "c.claim_amount DESC",
        "Lowest value": "c.claim_amount ASC",
    }[order]
    with connect() as conn:
        return conn.execute(
            f"""
            SELECT r.run_id, r.claim_id, r.ai_verdict, r.ai_reason_detail,
                   r.ai_confidence, r.check_results, r.created_at,
                   r.policy_version, r.model_version, r.run_label,
                   c.submitter, c.claim_amount, c.claim_currency,
                   c.claim_category, c.claim_date, c.business_purpose,
                   c.tax_amount, c.cost_exception_rationale,
                   c.other_category_rationale, c.extraction, c.record_id, c.submitted_at
              FROM runs r
              JOIN claims c ON c.claim_id = r.claim_id
             WHERE r.claim_status = 'awaiting_review'
             ORDER BY {clause}
            """
        ).fetchall()


def mark_opened(run_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE runs SET detail_opened = true WHERE run_id = %s", (run_id,))


def record_decision(
    run_id: str, verdict: str, overturn_rationale: str, reviewer: str, response: str
) -> None:
    status = "approved" if verdict == "approve" else "declined"
    with connect() as conn:
        conn.execute(
            """
            UPDATE runs
               SET human_verdict = %(verdict)s,
                   human_reason_detail = %(rationale)s,
                   final_reason_detail = %(response)s,
                   reviewer_id = %(reviewer)s,
                   decided_at = now(),
                   claim_status = %(status)s,
                   closed_at = now(),
                   current_stage = NULL
             WHERE run_id = %(run_id)s
            """,
            {
                "run_id": run_id,
                "verdict": verdict,
                "rationale": overturn_rationale or None,
                "response": response,
                "reviewer": reviewer,
                "status": status,
            },
        )
        seq = conn.execute(
            "SELECT coalesce(max(sequence), 0) + 1 AS n FROM run_events WHERE run_id = %s",
            (run_id,),
        ).fetchone()["n"]
        conn.execute(
            """
            INSERT INTO run_events (run_id, sequence, stage, event_type, actor, detail_json)
            VALUES (%s, %s, NULL, 'decision_recorded', 'reviewer', %s)
            """,
            (run_id, seq, json.dumps({"verdict": verdict, "reviewer": reviewer})),
        )


def decided_rows() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT r.claim_id, r.ai_verdict, r.human_verdict, r.agreement,
                   r.detail_opened, r.ai_confidence, r.reviewer_id,
                   r.human_reason_detail, r.final_reason_detail,
                   r.reason_overwritten, r.decided_at, r.run_label,
                   c.claim_amount, c.claim_category, c.submitter, c.record_id
              FROM runs r
              JOIN claims c ON c.claim_id = r.claim_id
             WHERE r.human_verdict IS NOT NULL
             ORDER BY r.decided_at DESC
            """
        ).fetchall()


def pending_count() -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM runs WHERE claim_status = 'awaiting_review'"
        ).fetchone()["n"]


def completed_count() -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM runs WHERE human_verdict IS NOT NULL"
        ).fetchone()["n"]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def receipt_for(record_id: str | None) -> Path | None:
    """The receipt image for a claim.

    Generated from the record's extraction block by scripts/make_receipts.py,
    so the image and the data the checks read cannot drift apart.
    """
    if record_id:
        path = RECEIPTS / f"{record_id}.png"
        if path.exists():
            return path
    return FALLBACK_RECEIPT if FALLBACK_RECEIPT.exists() else None


def as_dict(value):
    return json.loads(value) if isinstance(value, str) else (value or {})


def as_list(value):
    return json.loads(value) if isinstance(value, str) else (value or [])


def render_verdict(row) -> None:
    verdict = row["ai_verdict"]
    conf = row["ai_confidence"]
    conf_text = f" · confidence {float(conf):.2f}" if conf is not None else ""

    st.markdown("**AI Review**")
    if verdict == "approve":
        st.success(f"**Approve**{conf_text}")
    else:
        st.error(f"**Decline**{conf_text}")
    if row["ai_reason_detail"]:
        st.markdown(row["ai_reason_detail"])


def render_cited_clauses(check_results) -> None:
    """Verbatim text of every clause the checks cited.

    The reviewer is otherwise reading policy filtered through the model. A
    clause cited but absent from the loaded policy is called out rather than
    silently omitted — that absence is the finding.
    """
    refs: list[str] = []
    for c in check_results:
        for ref in c.get("clause_refs") or []:
            base = str(ref).split("(")[0].strip()
            if base and base not in refs:
                refs.append(base)

    if not refs:
        return

    policy = policy_object()

    def sort_key(r: str):
        parts = [p for p in r.split(".") if p.isdigit()]
        return [int(p) for p in parts] or [99]

    st.markdown("**Policy clauses cited — verbatim**")
    for ref in sorted(refs, key=sort_key):
        try:
            clause = policy.clause(ref)
        except KeyError:
            st.warning(
                f"Clause {ref} was cited but does not appear in policy v{policy.version}."
            )
            continue
        st.markdown(f"**{clause.ref}**  {clause.text}")


def render_evidence(row) -> None:
    extraction = as_dict(row["extraction"])
    left, right = st.columns(2)

    with left:
        st.markdown("**Claim as submitted**")
        st.write(
            pd.DataFrame(
                [
                    ("Submitter", row["submitter"]),
                    ("Amount", f"{row['claim_currency']} {row['claim_amount']}"),
                    ("Category", row["claim_category"]),
                    ("Date incurred", str(row["claim_date"])),
                    ("Business purpose", row["business_purpose"]),
                    ("Tax declared", row["tax_amount"] if row["tax_amount"] is not None else "—"),
                ],
                columns=["Field", "Value"],
            ).set_index("Field")
        )

        for label, key in (
            ("Cost Exception — User Submitted Rationale", "cost_exception_rationale"),
            ("Other Category — User Submitted Rationale", "other_category_rationale"),
        ):
            if row[key]:
                st.markdown(f"**{label}**")
                st.info(row[key])

    with right:
        st.markdown("**Receipt**")
        image = receipt_for(row.get("record_id"))
        if image is not None:
            st.image(str(image), width=320)
        else:
            st.caption("No receipt image. Run scripts/make_receipts.py.")

        st.markdown("**Read from receipt — extracted**")
        conf = extraction.get("confidence") or {}
        rows = []
        for field in ("retailer", "date", "total", "vat_number", "vat_amount"):
            value = extraction.get(field)
            c = conf.get(field)
            suffix = f"  ({c})" if c is not None else ""
            rows.append((field, f"{value if value is not None else '—'}{suffix}"))
        st.write(pd.DataFrame(rows, columns=["Field", "Value"]).set_index("Field"))

        items = extraction.get("line_items") or []
        if items:
            st.write(pd.DataFrame(items))

    results = as_list(row["check_results"])

    st.markdown("**Checks**")
    table = []
    for c in sorted(results, key=lambda x: x["check_id"]):
        rationale = c.get("detail", "")
        if c["check_id"] in DETERMINISTIC_CHECKS and rationale:
            rationale = f"[deterministic] {rationale}"
        table.append(
            {
                "": RESULT_ICON.get(c["result"], "?"),
                "Check": f"{c['check_id']}. {c['name']}",
                "Result": c["result"],
                "Policy Clause": ", ".join(c.get("clause_refs") or []),
                "AI Rationale": rationale,
            }
        )
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption(
        "Checks 1–3 are deterministic and marked accordingly. "
        f"Assessed against policy v{row['policy_version']} · {row['model_version']}"
    )

    render_cited_clauses(results)


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------


def render_submission_form(claim: dict, record_id: str) -> None:
    """The claim as the submitter filled it in.

    Rendered as a form rather than a table because this is the only place the
    submission interface exists in this build, and because an empty field is
    itself the thing under test in several records — a table row cannot show
    an absence, a blank form field can.
    """
    st.markdown("**Expense claim form** · as submitted")

    left, right = st.columns(2)
    with left:
        st.text_input("Submitter", value=claim["submitter"], disabled=True,
                      key=f"f_sub_{id(claim)}")
        st.text_input("Amount", value=f"{claim['claim_currency']} {claim['claim_amount']:.2f}",
                      disabled=True, key=f"f_amt_{id(claim)}")
        st.text_input("Date incurred", value=str(claim["claim_date"]), disabled=True,
                      key=f"f_date_{id(claim)}")
    with right:
        st.text_input("Category", value=claim["claim_category"], disabled=True,
                      key=f"f_cat_{id(claim)}")
        tax = claim.get("tax_amount")
        st.text_input(
            "Tax amount",
            value=f"{tax:.2f}" if tax is not None else "",
            disabled=True,
            key=f"f_tax_{id(claim)}",
        )
        st.text_input("Attachment", value="receipt.png", disabled=True,
                      key=f"f_att_{id(claim)}")

    st.text_area("Business purpose", value=claim["business_purpose"], disabled=True,
                 height=70, key=f"f_purp_{id(claim)}")

    st.text_area(
        "Please give a rationale if cost exceeds policy",
        value=claim.get("cost_exception_rationale") or "",
        disabled=True,
        height=90,
        key=f"f_cost_{id(claim)}",
    )
    st.text_area(
        "If Other selected for reason, please enter reason",
        value=claim.get("other_category_rationale") or "",
        disabled=True,
        height=90,
        key=f"f_other_{id(claim)}",
    )

    image = receipt_for(record_id)
    if image is not None:
        st.markdown("**Attached receipt**")
        st.image(str(image), width=300)


def tab_submit() -> None:
    st.subheader("Test Claims")

    records = load_corpus()
    counts = submitted_counts()
    awaiting = pending_record_ids()

    # Checkbox keys carry a nonce. Deleting a widget's key does not reliably
    # reset it — Streamlit restores the value from the incoming frontend
    # state when the widget is recreated in the same run. Bumping the nonce
    # sidesteps that: the checkboxes become new widgets with no history, so
    # they take their default of unchecked.
    nonce = st.session_state.get("pick_nonce", 0)

    def pick_key(record_id: str) -> str:
        return f"pick_{nonce}_{record_id}"

    head, tail = st.columns([1, 5])
    with head:
        if st.button("Select all"):
            for r in records:
                st.session_state[pick_key(r["record_id"])] = True
            st.rerun()
    with tail:
        if st.button("Clear selection"):
            st.session_state["pick_nonce"] = nonce + 1
            st.rerun()

    st.divider()

    for rec in records:
        claim = rec["claim"]
        rid = rec["record_id"]
        # This tab selects test cases, not claims: the header says what each
        # one tests rather than describing someone's dinner.
        label = rec.get("label") or rec["purpose"].rstrip(".")
        reference = rec.get("reference", rid)
        state = "  ·  awaiting assessment" if rid in awaiting else ""

        pick, body = st.columns([1, 14])
        with pick:
            st.checkbox("", key=pick_key(rid), label_visibility="collapsed")
        with body:
            header = (
                f"{reference}  ·  {label}  ·  £{claim['claim_amount']:.2f}  ·  "
                f"Expected Result: {rec['expected_verdict'].upper()}{state}"
            )
            with st.expander(header):
                st.caption(rec["purpose"])
                render_submission_form(claim, rid)

    chosen = [r["record_id"] for r in records if st.session_state.get(pick_key(r["record_id"]))]

    st.divider()
    if st.button(f"Submit and assess ({len(chosen)})", disabled=not chosen, type="primary"):
        # A record already awaiting assessment is not submitted again. Doing
        # so would leave two claims for one test case, and processing would
        # take both.
        already = pending_record_ids()
        fresh = [r for r in records if r["record_id"] in chosen and r["record_id"] not in already]
        waiting = [r for r in records if r["record_id"] in chosen and r["record_id"] in already]

        if waiting:
            st.info(
                f"{len(waiting)} already awaiting assessment "
                f"({', '.join(r['record_id'] for r in waiting)}) — assessing rather "
                "than submitting again."
            )
        if fresh:
            n = submit_claims(fresh)
            st.caption(f"{n} claim(s) written. Assessing…")

        results = run_processing(only=[r["record_id"] for r in records if r["record_id"] in chosen])
        report_processing(results)
        # Bumping the nonce clears the selection on the next run. The nonce is
        # not itself a widget key, so assigning to it here is permitted even
        # though the checkboxes already exist.
        st.session_state["pick_nonce"] = nonce + 1


def tab_review() -> None:
    st.subheader("Review Queue")

    # Anything written but not yet assessed — a failed run returned to
    # pending, or claims seeded from the command line.
    waiting = unprocessed_count()
    if waiting:
        left, right = st.columns([3, 1])
        with left:
            st.warning(f"{waiting} claim(s) submitted but not yet assessed.")
        with right:
            if st.button("Assess now", type="primary"):
                results = run_processing()
                report_processing(results)

    order = st.selectbox(
        "Order by",
        ["Oldest first", "Newest first", "Highest value", "Lowest value"],
        label_visibility="collapsed",
    )

    rows = queue_rows(order)
    if not rows:
        st.info("Nothing awaiting review.")
        return

    refs = reference_map()

    st.caption(f"{len(rows)} claim(s) awaiting a decision.")

    for row in rows:
        run_id = str(row["run_id"])
        verdict = row["ai_verdict"]
        icon = "🟢" if verdict == "approve" else "🔴"
        reference = refs.get(row.get("record_id"), row["claim_id"])
        submitted = row.get("submitted_at")
        when = f"  ·  {submitted:%d %b %H:%M}" if submitted else ""

        header = (
            f"{icon}  {reference}  ·  {row['claim_currency']} {row['claim_amount']}  "
            f"·  {row['claim_category']}  ·  {row['submitter']}{when}"
        )

        with st.expander(header):
            if not st.session_state.get(f"opened_{run_id}"):
                mark_opened(run_id)
                st.session_state[f"opened_{run_id}"] = True

            if VERDICT_FIRST:
                render_verdict(row)
                st.divider()
                render_evidence(row)
            else:
                render_evidence(row)
                st.divider()
                render_verdict(row)

            st.divider()

            reviewer = st.text_input(
                "Reviewer", value=st.session_state.get("reviewer", ""), key=f"who_{run_id}"
            )
            st.session_state["reviewer"] = reviewer

            # Displayed capitalised; stored lowercase to satisfy the enum.
            label = st.radio(
                "Decision",
                ["Approve", "Decline"],
                index=None,
                horizontal=True,
                key=f"choice_{run_id}",
            )
            choice = label.lower() if label else None

            overturning = choice is not None and choice != verdict
            rationale = ""
            if overturning:
                st.warning(
                    f"The system assessed this as **{verdict.capitalize()}**. "
                    f"You have chosen **{choice.capitalize()}**."
                )
                rationale = st.text_area(
                    "Overturn Rationale", key=f"rationale_{run_id}", height=80
                )

            response = st.text_area(
                "AI Generated Response",
                value=row["ai_reason_detail"] or "",
                key=f"response_{run_id}",
                height=100,
                help="What the submitter receives. Edit before recording if needed.",
            )

            blocked = (
                choice is None
                or not reviewer.strip()
                or (overturning and not rationale.strip())
            )
            if st.button("Record Decision", key=f"go_{run_id}", disabled=blocked, type="primary"):
                record_decision(run_id, choice, rationale.strip(), reviewer.strip(), response)
                st.success(f"{row['claim_id']} recorded as {choice}.")
                st.rerun()


def tab_completed() -> None:
    st.subheader("Completed")

    rows = decided_rows()
    if not rows:
        st.info("No decisions recorded yet.")
        return

    df = pd.DataFrame(rows)

    st.dataframe(
        df[["claim_id", "submitter", "claim_category", "claim_amount",
            "ai_verdict", "human_verdict", "agreement", "detail_opened",
            "reviewer_id", "decided_at"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### Results")

    total = len(df)
    agreed = int((df["agreement"] == "agreed").sum())
    overturned = int((df["agreement"] == "overturned").sum())
    rewritten = int(df["reason_overwritten"].fillna(False).sum())

    a, b, c, d = st.columns(4)
    a.metric("Decisions", total)
    b.metric("Agreed", agreed)
    c.metric("Overturned", overturned)
    d.metric("Response rewritten", rewritten)

    if overturned:
        st.markdown("**Overturns**")
        st.caption(
            "Where the reviewer disagreed with the AI. These are the improvement "
            "backlog: each one is a case the corpus should contain."
        )
        st.dataframe(
            df[df["agreement"] == "overturned"][
                ["claim_id", "claim_category", "claim_amount",
                 "ai_verdict", "human_verdict", "human_reason_detail", "reviewer_id"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Agreement by AI confidence**")
    st.caption(
        "A band with sustained agreement is the evidence that would justify "
        "auto-clearing it at a later autonomy level."
    )
    band = df.dropna(subset=["ai_confidence"]).copy()
    if band.empty:
        st.caption("No confidence scores recorded.")
    else:
        band["ai_confidence"] = band["ai_confidence"].astype(float)
        band["band"] = pd.cut(
            band["ai_confidence"],
            bins=[0, 0.7, 0.85, 0.95, 1.0],
            labels=["<0.70", "0.70–0.85", "0.85–0.95", "0.95+"],
        )
        summary = (
            band.groupby("band", observed=False)
            .agg(decisions=("claim_id", "count"),
                 overturned=("agreement", lambda s: int((s == "overturned").sum())))
            .reset_index()
        )
        summary["agreement rate"] = summary.apply(
            lambda r: f"{(r['decisions'] - r['overturned']) / r['decisions']:.0%}"
            if r["decisions"] else "—",
            axis=1,
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


def tab_about() -> None:
    st.markdown("""
A learning experiment in building an agentic decision-making system in a form
that could plausibly be deployed in an enterprise.

The objective was to build a working prototype and focus on testing and
recording information on AI decisions relative to human review outcome.

#### What it does

Performs an AI review of a pre-populated mock expense claim against an Expense
Policy and presents a queue item to a human reviewer.

#### How it works

1. Select a claim on the Test Claims tab and submit it
2. On the Review Queue tab you see the receipt, extracted values, checks performed
   and the verbatim clauses cited, then approve or decline
3. Disagreeing with the AI requires a reason to overturn
4. All completed claims are recorded on the Completed tab, with the rate of
   overturned decisions

#### Architecture

1. Interface built in Python using Streamlit
2. Database on Neon (hosted Postgres) holding claims, runs and the event log
3. Checks and verdict logic in Python
4. Judgement-based checks call the Claude API; receipt extraction is stubbed
   behind a real interface
5. Policy version, model version and run label recorded on every run so
   results stay comparable
""")


def tab_policy() -> None:
    st.subheader("Expense policy")
    policy = policy_object()
    st.caption(
        f"Version {policy.version} · the document the system assessed against. "
        "Use it to verify any clause cited on a claim."
    )
    # Plain text rather than rendered markdown. The tab exists so a reviewer
    # can check what a cited clause actually says, and uniform type is easier
    # to scan for a clause number than a document with five heading sizes
    # competing for attention. It also keeps the limits table aligned, which
    # markdown rendering does not.
    st.text(readable_policy())


# --------------------------------------------------------------------------


def main() -> None:
    st.title("Prototype Expense Claim Agent")

    try:
        waiting = pending_count()
    except Exception as exc:
        st.error(f"Cannot reach the database: {exc}")
        st.stop()

    # Submit counts corpus records; Review and Completed count runs. A record
    # can be run more than once, so the figures are not meant to sum.
    cases = len(load_corpus())
    decided = completed_count()

    about, submit, review, completed, policy = st.tabs(
        ["About", f"Test Claims ({cases})", f"Review Queue ({waiting})",
         f"Completed ({decided})", "Policy"]
    )
    with about:
        tab_about()
    with submit:
        tab_submit()
    with review:
        tab_review()
    with completed:
        tab_completed()
    with policy:
        tab_policy()


main()
