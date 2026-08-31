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
from agent.db import connect, corpus_path, policy_path, setting  # noqa: E402
from agent.pipeline import pending_count as unprocessed_count  # noqa: E402
from agent.pipeline import process_pending  # noqa: E402
from agent.policy import load as load_policy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = corpus_path()
POLICY_FILE = policy_path()
RECEIPTS = Path(__file__).resolve().parent / "assets" / "receipts"
FALLBACK_RECEIPT = Path(__file__).resolve().parent / "assets" / "demo_receipt.png"

# Placement of the AI verdict in the detail view. Leading with a conclusion
# is faster but means agreement partly measures compliance rather than
# independent judgement. Flip this to compare overturn rates either way.
VERDICT_FIRST = setting("SHOW_VERDICT_FIRST", "true").lower() == "true"

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
      button[kind="primary"],
      button[kind="primaryFormSubmit"] {
          background-color: #E8F0FA !important;
          color: #143F73 !important;
          border: 1px solid #1F5FA8 !important;
          font-weight: 600 !important;
      }
      button[kind="primary"]:hover:enabled,
      button[kind="primaryFormSubmit"]:hover:enabled {
          background-color: #1F5FA8 !important;
          color: #FFFFFF !important;
      }
      /* Streamlit prints "Press Cmd+Enter to submit form" inside every
         multiline text area in a form. There is no option to disable it, the
         shortcut does not reliably work, and the decision should be recorded
         by pressing the button rather than by a hidden key combination. */
      div[data-testid="InputInstructions"] { display: none !important; }
      button[kind="primary"]:disabled,
      button[kind="primaryFormSubmit"]:disabled {
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


@st.cache_resource
def policy_object():
    """The parsed policy.

    Cached for the life of the process: parsing runs validate(), which
    checks the document still contains what the check mapping expects. That
    should fail once at startup rather than partway through a batch.
    """
    return load_policy(POLICY_FILE)


@st.cache_data
def policy_blocks() -> list[tuple[str, object]]:
    """The policy split into text and table blocks.

    Rendered separately because they need different widgets. Padding table
    cells to fixed character widths only aligns in a monospace font, and the
    theme font is proportional — which is why the limits table, the part of
    the policy most often checked, came out staggered.
    """
    blocks: list[tuple[str, object]] = []
    text: list[str] = []
    table: list[list[str]] = []

    def flush_text() -> None:
        if text:
            body = "\n".join(text).strip("\n")
            if body.strip():
                blocks.append(("text", body))
            text.clear()

    def flush_table() -> None:
        if table:
            header, *rows = table
            blocks.append(("table", pd.DataFrame(rows, columns=header)))
            table.clear()

    for raw in policy_text().splitlines():
        line = raw.rstrip()

        if line.startswith("|"):
            # separator rows carry no data
            if set(line) <= set("|-: "):
                continue
            flush_text()
            table.append([c.strip() for c in line.strip("|").split("|")])
            continue

        flush_table()

        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                text.append("")
                text.append(heading.upper())
                text.append("-" * len(heading))
            continue

        if line.strip() in {"---", "***", "___"}:
            text.append("")
            continue

        text.append(line.replace("**", "").replace("*", ""))

    flush_table()
    flush_text()
    return blocks


@st.cache_data
def reference_map() -> dict[str, str]:
    """Record id to the reference shown in the interface.

    Since the corpus moved to sequential ids these are the same string, so
    the map is effectively an identity. It is kept because claims seeded
    under the old lettered ids are still in the database, and a stale row
    should fall back to its own id rather than disappear from the queue.
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
    shows each check calling the model with a named slice of policy, and the
    verdict computed at the end in code with no model call at all.
    """
    policy = policy_object()
    lines: list[str] = []

    # st.status pins a labelled block that stays put and reports its own
    # state. The previous arrangement rendered the log at the foot of the
    # page, where a reviewer could be scrolled past it and not know anything
    # was happening.
    status = st.status("Assessing claims…", expanded=True)
    with status:
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

    status.update(label=f"Assessed {len(results)} claim(s)", state="complete")
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
                   r.ai_reason_plain, r.ai_confidence, r.check_results, r.created_at,
                   r.policy_version, r.model_version, r.run_label,
                   c.submitter, c.claim_amount, c.claim_currency,
                   c.claim_category, c.claim_date, c.business_purpose,
                   c.cost_exception_rationale,
                   c.other_category_rationale, c.extraction, c.record_id, c.submitted_at
              FROM runs r
              JOIN claims c ON c.claim_id = r.claim_id
             WHERE r.claim_status = 'awaiting_review'
             ORDER BY {clause}
            """
        ).fetchall()


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
                   r.ai_confidence, r.reviewer_id,
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
    elif verdict == "decline":
        st.error(f"**Decline**{conf_text}")
    else:
        # The system reporting that it could not reach a position, which is
        # not the same as reaching a negative one.
        st.warning(f"**Review** — could not be determined{conf_text}")
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
        # Plain text, not markdown. This is a quotation from a document, and
        # rendering it as markdown lets the policy's own punctuation change
        # how it looks — clause 2.3 opens with a quote mark, which markdown
        # reads as formatting and renders as a heading.
        st.text(f"{clause.ref}  {clause.text}")


def render_findings(row) -> None:
    """What the system found, and the policy it applied.

    Placed above the claim because that is the order a reviewer works in:
    most claims are routine, so the finding comes first and the evidence is
    consulted only where something needs checking.
    """
    results = as_list(row["check_results"])

    st.markdown("**Checks**")
    table = []
    for c in sorted(results, key=lambda x: x["check_id"]):
        rationale = c.get("detail", "")
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
        f"Assessed against policy v{row['policy_version']} · {row['model_version']}"
    )


def render_claim(row) -> None:
    """The claim as the submitter filled it in.

    The same renderer the Test Claims page uses. A reviewer and a submitter
    should be looking at the same thing: rendering the claim as a table here
    and as a form there would mean two representations of one record, and
    the reviewer's job is partly to check the form against the receipt.
    """
    claim = {
        "submitter": row["submitter"],
        "claim_amount": float(row["claim_amount"]),
        "claim_currency": row["claim_currency"],
        "claim_category": row["claim_category"],
        "claim_date": row["claim_date"],
        "business_purpose": row["business_purpose"],
        "cost_exception_rationale": row["cost_exception_rationale"],
        "other_category_rationale": row["other_category_rationale"],
    }
    render_submission_form(claim, row.get("record_id"), key_prefix=f"rev_{row['run_id']}")







# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------


def render_submission_form(claim: dict, record_id: str | None, *, key_prefix: str = "sub") -> None:
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
                      key=f"{key_prefix}_sub_{record_id}")
        st.text_input("Amount", value=f"{claim['claim_currency']} {claim['claim_amount']:.2f}",
                      disabled=True, key=f"{key_prefix}_amt_{record_id}")
        st.text_input("Date incurred", value=str(claim["claim_date"]), disabled=True,
                      key=f"{key_prefix}_date_{record_id}")
    with right:
        st.text_input("Category", value=claim["claim_category"], disabled=True,
                      key=f"{key_prefix}_cat_{record_id}")
        st.text_input("Attachment", value="receipt.png", disabled=True,
                      key=f"{key_prefix}_att_{record_id}")

    st.text_area("Business purpose", value=claim["business_purpose"], disabled=True,
                 height=70, key=f"{key_prefix}_purp_{record_id}")

    st.text_area(
        "Please give a rationale if cost exceeds policy",
        value=claim.get("cost_exception_rationale") or "",
        disabled=True,
        height=90,
        key=f"{key_prefix}_cost_{record_id}",
    )
    st.text_area(
        "If Other selected for reason, please enter reason",
        value=claim.get("other_category_rationale") or "",
        disabled=True,
        height=90,
        key=f"{key_prefix}_other_{record_id}",
    )

    image = receipt_for(record_id)
    st.markdown("**Attached receipt**")
    if image is not None:
        st.image(str(image), width=320)
    else:
        st.caption(f"No image for {record_id}. Run scripts/make_receipts.py.")


def tab_submit() -> None:
    st.subheader("Test Claims")

    # Two views, not one. Streamlit cannot disable a control that is already
    # painted in the browser: while a run is in progress the server processes
    # nothing, so a disabled= flag never reaches the page and clicks queue up
    # instead. Rendering a different view is the one thing it can do — during
    # a run the corpus list is not on the page at all, so there is nothing to
    # click and the console is the only thing in view.
    if st.session_state.get("run_request"):
        _render_processing_view()
    else:
        _render_selection_view()


def _render_selection_view() -> None:
    records = load_corpus()
    awaiting = pending_record_ids()

    # Checkbox keys carry a nonce. Deleting a widget's key does not reliably
    # reset it — Streamlit restores the value from the incoming frontend
    # state when the widget is recreated in the same run. Bumping the nonce
    # sidesteps that: the checkboxes become new widgets with no history.
    nonce = st.session_state.get("pick_nonce", 0)

    def pick_key(record_id: str) -> str:
        return f"pick_{nonce}_{record_id}"

    # Clear sits outside the form because it must act immediately. There is
    # deliberately no Select all: it is the obvious button to press, it would
    # run the whole corpus, and a visitor would fire two dozen model calls
    # before reading anything.
    if st.button("Clear Selection"):
        st.session_state["pick_nonce"] = nonce + 1
        st.rerun()

    st.divider()

    # A form batches the checkboxes so nothing re-runs until Submit is
    # pressed. Without it every tick re-runs the script, which resets the
    # tabs and throws you back to About.
    with st.form(key=f"submit_{nonce}"):
        # Submit sits above the list. With thirty records the button was a
        # long scroll away, and the run then rendered below the fold where a
        # reviewer could miss it entirely.
        submitted = st.form_submit_button("Submit", type="primary")
        # Reserved for the validation message, so it appears beside the
        # button rather than a page below it.
        error_slot = st.empty()
        st.divider()

        for rec in records:
            claim = rec["claim"]
            rid = rec["record_id"]

            label = rec.get("label") or rec["purpose"].rstrip(".")
            reference = rec.get("reference", rid)
            state = "  ·  awaiting assessment" if rid in awaiting else ""

            pick, body = st.columns([1, 14])
            with pick:
                st.checkbox("", key=pick_key(rid), label_visibility="collapsed")
            with body:
                header = (
                    f"{reference}  ·  {label}  ·  £{claim['claim_amount']:.2f}"
                    f"  ·  Expected Result: {rec['expected_verdict'].upper()}{state}"
                )
                with st.expander(header):
                    render_submission_form(claim, rid)

    if submitted:
        chosen = [
            r["record_id"] for r in records
            if st.session_state.get(pick_key(r["record_id"]))
        ]
        if not chosen:
            error_slot.error("Select at least one claim.")
        else:
            # Record what to run and re-render. The work happens on the next
            # pass, in the processing view, so the corpus list is gone from
            # the page before the first model call is made.
            st.session_state["run_request"] = chosen
            st.rerun()


def _pin_page(page: str) -> None:
    """Request a page for the next run.

    Written to its own key, not the radio's. By the time a page function
    runs, main() has already created the widget with key "page", and
    Streamlit discards writes to an instantiated widget's state. main()
    applies this at the top of the next run, before the radio exists.
    """
    st.session_state["_pinned_page"] = page


def _go_to(page: str) -> None:
    """Set the active page.

    Used as an on_click callback rather than assigned inside a button's if
    block. main() renders the sidebar radio before it calls the page
    function, so by the time that block runs the widget with key "page"
    already exists in the same script run — and Streamlit discards writes to
    an instantiated widget's state. A callback runs before the next run
    begins, when no widget has been created yet.
    """
    st.session_state["page"] = page


def _render_processing_view() -> None:
    chosen = st.session_state.get("run_request") or []
    records = load_corpus()
    nonce = st.session_state.get("pick_nonce", 0)

    st.caption(f"Assessing {len(chosen)} claim(s).")

    # A record already awaiting assessment is not submitted again. Doing so
    # would leave two claims for one test case, and processing would take
    # both.
    already = pending_record_ids()
    fresh = [r for r in records if r["record_id"] in chosen and r["record_id"] not in already]
    waiting = [r for r in records if r["record_id"] in chosen and r["record_id"] in already]

    if waiting:
        st.info(
            f"{len(waiting)} already awaiting assessment "
            f"({', '.join(r['record_id'] for r in waiting)}) — assessing rather "
            "than submitting again."
        )

    try:
        if fresh:
            # Written, not announced. "N claim(s) written" described a
            # database insert, which is not something a reviewer needs to
            # know about — the console below reports the assessment, which is
            # what they are waiting for.
            submit_claims(fresh)
        results = run_processing(only=chosen)
        report_processing(results)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The run failed: {exc}")
        results = []
    finally:
        # Cleared whatever happened: a run that raises must not leave the tab
        # stuck on a processing view with no way back.
        st.session_state["run_request"] = None
        st.session_state["pick_nonce"] = nonce + 1

    st.divider()
    st.button(
        "Back to Test Claims",
        type="primary",
        key="back_to_claims",
        on_click=_go_to,
        args=("Test Claims",),
    )


def tab_review() -> None:
    st.subheader("Review Queue")

    if st.session_state.get("assess_pending"):
        st.caption("Assessing pending claims.")
        try:
            results = run_processing()
            report_processing(results)
        except Exception as exc:  # noqa: BLE001
            st.error(f"The run failed: {exc}")
        finally:
            st.session_state["assess_pending"] = False
        st.divider()
        st.button(
            "Back to the Queue",
            type="primary",
            key="back_to_queue",
            on_click=_go_to,
            args=("Review Queue",),
        )
        return

    # Anything written but not yet assessed — a failed run returned to
    # pending, or claims seeded from the command line.
    waiting = unprocessed_count()
    if waiting:
        left, right = st.columns([3, 1])
        with left:
            st.warning(f"{waiting} claim(s) submitted but not yet assessed.")
        with right:
            if st.button("Assess now", type="primary"):
                # Same reasoning as the Test Claims tab: record the request
                # and re-render, so the queue is off the page before the run
                # starts rather than sitting there looking clickable.
                st.session_state["assess_pending"] = True
                st.rerun()

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
        icon = {"approve": "🟢", "decline": "🔴"}.get(verdict, "🟡")
        reference = refs.get(row.get("record_id"), row["claim_id"])
        submitted = row.get("submitted_at")
        when = f"  ·  {submitted:%d %b %H:%M}" if submitted else ""

        header = (
            f"{icon}  {reference}  ·  {row['claim_currency']} {row['claim_amount']}  "
            f"·  {row['claim_category']}  ·  {row['submitter']}{when}"
        )

        with st.expander(header):
            # Findings sit next to the claim so the two can be compared, with
            # the verbatim policy last: it is long, and it is consulted to
            # verify a citation rather than read on the way past.
            if VERDICT_FIRST:
                render_verdict(row)
                st.divider()
                render_findings(row)
                st.divider()
                render_claim(row)
            else:
                # The alternative ordering exists to measure whether leading
                # with a conclusion changes overturn rates.
                render_claim(row)
                st.divider()
                render_findings(row)
                st.divider()
                render_verdict(row)

            st.divider()
            render_cited_clauses(as_list(row["check_results"]))

            st.divider()

            # A form batches the inputs so nothing re-runs until the button
            # is pressed. Without it every radio click re-runs the script,
            # which resets the tabs and closes the claim mid-decision.
            #
            # The cost is that nothing can react as you type: the overturn
            # warning that used to appear on choosing a different verdict is
            # gone, and a missing rationale is caught on submit instead.
            with st.form(key=f"decide_{run_id}"):
                # Both are required. A form cannot flag them as you type, so
                # the label says so and the button reports what is missing.
                # Deliberately not remembered between claims. Carrying the
                # name forward made it look as though it had been applied to
                # every open claim, and it removed a small friction worth
                # keeping: entering it each time makes each decision a
                # separate act rather than a default, and stops a reviewer
                # working through the queue without pausing.
                reviewer = st.text_input(
                    "Reviewer *",
                    placeholder="Required",
                )

                label = st.radio(
                    "Decision *",
                    ["Approve", "Decline"],
                    index=None,
                    horizontal=True,
                )

                rationale = st.text_area(
                    "Overturn Rationale",
                    height=80,
                    help="Required where your decision differs from the system's.",
                )

                response = st.text_area(
                    "AI Generated Response",
                    value=row["ai_reason_plain"] or row["ai_reason_detail"] or "",
                    height=100,
                    help="What the submitter receives. Edit before recording if needed.",
                )

                st.caption("\\* required")
                submitted = st.form_submit_button("Record Decision", type="primary")

            if submitted:
                choice = label.lower() if label else None
                # Deferring is not a position, so a human decision after a
                # review verdict is not an overturn and needs no justification.
                deferred = verdict == "review"
                overturning = choice is not None and not deferred and choice != verdict

                if choice is None:
                    st.error("Choose Approve or Decline.")
                elif not reviewer.strip():
                    st.error("Enter a reviewer name.")
                elif overturning and not rationale.strip():
                    st.error(
                        f"The system assessed this as {verdict.capitalize()} and you "
                        f"have chosen {choice.capitalize()}. An overturn rationale "
                        "is required."
                    )
                else:
                    record_decision(
                        run_id, choice, rationale.strip(), reviewer.strip(), response
                    )
                    # Rerun so the decided claim leaves the queue immediately.
                    # The page is pinned first: main() renders the sidebar
                    # radio before the page function, so a rerun from here
                    # would otherwise reset the selection to About.
                    #
                    # _pin_page writes to a separate key rather than the
                    # radio's own, because the widget already exists in this
                    # run and Streamlit discards writes to an instantiated
                    # widget's state.
                    _pin_page("Review Queue")
                    st.success(f"{row['claim_id']} recorded as {choice}.")
                    st.rerun()


def tab_completed() -> None:
    st.subheader("Completed")

    rows = decided_rows()
    if not rows:
        st.info("No decisions recorded yet.")
        return

    df = pd.DataFrame(rows)

    # claim_id is the stable internal key (EXP-A1). The reference is the label
    # people see, and it must match what the other two tabs show.
    refs = reference_map()
    df["reference"] = df["record_id"].map(refs).fillna(df["claim_id"])

    st.dataframe(
        df[["reference", "submitter", "claim_category", "claim_amount",
            "ai_verdict", "human_verdict", "agreement",
            "reviewer_id", "decided_at"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### Results")

    total = len(df)
    agreed = int((df["agreement"] == "agreed").sum())
    overturned = int((df["agreement"] == "overturned").sum())
    deferred = int((df["agreement"] == "deferred").sum())
    rewritten = int(df["reason_overwritten"].fillna(False).sum())

    # Deferrals are excluded from the rate: the system took no position, so
    # there was nothing to overturn. Counting them would penalise restraint.
    decided = total - deferred
    rate = f"{overturned / decided:.0%}" if decided else "—"

    a, b, c, d, e = st.columns(5)
    a.metric("Decisions", total)
    b.metric("Agreed", agreed)
    c.metric("Overturned", overturned)
    d.metric("Deferred", deferred)
    e.metric("Overturn rate", rate)
    if rewritten:
        st.caption(f"{rewritten} response(s) rewritten before sending.")

    if overturned:
        st.markdown("**Overturns**")
        st.caption(
            "Where the reviewer disagreed with the AI. These are the improvement "
            "backlog: each one is a case the corpus should contain."
        )
        st.dataframe(
            df[df["agreement"] == "overturned"][
                ["reference", "claim_category", "claim_amount",
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
            .agg(decisions=("reference", "count"),
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

Six checks run against each claim. No limits, categories or conditions are
hardcoded: every threshold is read from the policy document at assessment
time, so amending the policy changes the system's behaviour without any change
to the code. Judgement is performed by the model; the verdict is computed in
code from its findings, so the same findings always produce the same outcome.

#### What it does not do

The receipt is shown to the reviewer but is not read by the system. Claim data
is supplied rather than extracted, so there are no checks comparing a claimed
amount against one taken from the image. Reading the receipt — and marking on
it where each value was found — is the obvious next layer, and the extraction
step exists in the code as the seam it would sit behind.

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
    # Prose as plain text, tables as tables. Uniform type is easier to scan
    # for a clause number than a document with five heading sizes competing
    # for attention, but a table needs to be a table to be legible.
    for kind, block in policy_blocks():
        if kind == "table":
            st.table(block)
        else:
            st.text(block)


# --------------------------------------------------------------------------


PAGES = {
    "About": "tab_about",
    "Test Claims": "tab_submit",
    "Review Queue": "tab_review",
    "Completed": "tab_completed",
    "Policy": "tab_policy",
}


def main() -> None:
    st.title("Prototype Expense Claim Agent")

    try:
        waiting = pending_count()
    except Exception as exc:
        st.error(f"Cannot reach the database: {exc}")
        st.stop()

    # Test Claims counts corpus records; Review and Completed count runs. A
    # record can be run more than once, so the figures are not meant to sum.
    counts = {
        "Test Claims": len(load_corpus()),
        "Review Queue": waiting,
        "Completed": completed_count(),
    }

    # Sidebar navigation rather than tabs. Streamlit preserves sidebar state
    # across re-runs; st.tabs does not, and gives no way to set which tab is
    # active.
    #
    # The option values are the bare page names, with the counts added only
    # for display. Putting a count in the option itself makes the list change
    # whenever a claim is submitted or decided, and Streamlit cannot then
    # match the stored selection to the new options — so it falls back to the
    # first entry and throws you back to About.
    def label(name: str) -> str:
        n = counts.get(name)
        return f"{name} ({n})" if n is not None else name

    # Applied before the radio is created, which is the only point at which
    # its state can be set.
    pinned = st.session_state.pop("_pinned_page", None)
    if pinned:
        st.session_state["page"] = pinned

    with st.sidebar:
        st.markdown("### Navigation")
        choice = st.radio(
            "Page",
            list(PAGES),
            format_func=label,
            label_visibility="collapsed",
            key="page",
        )

    globals()[PAGES[choice]]()


main()
