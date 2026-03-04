from __future__ import annotations
import json
import logging
import os
import shutil
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from ...core.config import settings

logger = logging.getLogger(__name__)
from ...core.db import get_conn
from ...rag.parse import parse_any
from ...rag.index import index
from ...utils.ids import new_id

router = APIRouter(prefix="/docs", tags=["docs"])

# Uploaded files are persisted here so the frontend can render them at a specific page.
_UPLOAD_DIR = os.path.join(settings.DATA_DIR, "uploads")


def _upload_path(doc_id: str, filename: str) -> str:
    """Return the full disk path for a stored upload. Uses the original filename so
    FileResponse sends the right Content-Type and Content-Disposition."""
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    return os.path.join(_UPLOAD_DIR, f"{doc_id}{ext}")

# Sample transcript chunks for testing RAG and time-range retrieval.
# Each dict: text (longer transcript-style content that chunks into 5–10 pieces), tags, location, and either hours_ago or date.
SAMPLE_TRANSCRIPT_CHUNKS = [
    {
        "text": """Q4 pricing strategy meeting. We went through the proposal from finance. The team agreed on a 15% increase for the enterprise tier starting January. Starter plan stays free to drive adoption. Pro tier gets a 10% bump. We'll need to update the pricing page and send the announcement by next Monday. Sarah will draft the customer email. Mike will handle the billing system changes. We also discussed grandfathering existing customers for six months. Legal needs to review the terms. Follow-up scheduled for Thursday.""",
        "hours_ago": 1,
        "tags": ["pricing", "enterprise", "Q4", "billing"],
        "location": "office",
    },
    {
        "text": """Project kickoff for the new mobile app. Sarah will lead frontend, Mike on backend. We set the target launch for March 15. First milestone is the API spec by Friday. We're using React Native for cross-platform. Design system is already in Figma. We need to decide on auth flow—OAuth vs magic link. DevOps will set up the CI pipeline next week. We allocated two engineers full-time. Risk: dependency on the new payment SDK. Mitigation: have a fallback to Stripe direct.""",
        "hours_ago": 3,
        "tags": ["project", "mobile", "kickoff", "React Native"],
        "location": "office",
    },
    {
        "text": """Sprint planning session. We're committing to 23 story points. Main focus is the payment integration—Stripe webhooks and refund flow. Second priority is fixing the login bug that affects Safari users. We'll do a staging deploy by Wednesday. QA will run the regression suite. No new bugs reported from last release. Alex is on PTO Thursday. We're pulling in the analytics ticket from the backlog. Retro is scheduled for Friday afternoon. Action items: update the runbook, document the new API endpoints.""",
        "hours_ago": 8,
        "tags": ["sprint", "planning", "engineering", "Stripe"],
        "location": "office",
    },
    {
        "text": """Customer feedback session with Acme Corp. They're our largest enterprise client. Main requests: better reporting with custom date ranges, CSV export for all tables, and white-label options. We promised a beta by end of month. Follow-up call scheduled for the 28th. They also asked about SSO—we're tracking that for Q2. Their contract renews in April. We need to show progress on the reporting piece. I'll share the notes with product and engineering. Design will mock up the export flow.""",
        "hours_ago": 24,
        "tags": ["customer", "feedback", "Acme", "enterprise"],
        "location": "office",
    },
    {
        "text": """Product roadmap brainstorm. We listed twenty ideas and voted. Top three: dark mode, keyboard shortcuts for power users, and bulk actions for admin. We'll prioritize based on user votes from the in-app survey. Design mockups due next week. Engineering estimates: dark mode two sprints, shortcuts one sprint, bulk actions three sprints. We're also considering a public API for integrations. That's a bigger lift—maybe Q3. Next product sync is Tuesday. I'll circulate the summary.""",
        "hours_ago": 36,
        "tags": ["product", "roadmap", "features", "API"],
        "location": "home",
    },
    {
        "text": """Team standup. Blockers: still waiting on API keys from finance for the sandbox. I've escalated to their manager. Progress: auth refactor is 80% done, should merge by EOD. No new bugs from production. Everyone is on track. Reminder: we're switching to the new incident process starting next week. Post-mortem for last week's outage is in Confluence. Don't forget the security training—due by Friday. We have a guest from the Berlin office joining next sprint.""",
        "hours_ago": 48,
        "tags": ["standup", "engineering", "blockers", "incident"],
        "location": "office",
    },
    {
        "text": """Budget review for next quarter. Marketing is asking for 20% more—mainly for the conference and paid ads. Engineering headcount approved: two senior engineers. We're under on cloud spend thanks to the optimization work. Over on contractors—we'll try to convert one to FTE. Finance wants the revised forecast by Friday. We're also looking at tooling costs. Some licenses we can drop. HR will share the new band structure. Allocations are final once the board approves.""",
        "hours_ago": 2,
        "tags": ["budget", "finance", "Q1", "headcount"],
        "location": "office",
    },
    {
        "text": """Year-end planning meeting. We reviewed the December 1 milestones. Holiday release is on track—feature freeze is next Friday. Final QA sign-off needed by Dec 5. We're pushing the analytics dashboard to January to avoid risk. Support will have extra coverage for the launch week. We're also planning the holiday party—venue is booked. Reminder: submit expense reports before the 15th. We'll do a lightweight retro after the release. Goals for next year: we'll finalize in the January offsite.""",
        "date": "2025-12-01",
        "tags": ["year-end", "planning", "December", "release"],
        "location": "office",
    },
    {
        "text": """October 10 retrospective. We shipped the analytics dashboard on time. Team velocity is up 12% from last quarter. What went well: clear scope, good pair programming sessions. What didn't: we underestimated the data migration. We'll add buffer for similar work next time. Action: document the migration runbook. Next quarter we're focusing on performance and accessibility. We're also starting the mobile project. We'll try the new async standup format. Thanks everyone for the hard work.""",
        "date": "2025-10-10",
        "tags": ["retrospective", "analytics", "October", "velocity"],
        "location": "office",
    },
    {
        "text": """Design review for the onboarding flow. We walked through the new wizard steps. UX raised concerns about the progress indicator—users might not know how many steps remain. We'll add a step counter. Copy needs to be shorter—legal is reviewing. We're A/B testing the welcome email. First results in two weeks. Design will hand off to engineering by Friday. We're using the new component library. One thing: we need to handle the edge case when users skip the company setup. We'll add a way to revisit that later.""",
        "hours_ago": 5,
        "tags": ["design", "onboarding", "UX", "A/B test"],
        "location": "office",
    },
]


def _vector_db_usage_bytes() -> int:
    """Total size of vector DB files: FAISS index, meta JSON, sparse meta, SQLite DB."""
    total = 0
    for path in (
        settings.FAISS_PATH,
        settings.META_PATH,
        settings.SPARSE_META_PATH,
        settings.FAISS_TRANSCRIPT_PATH,
        settings.META_TRANSCRIPT_PATH,
        settings.SPARSE_TRANSCRIPT_META_PATH,
        settings.DB_PATH,
    ):
        if path and os.path.exists(path):
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
    return total


@router.get("/usage")
def storage_usage():
    """Return vector DB storage usage and disk capacity (for sidebar usage bar)."""
    usage_bytes = _vector_db_usage_bytes()
    capacity_bytes = None
    try:
        disk = shutil.disk_usage(settings.DATA_DIR)
        capacity_bytes = disk.total
    except OSError:
        pass
    return {"usage_bytes": usage_bytes, "capacity_bytes": capacity_bytes}


@router.get("/list")
def list_docs():
    """List uploaded documents only (exclude transcript entries; those appear in Transcripts panel)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, filetype, created_at FROM documents WHERE filename NOT LIKE 'transcript_%' ORDER BY created_at DESC"
        ).fetchall()
    return {"documents": [{"id": r[0], "filename": r[1], "filetype": r[2], "created_at": r[3]} for r in rows]}


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    filetype, text, estimated_pages, page_offsets = parse_any(file.filename, data)
    res = await index.add_document(
        file.filename,
        filetype,
        text,
        {"filename": file.filename, "filetype": filetype},
        estimated_pages=estimated_pages,
        page_offsets=page_offsets,
    )
    # Persist raw file so it can be served back for in-browser preview.
    doc_id = res.get("doc_id") or res.get("id")
    if doc_id:
        dest = _upload_path(doc_id, file.filename)
        try:
            with open(dest, "wb") as fh:
                fh.write(data)
        except Exception as exc:
            logger.warning("Could not persist upload for preview: %s", exc)
    return {"ok": True, **res}


@router.get("/{doc_id}/file")
async def serve_file(doc_id: str):
    """Serve the original uploaded file so the frontend can render it (e.g. PDF iframe at a page)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT filename FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    filename = row[0]
    path = _upload_path(doc_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not stored on server — please re-upload to enable preview")
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    media_type_map = {".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".txt": "text/plain"}
    media_type = media_type_map.get(ext, "application/octet-stream")
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.delete("/{doc_id}")
async def delete_doc(doc_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT id, filename FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    _id, filename = row
    await index.delete_document(doc_id)
    # Remove stored file (best-effort)
    try:
        upload_path = _upload_path(doc_id, filename or "")
        if os.path.exists(upload_path):
            os.remove(upload_path)
    except Exception as exc:
        logger.warning("Could not remove stored upload %s: %s", doc_id, exc)
    return {"ok": True, "deleted": doc_id}


@router.get("/data-preview")
def data_preview():
    """Full data preview: documents, chunks, transcripts (for Usage popover)."""
    with get_conn() as conn:
        docs = conn.execute(
            "SELECT id, filename, filetype, created_at, meta_json FROM documents ORDER BY created_at DESC"
        ).fetchall()
        chunks = conn.execute(
            "SELECT id, doc_id, chunk_index, substr(text, 1, 200) as text_preview FROM chunks ORDER BY doc_id, chunk_index"
        ).fetchall()
        transcripts = conn.execute(
            "SELECT id, title, tags_json, echotag, created_at, length(raw_text) as raw_len, length(polished_text) as polished_len FROM transcripts ORDER BY created_at DESC"
        ).fetchall()
    documents = [{"id": r[0], "filename": r[1], "filetype": r[2], "created_at": r[3], "meta_json": r[4]} for r in docs]
    chunks_out = [{"id": r[0], "doc_id": r[1], "chunk_index": r[2], "text_preview": (r[3] or "") + ("..." if (r[3] and len(r[3]) >= 200) else "")} for r in chunks]
    transcripts_out = []
    for r in transcripts:
        tid, title, tags_json, echotag, created_at, raw_len, polished_len = r
        tags = []
        if tags_json:
            try:
                tags = json.loads(tags_json) if isinstance(tags_json, str) else (tags_json or [])
            except Exception:
                pass
        transcripts_out.append({
            "id": tid,
            "title": title or tid,
            "tags": tags,
            "echotag": echotag or "",
            "created_at": created_at or "",
            "raw_length": raw_len or 0,
            "polished_length": polished_len or 0,
        })
    return {"documents": documents, "chunks": chunks_out, "transcripts": transcripts_out}


@router.post("/delete-all")
async def delete_all_data():
    """Delete all data: documents, chunks, transcripts, chats, messages. Uses bulk clear for speed."""
    with get_conn() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM transcripts")
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM chats")
        conn.commit()
    index.clear_all()
    # Remove all stored upload files
    try:
        if os.path.isdir(_UPLOAD_DIR):
            shutil.rmtree(_UPLOAD_DIR)
    except Exception as exc:
        logger.warning("Could not clear uploads dir: %s", exc)
    return {"ok": True, "message": "All data deleted."}


@router.post("/add-sample-transcripts")
async def add_sample_transcripts():
    """
    Add sample transcript chunks to the transcripts table AND embedding index for testing.
    Chunks appear in the Transcripts tab (Resources panel) and are searchable in chat.
    Chunks use either hours_ago (relative, last 48h) or date (fixed, e.g. 2025-12-01, 2025-10-10)
    so time-range queries like "last 2 hours", "2 Dec 2025", or "10 Oct 2025" work correctly.
    """
    now = datetime.now(timezone.utc)
    added = []
    for i, item in enumerate(SAMPLE_TRANSCRIPT_CHUNKS):
        text = item["text"]
        tags = item["tags"]
        location = item["location"]
        name = f"Sample {i + 1}"
        if "date" in item:
            date_str = item["date"]
            try:
                ts = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            except ValueError:
                ts = now
            label = date_str
        else:
            hours_ago = item["hours_ago"]
            ts = now - timedelta(hours=hours_ago)
            label = f"{hours_ago}h ago"
        epoch = int(ts.timestamp())
        echodate = ts.isoformat()
        echotag = ",".join(tags) if tags else name

        tid = new_id("trn")
        title = f"{name} ({label})"

        # Log what gets added to transcripts table and embedding index
        logger.info(
            "add_sample_transcripts: inserting transcript | tid=%s | title=%s | text_len=%d | tags=%s | echodate=%s | epoch=%s | location=%s | name=%s | doc_filename=transcript_%s",
            tid, title, len(text), tags, echodate, epoch, location, name, tid,
        )
        text_preview = (text[:80] + "...") if (text and len(text) > 80) else (text or "")
        logger.debug("add_sample_transcripts: text_preview=%s | meta=%s", text_preview, {"type": "transcript", "tags": tags, "echodate": echodate, "epoch": epoch, "location": location, "transcript_id": tid})

        # Insert into transcripts table so it appears in the Transcripts tab
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO transcripts (id, title, raw_text, polished_text, tags_json, echotag, echodate, created_at, updated_at, name, location) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (tid, title, text, None, json.dumps(tags), echotag, echodate, echodate, echodate, name, location),
            )
            conn.commit()

        # Add to RAG index for chat search
        meta = {
            "type": "transcript",
            "tags": tags,
            "echodate": echodate,
            "location": location,
            "name": name,
            "epoch": epoch,
            "transcript_id": tid,
        }
        await index.add_text(f"transcript_{tid}", text, meta)

        added.append({"id": tid, "label": label, "tags": tags})
    return {"ok": True, "added": len(added), "items": added}
