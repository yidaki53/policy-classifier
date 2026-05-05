"""Simple FastAPI web UI for human labeling with SQLite persistence.

Endpoints:
- GET /api/categories -> list category names
- GET /api/next?threshold=0.2 -> next unlabeled motion (low-confidence)
- POST /api/annotate -> save annotation
- GET /api/progress -> counts

Static UI served at `/`.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from swedish_parliament_policy_classifier.classifier.scorer import load_definitions
from swedish_parliament_policy_classifier.classifier import persist as persist_mod
from swedish_parliament_policy_classifier.db.schema import get_connection

LOG = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[0]
STATIC_DIR = Path(__file__).resolve().parents[0] / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/categories")
def api_categories():
    cats = load_definitions()
    return JSONResponse([ {"name": c.name, "definition": c.definition} for c in cats.values() ])


@app.get("/api/next")
def api_next(threshold: float = 0.2):
    conn = get_connection(Path(__file__).resolve().parents[2] / "data" / "swedish_parliament.db")
    row = persist_mod.get_next_unlabeled_motion(conn, threshold=threshold)
    if not row:
        return JSONResponse({"found": False})
    return JSONResponse({
        "found": True,
        "motion_id": row[0],
        "title": row[1],
        "text": row[2],
        "party": row[3],
        "date": row[4],
        "top_category": row[5],
        "top_weight": row[6],
    })


@app.post("/api/annotate")
async def api_annotate(req: Request):
    payload = await req.json()
    motion_id = payload.get("motion_id")
    annotator = payload.get("annotator", "human")
    labels = payload.get("labels", [])
    notes = payload.get("notes")
    status = payload.get("status", "annotated")

    if not motion_id:
        raise HTTPException(status_code=400, detail="motion_id required")

    conn = get_connection(Path(__file__).resolve().parents[2] / "data" / "swedish_parliament.db")
    aid = persist_mod.save_annotation(conn, motion_id, annotator, labels, notes=notes, status=status)
    return JSONResponse({"ok": True, "annotation_id": aid})


@app.get("/api/progress")
def api_progress(threshold: float = 0.2):
    conn = get_connection(Path(__file__).resolve().parents[2] / "data" / "swedish_parliament.db")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM normalized_motions nm LEFT JOIN classifications c ON c.id = (SELECT id FROM classifications WHERE motion_id = nm.id ORDER BY normalized_weight DESC LIMIT 1) WHERE COALESCE(c.normalized_weight,0) < ?", (threshold,))
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT motion_id) FROM annotations WHERE status = 'annotated'")
    labelled = cur.fetchone()[0]
    return JSONResponse({"total_low_confidence": total, "labelled": labelled})


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        # fallback basic page
        return HTMLResponse("<html><body><h1>Labeling UI</h1><p>Place an index.html in web/static/</p></body></html>")
    return FileResponse(index_file)
