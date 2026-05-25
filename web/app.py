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

from swedish_parliament_policy_classifier.exports import (
    load_definitions,
    classify_motion,
    classify_and_persist,
)
import pandas as pd

# prefer the parquet persistence helpers when available
try:
    from classifier.persist_parquet import save_annotation as _save_annotation, get_next_unlabeled_motion as _get_next_unlabeled_motion, _read_table_compat as _read_table
except Exception:
    try:
        from swedish_parliament_policy_classifier.classifier.persist_parquet import save_annotation as _save_annotation, get_next_unlabeled_motion as _get_next_unlabeled_motion, _read_table_compat as _read_table
    except Exception:
        _save_annotation = None
        _get_next_unlabeled_motion = None
        _read_table = None

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
    # Parquet-first: load normalized motions + classifications + annotations
    repo_root = Path(__file__).resolve().parents[2]
    nm_path = repo_root / "data" / "parquet" / "normalized_motions.parquet"
    cls_path = repo_root / "data" / "parquet" / "classifications.parquet"
    ann_path = repo_root / "data" / "parquet" / "annotations.parquet"

    try:
        if _get_next_unlabeled_motion is not None:
            row = _get_next_unlabeled_motion(normalized_parquet=nm_path, classifications_parquet=cls_path, annotations_parquet=ann_path, threshold=threshold)
            if not row:
                return JSONResponse({"found": False})
            return JSONResponse({"found": True, "motion_id": row[0], "title": row[1], "text": row[2], "party": row[3], "date": row[4], "top_category": row[5], "top_weight": row[6]})
    except Exception:
        pass

    if not nm_path.exists():
        return JSONResponse({"found": False})

    try:
        if _read_table is not None:
            nm = _read_table(nm_path)
        else:
            nm = pd.read_parquet(nm_path)
    except Exception:
        return JSONResponse({"found": False})
    nm["id"] = nm["id"].astype(str)

    top_map = {}
    if cls_path.exists():
        try:
            if _read_table is not None:
                cls = _read_table(cls_path)
            else:
                cls = pd.read_parquet(cls_path)
            cls_sorted = cls.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
            top = cls_sorted.groupby("motion_id", sort=False).first().reset_index()
            top_map = {str(r.motion_id): (r.category, float(r.normalized_weight)) for _, r in top.iterrows()}
        except Exception:
            top_map = {}

    annotated_ids = set()
    if ann_path.exists():
        try:
            if _read_table is not None:
                ann = _read_table(ann_path)
            else:
                ann = pd.read_parquet(ann_path)
            if "motion_id" in ann.columns:
                annotated_ids = set(ann[ann["status"] == "annotated"]["motion_id"].astype(str).unique())
        except Exception:
            annotated_ids = set()

    candidates = []
    for _, r in nm.iterrows():
        mid = str(r.get("id"))
        if mid in annotated_ids:
            continue
        tm = top_map.get(mid)
        top_cat = tm[0] if tm else None
        top_w = float(tm[1]) if tm else 0.0
        if top_w < threshold:
            candidates.append((mid, r.get("title"), r.get("text"), r.get("party"), r.get("date"), top_cat, top_w))

    if not candidates:
        return JSONResponse({"found": False})

    # prefer most recent by date
    candidates_sorted = sorted(candidates, key=lambda x: x[4] or "", reverse=True)
    row = candidates_sorted[0]
    return JSONResponse({"found": True, "motion_id": row[0], "title": row[1], "text": row[2], "party": row[3], "date": row[4], "top_category": row[5], "top_weight": row[6]})


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

    # Persist annotation to Parquet (fall back to exports.save_annotation if needed)
    repo_root = Path(__file__).resolve().parents[2]
    ann_path = repo_root / "data" / "parquet" / "annotations.parquet"
    try:
        if _save_annotation is not None:
            aid = _save_annotation(None, motion_id, annotator, labels, notes=notes, status=status, annotations_out=ann_path)
            return JSONResponse({"ok": True, "annotation_id": aid})
    except Exception:
        pass

    # local fallback
    now = datetime.utcnow().isoformat()
    row = {"motion_id": motion_id, "annotator": annotator, "labels": json.dumps(labels, ensure_ascii=False), "notes": notes, "status": status, "created_at": now, "updated_at": now}
    try:
        if ann_path.exists():
            prev = pd.read_parquet(ann_path)
            out_df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        else:
            out_df = pd.DataFrame([row])
        out_df.to_parquet(ann_path, index=False)
        return JSONResponse({"ok": True, "annotation_id": int(out_df.index[-1])})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist annotation: {e}")


@app.post("/api/classify")
async def api_classify(req: Request):
    payload = await req.json()
    motion_id = payload.get("motion_id")
    title = payload.get("title")
    text = payload.get("text")
    party = payload.get("party")
    date = payload.get("date")
    persist = payload.get("persist", False)

    if not text:
        raise HTTPException(status_code=400, detail="text required")

    # Optionally persist results into Parquet (via classifier.boundary.classify_and_persist)
    if persist:
        nm = {
            "id": motion_id or f"web-{int(time.time()*1000)}",
            "title": title,
            "text": text,
            "date": date,
            "party": party,
            "metadata": {},
        }
        # call the boundary function which will use parquet persist when configured
        results = classify_and_persist(nm, db_conn=None)
    else:
        results = classify_motion(motion_id or f"web-{int(time.time()*1000)}", text)

    out = []
    for r in results:
        created_iso = None
        try:
            ca = getattr(r, "created_at", None)
            created_iso = ca.isoformat() if ca is not None else None
        except Exception:
            created_iso = None
        out.append(
            {
                "motion_id": getattr(r, "motion_id", None),
                "category": getattr(r, "category", None),
                "raw_score": float(getattr(r, "raw_score", 0.0)),
                "normalized_weight": float(getattr(r, "normalized_weight", 0.0)),
                "matched_rules": getattr(r, "matched_rules", []),
                "classifier_version": getattr(r, "classifier_version", None),
                "created_at": created_iso,
            }
        )

    return JSONResponse({"ok": True, "results": out, "persisted": bool(persist)})


@app.get("/api/progress")
def api_progress(threshold: float = 0.2):
    repo_root = Path(__file__).resolve().parents[2]
    nm_path = repo_root / "data" / "parquet" / "normalized_motions.parquet"
    cls_path = repo_root / "data" / "parquet" / "classifications.parquet"
    ann_path = repo_root / "data" / "parquet" / "annotations.parquet"

    total = 0
    labelled = 0
    if nm_path.exists():
        nm = pd.read_parquet(nm_path)
        nm["id"] = nm["id"].astype(str)
        top_map = {}
        if cls_path.exists():
            try:
                cls = pd.read_parquet(cls_path)
                cls_sorted = cls.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
                top = cls_sorted.groupby("motion_id", sort=False).first().reset_index()
                top_map = {str(r.motion_id): float(r.normalized_weight) for _, r in top.iterrows()}
            except Exception:
                top_map = {}

        annotated_ids = set()
        if ann_path.exists():
            try:
                ann = pd.read_parquet(ann_path)
                if "motion_id" in ann.columns:
                    annotated_ids = set(ann[ann["status"] == "annotated"]["motion_id"].astype(str).unique())
            except Exception:
                annotated_ids = set()

        for _, r in nm.iterrows():
            mid = str(r.get("id"))
            if mid in annotated_ids:
                continue
            w = top_map.get(mid, 0.0)
            if float(w) < float(threshold):
                total += 1

    if ann_path.exists():
        try:
            ann = pd.read_parquet(ann_path)
            if "motion_id" in ann.columns:
                labelled = int(ann[ann["status"] == "annotated"]["motion_id"].nunique())
        except Exception:
            labelled = 0

    return JSONResponse({"total_low_confidence": total, "labelled": labelled})


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        # fallback basic page
        return HTMLResponse("<html><body><h1>Labeling UI</h1><p>Place an index.html in web/static/</p></body></html>")
    return FileResponse(index_file)


# (removed Graphify import-hint)
if False:
    # Graphify hint: anchor direct implementation locations so semantic
    # extractors can resolve the symbol to the canonical files.
    from definitions.loader import load_verified_definitions as _hint_load_verified_definitions
    from swedish_parliament_policy_classifier.models import CategoryDef as _hint_CategoryDef
