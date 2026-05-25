"""Explainability exporter for motions.

Produces per-motion explanations combining deterministic matched rules,
embedding similarities, and supervised-model feature contributions. Exports
JSON (and optional simple HTML) for human review and integration into the
active-learning workflow.

Usage examples:
    uv run --active python -m swedish_parliament_policy_classifier.scripts.explain --limit 100 --out explanations.json
    uv run --active python -m swedish_parliament_policy_classifier.scripts.explain --id MOTION123 --out /tmp/explain.json --html /tmp/explain.html
"""

from pathlib import Path
import argparse
import json
import logging
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd

# Canonical import: load_definitions and score_motion must always be imported from exports.py
# This anchors the code graph and reduces INFERRED edges for Graphify/static analysis.
from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion

if False:
    from swedish_parliament_policy_classifier.exports import load_definitions as _ld, classify_motion as _cm
    _ = _ld, _cm
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher

LOG = logging.getLogger(__name__)


def load_motions_parquet(normalized_parquet: str | Path = "data/parquet/normalized_motions.parquet", motion_id: Optional[str] = None, limit: int = 200):
    p = Path(normalized_parquet)
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    if motion_id:
        row = df[df["id"].astype(str) == str(motion_id)]
        if row.empty:
            return []
        r = row.iloc[0]
        return [(str(r.get("id")), r.get("title"), r.get("text"), r.get("party"), r.get("date"))]
    else:
        df_sorted = df.sort_values("date", ascending=False)
        out = []
        for _, r in df_sorted.head(limit).iterrows():
            out.append((str(r.get("id")), r.get("title"), r.get("text"), r.get("party"), r.get("date")))
        return out


def explain_motion(motion_row, categories, embedding_matcher, supervised_pipeline, mlb, topk_features: int = 10):
    mid, title, text, party, date = motion_row

    # Deterministic scoring (keywords/regexes only)
    det_results = classify_motion(mid, text, categories, embedding_matcher=None, embedding_weight=0.0, use_supervised=False)
    det_map = {r.category: {"raw_score": r.raw_score, "normalized": r.normalized_weight, "matched_rules": r.matched_rules} for r in det_results}

    # Embedding matches (if matcher provided)
    emb_matches = []
    try:
        if embedding_matcher is not None:
            cat_embs = embedding_matcher.build_category_embeddings(categories)
            emb_matches = embedding_matcher.match(text, cat_embs, top_k=5)
    except Exception as e:
        LOG.warning("Embedding explain failed for %s: %s", mid, e)

    # Supervised model explanations (if available)
    supervised = None
    if supervised_pipeline is not None and mlb is not None:
        try:
            # predict probabilities
            probs = supervised_pipeline.predict_proba([text])
            prob_vec = probs[0]
            labels = list(mlb.classes_)

            # vectorizer + classifier internals
            try:
                vect = supervised_pipeline.named_steps["tfidf"]
                clf = supervised_pipeline.named_steps["clf"]
            except Exception:
                vect = None
                clf = None

            sup_preds = []
            for label, p in zip(labels, prob_vec):
                entry = {"label": label, "prob": float(p)}
                # coefficient-based feature contributions for linear models
                if vect is not None and clf is not None and hasattr(clf, "estimators_"):
                    try:
                        X = vect.transform([text])
                        xarr = X.toarray()[0]
                        # find estimator corresponding to label
                        idx = list(labels).index(label)
                        est = clf.estimators_[idx]
                        coef = est.coef_.ravel()
                        contrib = coef * xarr
                        # take top positive contributions
                        top_idx = np.argsort(contrib)[-topk_features:][::-1]
                        feature_names = vect.get_feature_names_out()
                        top_features = []
                        for i in top_idx:
                            if contrib[i] <= 0:
                                continue
                            top_features.append({
                                "feature": feature_names[i],
                                "contribution": float(contrib[i]),
                                "value": float(xarr[i]),
                            })
                        entry["top_features"] = top_features
                    except Exception as e:
                        LOG.debug("Failed per-feature explain for label %s: %s", label, e)
                sup_preds.append(entry)

            supervised = {"labels": sup_preds}
        except Exception as e:
            LOG.warning("Supervised explain failed: %s", e)

    explanation = {
        "motion_id": mid,
        "title": title,
        "party": party,
        "date": str(date),
        "deterministic": det_map,
        "embedding_matches": [{"category": name, "score": float(score)} for name, score in emb_matches],
        "supervised": supervised,
    }

    return explanation


def write_json(out_path: Path, data: List[dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_html(out_path: Path, data: List[dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset=\"utf-8\"><title>Explainability Export</title></head><body>\n")
        f.write(f"<h1>Explainability Export ({len(data)} motions)</h1>\n")
        for ex in data:
            f.write("<hr>\n")
            f.write(f"<h2>{ex['motion_id']}: {ex.get('title','')}</h2>\n")
            f.write(f"<p><strong>Party:</strong> {ex.get('party','')} &nbsp; <strong>Date:</strong> {ex.get('date','')}</p>\n")
            f.write("<h3>Deterministic Matches</h3>\n<ul>\n")
            for k, v in ex.get("deterministic", {}).items():
                f.write(f"<li>{k}: matched_rules={v.get('matched_rules')} raw_score={v.get('raw_score')} normalized={v.get('normalized')}</li>\n")
            f.write("</ul>\n")
            f.write("<h3>Embedding Matches</h3>\n<ul>\n")
            for em in ex.get("embedding_matches", []):
                f.write(f"<li>{em['category']}: {em['score']}</li>\n")
            f.write("</ul>\n")
            f.write("<h3>Supervised Predictions</h3>\n<ul>\n")
            if ex.get('supervised') and ex['supervised'].get('labels'):
                for lab in ex['supervised']['labels']:
                    f.write(f"<li>{lab['label']}: prob={lab['prob']}<br/>")
                    for tf in lab.get('top_features', [])[:10]:
                        f.write(f"&nbsp;&nbsp;{tf['feature']}: contrib={tf['contribution']:.4f} (val={tf['value']:.4f})<br/>")
                    f.write("</li>\n")
            f.write("</ul>\n")
        f.write("</body></html>")


def main(normalized_parquet: Optional[str], motion_id: Optional[str], limit: int, out_json: str, out_html: Optional[str], model_dir: Optional[str], topk: int):
    categories = load_definitions()

    embedding_matcher = EmbeddingMatcher()

    supervised_pipeline = None
    mlb = None
    try:
        if model_dir is None:
            model_dir = Path(__file__).resolve().parents[1] / "models"
        md = Path(model_dir)
        clf_path = md / "supervised_clf.joblib"
        mlb_path = md / "supervised_mlb.joblib"
        if clf_path.exists() and mlb_path.exists():
            supervised_pipeline = joblib.load(str(clf_path))
            mlb = joblib.load(str(mlb_path))
            LOG.info("Loaded supervised model from %s", md)
    except Exception as e:
        LOG.warning("Failed to load supervised model: %s", e)

    rows = load_motions_parquet(normalized_parquet or "data/parquet/normalized_motions.parquet", motion_id=motion_id, limit=limit)
    explanations = []
    for r in rows:
        ex = explain_motion(r, categories, embedding_matcher, supervised_pipeline, mlb, topk_features=topk)
        explanations.append(ex)

    write_json(Path(out_json), explanations)
    if out_html:
        write_html(Path(out_html), explanations)

    print(f"Wrote explanations for {len(explanations)} motions to {out_json}" + (f" and HTML {out_html}" if out_html else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None)
    parser.add_argument("--id", default=None, help="Single motion id to explain")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default="data/explanations.json", help="JSON output file")
    parser.add_argument("--html", default=None, help="Optional HTML output file")
    parser.add_argument("--model_dir", default=None, help="Directory containing supervised_clf.joblib and supervised_mlb.joblib")
    parser.add_argument("--topk", type=int, default=10, help="Top K features per class")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.db, args.id, args.limit, args.out, args.html, args.model_dir, args.topk)
