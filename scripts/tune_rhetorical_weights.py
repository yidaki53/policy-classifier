#!/usr/bin/env python3
"""Hyperparameter tuning for rhetorical pattern weights using ollama teacher labels.

Usage:
    uv run python scripts/tune_rhetorical_weights.py --teacher logs/ollama_teacher_labels.json --n-iter 50 --out models/rhetorical_weights_best.json

Search space: 7 base values + 7 per-count increments for the rhetorical pattern
adjustments in _detect_rhetorical_patterns. Objective: minimize MSE between pipeline
output probabilities and ollama teacher labels. Uses MLflow for logging.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from fractions import Fraction

from swedish_parliament_policy_classifier.exports import load_definitions
from swedish_parliament_policy_classifier.classifier.scorer import score_motion


CATEGORIES = ["far_left", "left", "centre_left", "centre", "centre_right", "right", "far_right"]


def load_teacher_labels(path: str) -> dict[str, dict[str, float]]:
    with open(path) as f:
        data = json.load(f)
    def _convert_scores(item):
        scores = item["scores"]
        return {k: Fraction(v) if not isinstance(v, Fraction) else v for k, v in scores.items()}
    return {item["speech_id"]: _convert_scores(item) for item in data}


def load_speeches(speech_ids: list[str], parquet_dir: Path) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df[all_df["anforande_id"].isin(speech_ids)]


def score_with_params(
    speech_id: str,
    text: str,
    categories: dict,
    params: dict,
    use_ollama: bool = False,
) -> dict[str, float]:
    """Run score_motion with rhetorical parameters injected."""
    from swedish_parliament_policy_classifier.classifier.scorer import _detect_rhetorical_patterns

    # Temporarily override the function
    orig_func = _detect_rhetorical_patterns
    def _patched_detect(text: str) -> dict[str, float]:
        if not text:
            return {}
        text_lower = text.lower()
        adjustments = {cat: 0.0 for cat in CATEGORIES}
        for cat in CATEGORIES:
            base_key = f"base_{cat}"
            inc_key = f"inc_{cat}"
            base = params.get(base_key, 0.0)
            inc = params.get(inc_key, 0.0)
            signals = _get_signals_for_category(cat)
            count = sum(1 for s in signals if s in text_lower)
            if count >= 2:
                adjustments[cat] += base + (count * inc)
            elif count == 1:
                adjustments[cat] += base * 0.5
        return adjustments

    # Monkey-patch
    import swedish_parliament_policy_classifier.classifier.scorer as scorer_mod
    scorer_mod._detect_rhetorical_patterns = _patched_detect

    try:
        results = score_motion(
            motion_id=speech_id,
            text=text,
            categories=categories,
            party=None,
            embedding_matcher=None,  # disable embedding matcher for clean tuning
            use_zero_shot=False,     # disable zero-shot for clean tuning
            skip_policy_extraction=True,
            use_speech_preprocessing=True,
            use_ollama=use_ollama,
            ollama_weight=0.60,
        )
        return {r.category: float(r.normalized_weight) for r in results}
    finally:
        scorer_mod._detect_rhetorical_patterns = orig_func


def _get_signals_for_category(cat: str) -> list[str]:
    # Return the signal lists from the original function for each category
    # These are the substrings we search for in the text
    from swedish_parliament_policy_classifier.classifier.scorer import _detect_rhetorical_patterns
    import inspect, textwrap, ast
    source = inspect.getsource(_detect_rhetorical_patterns)
    # Simpler: hardcode the lists here to match the current implementation
    signals = {
        "far_left": [
            "kapitalism", "kapitalistisk", "kapitalisterna", "borgarklass",
            "klasskamp", "klassamhälle", "profit", "vinstintresse", "marknadsfundamentalism",
            "nedrusta", "avveckla försvaret", "imperialism", "anti-imperialistisk",
            "kollektivt ägande", "samhälligt ägande", "företagsdemokrati", "demokratiskt ägande",
            "revolution", "radikal förändring", "omstörta", "systemkritisk", "systemfel",
        ],
        "left": [
            "omfördelning", "progressiv beskattning", "höj skatten", "skattehöjning",
            "fackförening", "kollektivavtal", "anställningstrygghet", "las",
            "stärka välfärden", "bygga ut välfärden", "offentlig sektor",
            "stoppa vinster", "vinster i välfärden", "vinstförbud",
            "arbetstagare", "löntagare", "vanliga människor", "folkflertalet",
            "socioekonomisk", "fattigdom", "inkomstskillnader", "jämlikhet",
            "social rättvisa", "rättvisa", "solidaritet", "sammanhållning",
            "allmännytta", "allmännyttan", "bostad för alla",
        ],
        "centre_left": [
            "välfärd", "sociala tjänster", "hälso- och sjukvård", "sjukvård",
            "omsorg", "äldreomsorg", "barnomsorg", "förskola", "skola",
            "utbildning", "kompetensutveckling", "livslångt lärande",
            "miljö", "klimat", "hållbarhet", "biologisk mångfald",
            "socialdemokrati", "socialdemokratisk", "reform", "gradvis reform",
            "jämställdhet", "integration", "inkludering", "mänskliga rättigheter",
            "folkhälsa", "förebyggande", "demokrati", "jämlikhet",
            "aktiv arbetsmarknadspolitik", "arbetslöshetsbekämpning", "jobb för alla",
        ],
        "centre": [
            "båda sidor", "alla partier", "över blockgränsen", "samarbete",
            "kompromiss", "pragmatisk", "balanserad", "lagenlig", "rättssäker",
            "evidensbaserad", "fakta", "konkret förslag", "konkreta åtgärder",
            "effektiv", "resultat", "uppföljning", "utvärdering", "granskning",
            "riksrevisionen", "myndighet", "process", "beredning",
            "oberoende", "opartisk", "saklig", "trovärdig",
        ],
        "centre_right": [
            "marknad", "marknadsekonomi", "marknadslösning", "privat", "privata aktörer",
            "företag", "företagare", "entreprenörskap", "näringsliv", "industri",
            "tillväxt", "kompetitivitet", "innovation", "effektivisera", "effektivitet",
            "skattepolitik", "skattenivå", "beskattning", "skattetryck",
            "budget", "budgetdisciplin", "finanspolitik", "statsfinanser",
            "reform", "modernisera", "förenkla", "färre regleringar",
            "arbetslinjen", "sysselsättning", "arbetskraftsdeltagande",
        ],
        "right": [
            "sänka skatter", "skattesänkning", "lägre skatt", "dereglering", "avreglering",
            "privatisera", "privatisering", "utförsäljning", "nedläggning",
            "försvar", "försvarsallians", "nato", "säkerhetspolitik", "försvarspolitik",
            "lag och ordning", "straff", "kriminalitet", "brott", "rättsväsende",
            "tradition", "kulturarv", "svenska värderingar", "jämställdhet traditionell",
            "familj", "föräldraskap", "uppfostran", "skolplikt", "disciplin",
            "suveränitet", "nationell", "nationellt självbestämmande", "självständig",
            "eu-kritisk", "eu-kritik", "bryta med eu", "lämna eu", "eu-skeptisk",
            "motståndare till eu", "eus inflytande", "budgetramar", "budgetdisciplin",
            "konservativ", "bevara", "värna", "tuffare", "strängare",
            "minska byråkrati", "minska regleringar", "minska staten",
            "tuffare tag", "ordning och reda", "svenska intressen",
            "egna intressen", "nationella intressen", "svensk suveränitet",
            "minska invandring", "minska migration", "minska asyl",
            "sverigedemokrat", "sverigedemokraterna", "sd",
        ],
        "far_right": [
            "svenskhet", "svenska folket", "etnisk", "etnicitet", "kulturarv",
            "massinvandring", "invandring", "invandrare", "asyl", "migration",
            "integration misslyckad", "integration har misslyckats", "parallellsamhälle",
            "islam", "islamisering", "muslim", "muslimsk", "sharia", "extrem islam",
            "svenska värden", "västerländska värden", "jämställdhet hotad", "kvinnoförtryck",
            "gräns", "gränskontroll", "återvandring", "återvandra", "repatriering",
            "folkomröstning", "folkets vilja", "eliten", "etablissemanget", "pk-elit",
            "globalisering", "globalism", "internationalism", "fn", "förenta nationerna",
            "censur", "yttrandefrihet hotad", "demokrati i fara", "förrädare",
            "försvara sverige", "sverige först", "sverige åt svenskarna", "vårt land",
            "sverigedemokrat", "sverigedemokraterna", "sd",
        ],
    }
    return signals.get(cat, [])


def compute_loss(pred: dict[str, float], teacher: dict[str, float], metric: str = "mse") -> float:
    cats = CATEGORIES
    # Convert to Fraction for exact arithmetic where possible, falling back to float for metrics
    p = np.array([float(pred.get(c, 0.0)) for c in cats], dtype=np.float64)
    t = np.array([float(teacher.get(c, 0.0)) for c in cats], dtype=np.float64)
    if metric == "mse":
        return float(np.mean((p - t) ** 2))
    elif metric == "kl":
        # KL divergence D(t || p)
        eps = 1e-12
        p = np.clip(p, eps, 1.0)
        t = np.clip(t, eps, 1.0)
        return float(np.sum(t * np.log(t / p)))
    elif metric == "js":
        return float(jensenshannon(p, t) ** 2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def random_params() -> dict:
    """Sample random hyperparameters for the 7 rhetorical categories."""
    return {
        "base_far_left": random.uniform(0.2, 2.0),
        "base_left": random.uniform(0.2, 2.0),
        "base_centre_left": random.uniform(0.2, 2.0),
        "base_centre": random.uniform(0.2, 1.5),
        "base_centre_right": random.uniform(0.2, 2.0),
        "base_right": random.uniform(0.2, 2.0),
        "base_far_right": random.uniform(0.2, 2.0),
        "inc_far_left": random.uniform(0.05, 0.5),
        "inc_left": random.uniform(0.05, 0.5),
        "inc_centre_left": random.uniform(0.05, 0.5),
        "inc_centre": random.uniform(0.05, 0.4),
        "inc_centre_right": random.uniform(0.05, 0.5),
        "inc_right": random.uniform(0.05, 0.5),
        "inc_far_right": random.uniform(0.05, 0.5),
    }


def evaluate_params(
    params: dict,
    speeches_df: pd.DataFrame,
    teacher_labels: dict[str, dict[str, float]],
    categories: dict,
    metric: str = "mse",
    use_ollama: bool = False,
) -> float:
    losses = []
    for _, row in speeches_df.iterrows():
        speech_id = str(row["anforande_id"])
        text = row.get("anforandetext") or ""
        teacher = teacher_labels.get(speech_id)
        if not teacher:
            continue
        pred = score_with_params(speech_id, text, categories, params, use_ollama=use_ollama)
        loss = compute_loss(pred, teacher, metric=metric)
        losses.append(loss)
    return float(np.mean(losses)) if losses else float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="logs/ollama_teacher_labels.json")
    p.add_argument("--parquet-dir", default="data/speeches/parquet")
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--metric", default="mse", choices=["mse", "kl", "js"])
    p.add_argument("--out", default="models/rhetorical_weights_best.json")
    p.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-ollama", action="store_true", help="Also include ollama signal in pipeline during tuning")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    teacher_labels = load_teacher_labels(args.teacher)
    speech_ids = list(teacher_labels.keys())
    parquet_dir = Path(args.parquet_dir)
    speeches_df = load_speeches(speech_ids, parquet_dir)
    categories = load_definitions()

    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_experiment("rhetorical-weight-tuning")
            mlflow_run = mlflow.start_run()
        except Exception as e:
            print(f"MLflow unavailable: {e}", file=sys.stderr)

    best_loss = float("inf")
    best_params = None

    print(f"Starting random search: {args.n_iter} iterations, metric={args.metric}", file=sys.stderr)
    for i in range(args.n_iter):
        params = random_params()
        loss = evaluate_params(params, speeches_df, teacher_labels, categories, metric=args.metric, use_ollama=args.use_ollama)

        print(f"  Iter {i+1}/{args.n_iter}: loss={loss:.6f}", file=sys.stderr)

        if mlflow_run is not None:
            try:
                mlflow.log_params(params)
                mlflow.log_metric("loss", loss, step=i)
            except Exception:
                pass

        if loss < best_loss:
            best_loss = loss
            best_params = deepcopy(params)
            print(f"    *** New best loss: {best_loss:.6f}", file=sys.stderr)

    if mlflow_run is not None:
        try:
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_loss", best_loss)
            mlflow.end_run()
        except Exception:
            pass

    # Save best params
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"best_loss": best_loss, "params": best_params}, f, indent=2, ensure_ascii=False)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best params written to {out_path}")


if __name__ == "__main__":
    main()
