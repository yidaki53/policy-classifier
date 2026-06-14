#!/usr/bin/env python3
"""Apply tuned rhetorical weights and validate on held-out sample.

Usage:
    uv run python scripts/validate_tuned_weights.py --weights models/rhetorical_weights_best.json --sample logs/validation_sample_ids.txt --out logs/validation_results.json

Compares tuned pipeline output vs ollama teacher labels on held-out speeches.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from fractions import Fraction

from swedish_parliament_policy_classifier.exports import load_definitions
from swedish_parliament_policy_classifier.classifier.scorer import score_motion


CATEGORIES = ["far_left", "left", "centre_left", "centre", "centre_right", "right", "far_right"]


def load_weights(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["params"]


def load_speeches(speech_ids: list[str], parquet_dir: Path) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df[all_df["anforande_id"].isin(speech_ids)]


def call_ollama_teacher(text: str, model: str = "qwen2.5-coder-14b-32k:latest") -> Optional[dict]:
    """Get 7-dimension teacher labels from ollama."""
    try:
        import ollama
        system_prompt = """Du är en expert på svensk partipolitik. Analysera talretoriken och ge en sannolikhetsfördelning över 7 ideologiska dimensioner: far_left, left, centre_left, centre, centre_right, right, far_right. Output endast JSON med exakt dessa nycklar. Summan ska vara 1.0."""
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Klassificera:\n\n{text[:3000]}"},
            ],
            options={"temperature": 0.2, "num_predict": 256},
        )
        content = response["message"]["content"]
        import re
        match = re.search(r'\{[^}]*\}', content)
        if not match:
            return None
        obj = json.loads(match.group())
        out = {k: float(obj.get(k, 0.0)) for k in CATEGORIES}
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        return out
    except Exception as e:
        print(f"Ollama failed: {e}", file=sys.stderr)
        return None


def score_with_tuned_weights(
    speech_id: str,
    text: str,
    categories: dict,
    weights: dict,
) -> dict[str, float]:
    """Run score_motion with tuned rhetorical parameters."""
    from swedish_parliament_policy_classifier.classifier.scorer import _detect_rhetorical_patterns

    # Build patched function using tuned weights
    orig_func = _detect_rhetorical_patterns

    def _patched_detect(text: str) -> dict[str, float]:
        if not text:
            return {}
        text_lower = text.lower()
        adjustments = {cat: 0.0 for cat in CATEGORIES}
        for cat in CATEGORIES:
            base = weights.get(f"base_{cat}", 0.0)
            inc = weights.get(f"inc_{cat}", 0.0)
            # We need the signal lists - hardcode them here
            signals = _get_signals(cat)
            count = sum(1 for s in signals if s in text_lower)
            if count >= 2:
                adjustments[cat] += base + (count * inc)
            elif count == 1:
                adjustments[cat] += base * 0.5
        return adjustments

    import swedish_parliament_policy_classifier.classifier.scorer as scorer_mod
    scorer_mod._detect_rhetorical_patterns = _patched_detect

    try:
        results = score_motion(
            motion_id=speech_id,
            text=text,
            categories=categories,
            party=None,
            embedding_matcher=None,
            use_zero_shot=False,
            skip_policy_extraction=True,
            use_speech_preprocessing=True,
            use_ollama=False,
        )
        return {r.category: float(r.normalized_weight) for r in results}
    finally:
        scorer_mod._detect_rhetorical_patterns = orig_func


def _get_signals(cat: str) -> list[str]:
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


def compute_metrics(pred: dict, teacher: dict) -> dict:
    p = np.array([pred.get(c, 0.0) for c in CATEGORIES], dtype=np.float64)
    t = np.array([teacher.get(c, 0.0) for c in CATEGORIES], dtype=np.float64)
    mse = float(np.mean((p - t) ** 2))
    js = float(jensenshannon(p, t) ** 2)
    top_pred = CATEGORIES[np.argmax(p)]
    top_teacher = CATEGORIES[np.argmax(t)]
    return {"mse": mse, "js": js, "top_pred": top_pred, "top_teacher": top_teacher, "agrees": top_pred == top_teacher}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/rhetorical_weights_best.json")
    p.add_argument("--sample", default="logs/validation_sample_ids.txt")
    p.add_argument("--parquet-dir", default="data/speeches/parquet")
    p.add_argument("--out", default="logs/validation_results.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--teacher-labels", default=None, help="Pre-computed teacher labels JSON to avoid calling ollama")
    args = p.parse_args()

    weights = load_weights(args.weights)
    categories = load_definitions()

    with open(args.sample) as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    parquet_dir = Path(args.parquet_dir)
    df = load_speeches(sample_ids, parquet_dir)

    teacher_labels = {}
    if args.teacher_labels:
        with open(args.teacher_labels) as f:
            teacher_labels = {item["speech_id"]: item["scores"] for item in json.load(f)}

    results = []
    mse_list = []
    js_list = []
    agrees = 0
    total = 0

    for _, row in df.iterrows():
        speech_id = str(row["anforande_id"])
        text = row.get("anforandetext") or ""
        speaker = row.get("talare", "")
        party = row.get("parti", "")

        pred = score_with_tuned_weights(speech_id, text, categories, weights)

        teacher = teacher_labels.get(speech_id)
        if teacher is None:
            teacher = call_ollama_teacher(text)
            if teacher:
                teacher_labels[speech_id] = teacher

        if teacher is None:
            print(f"  {speaker} ({party}): no teacher label, skipping", file=sys.stderr)
            continue

        metrics = compute_metrics(pred, teacher)
        mse_list.append(metrics["mse"])
        js_list.append(metrics["js"])
        if metrics["agrees"]:
            agrees += 1
        total += 1

        results.append({
            "speech_id": speech_id,
            "speaker": speaker,
            "party": party,
            "pred": pred,
            "teacher": teacher,
            "metrics": metrics,
        })

        print(f"  {speaker} ({party}): pred={metrics['top_pred']}, teacher={metrics['top_teacher']}, MSE={metrics['mse']:.4f}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n": total,
                "mean_mse": float(np.mean(mse_list)) if mse_list else None,
                "mean_js": float(np.mean(js_list)) if js_list else None,
                "top1_accuracy": agrees / total if total > 0 else 0,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nValidation: n={total}, mean_mse={np.mean(mse_list):.4f}, mean_js={np.mean(js_list):.4f}, top1_acc={agrees/total:.3f}")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
