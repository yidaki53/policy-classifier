"""Rhetorical pattern detection for Swedish parliamentary speeches.

This module extracts ideological rhetorical pattern detection from the legacy
scorer to enable:
- Independent testing of rhetoric detection
- Swappable rhetoric models (rule-based → learned classifiers)
- Clean separation of concerns from signal combination logic

Pattern detection uses 7-dimension Britannica-derived ideological signals
with tunable weights loaded from ``models/rhetorical_weights_best.json``.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

LOG = logging.getLogger(__name__)

# Cache for loaded rhetorical weights
_RHETORICAL_WEIGHTS: Optional[Dict[str, float]] = None


def load_rhetorical_weights(weights_path: Optional[Path] = None) -> Dict[str, float]:
    """Load tuned rhetorical weights from disk or return defaults.
    
    Args:
        weights_path: Optional path to weights JSON file.
            Defaults to ``models/rhetorical_weights_best.json``.
    
    Returns:
        Dictionary mapping weight keys to float values.
    """
    global _RHETORICAL_WEIGHTS
    if _RHETORICAL_WEIGHTS is not None:
        return _RHETORICAL_WEIGHTS

    default = {
        "base_far_left": 1.20, "inc_far_left": 0.25,
        "base_left": 1.00, "inc_left": 0.20,
        "base_centre_left": 0.80, "inc_centre_left": 0.15,
        "base_centre": 0.60, "inc_centre": 0.10,
        "base_centre_right": 0.80, "inc_centre_right": 0.15,
        "base_right": 1.00, "inc_right": 0.20,
        "base_far_right": 1.20, "inc_far_right": 0.25,
    }
    
    if weights_path is None:
        weights_path = Path("models/rhetorical_weights_best.json")
    
    if weights_path.exists():
        try:
            with open(weights_path) as f:
                data = json.load(f)
            loaded = data.get("params", {})
            if loaded:
                default.update(loaded)
                print(f"Loaded tuned rhetorical weights from {weights_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load tuned weights: {e}, using defaults", file=sys.stderr)

    _RHETORICAL_WEIGHTS = default
    return default


def detect_rhetorical_patterns(text: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Detect ideological rhetorical patterns using 7-dimension signals.
    
    Returns per-category adjustment floats that are added to the combined
    classification score when speech preprocessing is active.
    
    Args:
        text: Input text to analyze.
        weights: Optional weight dictionary. If None, loads from default path.
    
    Returns:
        Dictionary mapping category names to adjustment floats.
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    adjustments = {
        "far_left": 0.0, "left": 0.0, "centre_left": 0.0,
        "centre": 0.0, "centre_right": 0.0, "right": 0.0, "far_right": 0.0
    }

    w = weights if weights is not None else load_rhetorical_weights()

    def _apply(cat: str, signals: List[str]) -> None:
        count = sum(1 for s in signals if s in text_lower)
        base = w.get(f"base_{cat}", 0.0)
        inc = w.get(f"inc_{cat}", 0.0)
        if count >= 2:
            adjustments[cat] += base + (count * inc)
        elif count == 1:
            adjustments[cat] += base * 0.5

    # ─── FAR LEFT ───
    _apply("far_left", [
        "kapitalism", "kapitalistisk", "kapitalisterna", "borgarklass",
        "klasskamp", "klassamhälle", "profit", "vinstintresse", "marknadsfundamentalism",
        "nedrusta", "avveckla försvaret", "imperialism", "anti-imperialistisk",
        "kollektivt ägande", "samhälligt ägande", "företagsdemokrati", "demokratiskt ägande",
        "revolution", "radikal förändring", "omstörta", "systemkritisk", "systemfel",
    ])

    # ─── LEFT ───
    _apply("left", [
        "omfördelning", "progressiv beskattning", "höj skatten", "skattehöjning",
        "fackförening", "kollektivavtal", "anställningstrygghet", "las",
        "stärka välfärden", "bygga ut välfärden", "offentlig sektor",
        "stoppa vinster", "vinster i välfärden", "vinstförbud",
        "arbetstagare", "löntagare", "vanliga människor", "folkflertalet",
        "socioekonomisk", "fattigdom", "inkomstskillnader", "jämlikhet",
        "social rättvisa", "rättvisa", "solidaritet", "sammanhållning",
        "allmännytta", "allmännyttan", "bostad för alla",
    ])

    # ─── CENTRE LEFT ───
    _apply("centre_left", [
        "välfärd", "sociala tjänster", "hälso- och sjukvård", "sjukvård",
        "omsorg", "äldreomsorg", "barnomsorg", "förskola", "skola",
        "utbildning", "kompetensutveckling", "livslångt lärande",
        "miljö", "klimat", "hållbarhet", "biologisk mångfald",
        "socialdemokrati", "socialdemokratisk", "reform", "gradvis reform",
        "jämställdhet", "integration", "inkludering", "mänskliga rättigheter",
        "folkhälsa", "förebyggande", "demokrati", "jämlikhet",
        "aktiv arbetsmarknadspolitik", "arbetslöshetsbekämpning", "jobb för alla",
    ])

    # ─── CENTRE ───
    _apply("centre", [
        "båda sidor", "alla partier", "över blockgränsen", "samarbete",
        "kompromiss", "pragmatisk", "balanserad", "lagenlig", "rättssäker",
        "evidensbaserad", "fakta", "konkret förslag", "konkreta åtgärder",
        "effektiv", "resultat", "uppföljning", "utvärdering", "granskning",
        "riksrevisionen", "myndighet", "process", "beredning",
        "oberoende", "opartisk", "saklig", "trovärdig",
    ])

    # ─── CENTRE RIGHT ───
    _apply("centre_right", [
        "marknad", "marknadsekonomi", "marknadslösning", "privat", "privata aktörer",
        "företag", "företagare", "entreprenörskap", "näringsliv", "industri",
        "tillväxt", "kompetitivitet", "innovation", "effektivisera", "effektivitet",
        "skattepolitik", "skattenivå", "beskattning", "skattetryck",
        "budget", "budgetdisciplin", "finanspolitik", "statsfinanser",
        "reform", "modernisera", "förenkla", "färre regleringar",
        "arbetslinjen", "sysselsättning", "arbetskraftsdeltagande",
    ])

    # ─── RIGHT ───
    _apply("right", [
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
    ])

    # ─── FAR RIGHT ───
    _apply("far_right", [
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
    ])

    # ─── COMPOUND PATTERNS ───
    env_terms = ["miljö", "natura 2000", "biologisk mångfald", "naturvård"]
    extraction_terms = ["gruva", "gruvdrift", "malmbrytning"]
    pro_industry = ["ja till", "positivt", "möjliggöra"]
    if any(t in text_lower for t in env_terms) and any(t in text_lower for t in extraction_terms):
        if any(t in text_lower for t in pro_industry):
            adjustments["right"] += 0.80
            adjustments["far_right"] += 0.40
            adjustments["centre_right"] += 0.40
    
    healthcare_terms = ["sjukvård", "vård", "missbruksvård"]
    privatization_terms = ["privat", "privata aktörer", "företag", "marknad"]
    if any(t in text_lower for t in healthcare_terms) and any(t in text_lower for t in privatization_terms):
        adjustments["right"] += 0.60
        adjustments["centre_right"] += 0.30
    
    eu_budget_terms = ["eu-budget", "eu:s budget", "gemensam budget", "strukturfond", "budgetram", "budgetdisciplin"]
    conservative_terms = ["konservativ", "minska", "minskning", "effektivisera", "kritisk", "motsätter"]
    if any(t in text_lower for t in eu_budget_terms) and any(t in text_lower for t in conservative_terms):
        adjustments["right"] += 0.80
        adjustments["far_right"] += 0.40

    return adjustments


def detect_rhetorical_patterns_with_metadata(text: str) -> Dict[str, any]:
    """Detect rhetorical patterns and return adjustments plus metadata.
    
    Metadata includes:
    - total_signals: count of matched signal terms
    - dominant_category: category with highest adjustment
    - is_rhetorical: whether any adjustment is non-zero
    
    Args:
        text: Input text to analyze.
    
    Returns:
        Dictionary with 'adjustments', 'total_signals', 'dominant_category',
        and 'is_rhetorical' keys.
    """
    adjustments = detect_rhetorical_patterns(text)
    
    total = sum(1 for v in adjustments.values() if v != 0.0)
    dominant = max(adjustments.items(), key=lambda x: x[1])[0] if total > 0 else None
    
    return {
        "adjustments": adjustments,
        "total_signals": total,
        "dominant_category": dominant,
        "is_rhetorical": total > 0,
    }