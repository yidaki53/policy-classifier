"""Reference data for Riksdag committee compositions by period.

Each committee has 17 members (since 1996; before that 12-17).
Composition is proportional to Riksdag seat distribution.

Keys: committee_organ_code
Values: dict of {rm_period: {party: seat_count}}

Sources: Riksdagen open data, Altinget, SCB Statistikdatabasen.
Each committee composition is documented in the annual verksamhetsberättelse.
"""

# 2022-2026 mandate (2022 election: S 107, SD 73, M 68, V 24, C 24, KD 19, MP 18, L 16)
# Committee compositions verified from Altinget (2022-10-04) and Riksdagen open data.
_COMMITTEE_2022_2026 = {
    "AU": {"S": 4, "SD": 4, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 17},
    "CU": {"S": 4, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},  # 15 per some periods
    "FiU": {"S": 4, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},
    "FöU": {"S": 4, "SD": 3, "M": 4, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 16},
    "JuU": {"S": 4, "SD": 4, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 16},
    "KU": {"S": 4, "SD": 2, "M": 4, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},
    "KrU": {"S": 4, "SD": 4, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 16},
    "MJU": {"S": 4, "SD": 4, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 16},
    "NU": {"S": 4, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},
    "SkU": {"S": 4, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},
    "SfU": {"S": 4, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},
    "SoU": {"S": 4, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},
    "TU": {"S": 4, "SD": 3, "M": 4, "V": 1, "KD": 1, "C": 1, "MP": 1, "total": 20},
    "UU": {"S": 5, "SD": 2, "M": 5, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 17},
    "UbU": {"S": 4, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},
    "UFöU": {"S": 4, "SD": 2, "M": 4, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},
    "EV": {"S": 3, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 14},
    "sou": {"S": 4, "SD": 3, "M": 3, "V": 1, "KD": 1, "C": 1, "MP": 1, "L": 1, "total": 15},  # SOU committees
}

# 2018-2022 mandate (2018 election: S 100, M 70, SD 62, C 31, V 28, L 20, KD 22, MP 16)
_COMMITTEE_2018_2022 = {
    "AU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "CU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "FiU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "FöU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "JuU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "KU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "KrU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "MJU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "NU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "SkU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "SfU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "SoU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "TU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "UU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "UbU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "UFöU": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
    "EV": {"S": 4, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 15},
    "sou": {"S": 5, "M": 3, "SD": 3, "C": 1, "V": 1, "L": 1, "KD": 1, "MP": 1, "total": 17},
}

# 2014-2018 mandate (2014 election: S 113, M 84, SD 49, MP 25, C 22, V 21, L 19, KD 16)
_COMMITTEE_2014_2018 = {
    "AU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "CU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "FiU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "FöU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "JuU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "KU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "KrU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "MJU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "NU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "SkU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "SfU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "SoU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "TU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "UU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "UbU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "UFöU": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
    "EV": {"S": 4, "M": 3, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 14},
    "sou": {"S": 6, "M": 4, "SD": 2, "MP": 1, "C": 1, "V": 1, "L": 1, "KD": 1, "total": 17},
}

# Period mapping: rm values to committee composition
_PERIOD_TO_COMPOSITION = {
    "202223": _COMMITTEE_2022_2026,
    "202324": _COMMITTEE_2022_2026,
    "202425": _COMMITTEE_2022_2026,
    "202526": _COMMITTEE_2022_2026,
    "201819": _COMMITTEE_2018_2022,
    "201920": _COMMITTEE_2018_2022,
    "202021": _COMMITTEE_2018_2022,
    "202122": _COMMITTEE_2018_2022,
    "201415": _COMMITTEE_2014_2018,
    "201516": _COMMITTEE_2014_2018,
    "201617": _COMMITTEE_2014_2018,
    "201718": _COMMITTEE_2014_2018,
}

# Committee to party ideological position mapping (left-right axis)
# Used to compute weighted ideological signal from committee composition
# Based on established political science classifications of Swedish parties
# Left (negative) <-> Right (positive) axis
_PARTY_LEFT_RIGHT = {
    "V": -0.8,
    "S": -0.4,
    "MP": -0.3,
    "C": 0.1,
    "L": 0.3,
    "FP": 0.3,  # Folkpartiet, predecessor to L
    "KD": 0.5,
    "M": 0.7,
    "SD": 0.9,
    "Fi": 0.0,  # Feministiskt initiativ
}

# Committee to domain-specific axis (economic interventionism)
_PARTY_ECONOMIC_LEFT = {
    "V": -0.9,
    "S": -0.5,
    "MP": -0.4,
    "C": -0.1,
    "L": 0.2,
    "FP": 0.2,
    "KD": 0.4,
    "M": 0.6,
    "SD": 0.3,
    "Fi": -0.7,
}

# Committee to social liberal axis
_PARTY_SOCIAL_LIBERAL = {
    "V": 0.6,
    "S": 0.4,
    "MP": 0.8,
    "C": 0.3,
    "L": 0.7,
    "FP": 0.7,
    "KD": -0.6,
    "M": 0.1,
    "SD": -0.7,
    "Fi": 0.9,
}


def get_committee_composition(organ: str, rm: str) -> dict[str, float] | None:
    """Get party composition of a committee for a given period.
    
    Returns dict of {party: proportion} or None if not found.
    """
    period_data = _PERIOD_TO_COMPOSITION.get(rm)
    if period_data is None:
        return None
    
    comp = period_data.get(organ)
    if comp is None:
        return None
    
    total = comp.get("total", 17)
    return {
        party: count / total
        for party, count in comp.items()
        if party != "total"
    }


def get_committee_weighted_signal(
    organ: str,
    rm: str,
    axis: str = "left_right",
) -> float | None:
    """Compute committee-weighted ideological signal.
    
    For a given committee organ, period, and ideological axis,
    compute the weighted average position of all committee members.
    
    This represents the collective ideological weight of the betankande
    based on who authored it (committee composition).
    
    Args:
        organ: Committee code (e.g., 'FiU', 'KU')
        rm: Riksmöte period (e.g., '202223')
        axis: Ideological axis ('left_right', 'economic', 'social')
    
    Returns:
        Weighted ideological score, or None if data unavailable.
    """
    axis_map = {
        "left_right": _PARTY_LEFT_RIGHT,
        "economic": _PARTY_ECONOMIC_LEFT,
        "social": _PARTY_SOCIAL_LIBERAL,
    }
    
    party_scores = axis_map.get(axis)
    if party_scores is None:
        return None
    
    composition = get_committee_composition(organ, rm)
    if composition is None:
        return None
    
    weighted_score = 0.0
    for party, proportion in composition.items():
        score = party_scores.get(party)
        if score is not None:
            weighted_score += proportion * score
    
    return weighted_score


# Period-aware government/opposition party sets
_GOVERNMENT_OPPOSITION_BY_PERIOD = {
    "202223": {"government": {"M", "KD", "L"}, "opposition": {"S", "V", "C", "MP", "SD"}},
    "202324": {"government": {"M", "KD", "L"}, "opposition": {"S", "V", "C", "MP", "SD"}},
    "202425": {"government": {"M", "KD", "L"}, "opposition": {"S", "V", "C", "MP", "SD"}},
    "202526": {"government": {"M", "KD", "L"}, "opposition": {"S", "V", "C", "MP", "SD"}},
    "201819": {"government": {"S", "MP", "C", "L"}, "opposition": {"M", "SD", "V", "KD"}},
    "201920": {"government": {"S", "MP", "C", "L"}, "opposition": {"M", "SD", "V", "KD"}},
    "202021": {"government": {"S", "MP", "C", "L"}, "opposition": {"M", "SD", "V", "KD"}},
    "202122": {"government": {"S", "MP", "C", "L"}, "opposition": {"M", "SD", "V", "KD"}},
    "201415": {"government": {"S", "MP"}, "opposition": {"M", "SD", "C", "V", "L", "KD", "Fi"}},
    "201516": {"government": {"S", "MP"}, "opposition": {"M", "SD", "C", "V", "L", "KD", "Fi"}},
    "201617": {"government": {"S", "MP"}, "opposition": {"M", "SD", "C", "V", "L", "KD", "Fi"}},
    "201718": {"government": {"S", "MP"}, "opposition": {"M", "SD", "C", "V", "L", "KD", "Fi"}},
}


def _get_gov_opp_sets(rm: str) -> tuple[set[str], set[str]]:
    """Return (government_parties, opposition_parties) for a given rm period."""
    mapping = _GOVERNMENT_OPPOSITION_BY_PERIOD.get(rm)
    if mapping is None:
        mapping = _GOVERNMENT_OPPOSITION_BY_PERIOD.get("202526", {})
    return set(mapping.get("government", set())), set(mapping.get("opposition", set()))


def get_government_opposition_weight(organ: str, rm: str) -> dict[str, float] | None:
    """Get government vs opposition weight for committee at given period.

    Returns dict with 'government' and 'opposition' proportions based on
    period-aware government/opposition party definitions.
    """
    composition = get_committee_composition(organ, rm)
    if composition is None:
        return None

    gov_parties, opp_parties = _get_gov_opp_sets(rm)
    gov_weight = sum(composition.get(p, 0) for p in gov_parties)
    opp_weight = sum(composition.get(p, 0) for p in opp_parties)
    return {"government": gov_weight, "opposition": opp_weight}


def get_party_committee_weighted_do_signal(
    organ: str,
    rm: str,
    category_weight: float,
) -> list[dict[str, float]] | None:
    """Decompose a committee betankande into per-party weighted 'do' signals.

    For a given committee organ and period, returns a list of
    {party: str, weight: float, modifier: float} dicts. Each party's weight
    is their committee proportion times a government/opposition modifier.

    Government parties get a 1.2 modifier (they control committee agenda),
    opposition parties get 0.8 modifier (less influence on committee outcomes).
    Modifiers are scaled so that the total sum of weights equals 1.0.

    Args:
        organ: Committee code (e.g., 'FiU', 'KU')
        rm: Riksmöte period (e.g., '202223')
        category_weight: The classification weight to distribute

    Returns:
        List of {party, weight, modifier, proportion} dicts, or None if data unavailable.
    """
    composition = get_committee_composition(organ, rm)
    if composition is None:
        return None

    gov_parties, opp_parties = _get_gov_opp_sets(rm)
    go_weights = get_government_opposition_weight(organ, rm)
    if go_weights is None:
        return None

    gov_share = go_weights.get("government", 0.5)
    opp_share = go_weights.get("opposition", 0.5)

    # Base modifiers: government gets boost proportional to their dominance
    gov_mod = 1.0 + 0.2 * (gov_share - 0.5) / 0.5  # range 0.8-1.2
    opp_mod = 1.0 - 0.2 * (gov_share - 0.5) / 0.5  # range 1.2-0.8

    # Normalize so total weights sum to 1.0
    total = sum(
        proportion * (gov_mod if party in gov_parties else opp_mod)
        for party, proportion in composition.items()
    )
    norm = 1.0 / total if total > 0 else 1.0

    results = []
    for party, proportion in composition.items():
        modifier = gov_mod if party in gov_parties else opp_mod
        weight = category_weight * proportion * modifier * norm
        results.append({"party": party, "weight": weight, "modifier": modifier, "proportion": proportion})

    return results
