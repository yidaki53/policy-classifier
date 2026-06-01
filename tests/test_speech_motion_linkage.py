import re

import pandas as pd

from scripts.build_speech_motion_linkage import (
    _extract_motion_candidates,
    _is_election_runup_year,
    _load_betankande_motion_refs,
    _select_candidate_motion,
)


def test_extract_motion_candidates_prefers_rel_dok_id():
    token_re = re.compile(r"\b[A-Za-z0-9:/-]{3,30}\b")
    entry = {
        "rel_dok_id": "H501UbU27",
        "dok_id": "H2091",
        "title": "This title mentions a different code H999999",
        "text": "",
    }
    motion_lower_map = {"h501ubu27": "H501UbU27"}
    motion_aliases = {"h999999": "H999999"}

    candidates, sources = _extract_motion_candidates(entry, token_re, motion_lower_map, motion_aliases)

    assert candidates[0] == "H501UbU27"
    assert sources["H501UbU27"] == "rel_dok_id"


def test_extract_motion_candidates_falls_back_to_text():
    token_re = re.compile(r"\b[A-Za-z0-9:/-]{3,30}\b")
    entry = {
        "rel_dok_id": None,
        "dok_id": None,
        "title": "",
        "text": "The speech references motion H40312 directly.",
    }
    motion_lower_map = {}
    motion_aliases = {"h40312": "H40312"}

    candidates, sources = _extract_motion_candidates(entry, token_re, motion_lower_map, motion_aliases)

    assert candidates == ["H40312"]
    assert sources["H40312"] == "text"


def test_election_runup_year_helper():
    assert _is_election_runup_year(2021)
    assert _is_election_runup_year(2022)
    assert not _is_election_runup_year(2020)


def test_committee_bridge_prefers_motion_ref(tmp_path):
    bet_dir = tmp_path / "betankande"
    bet_dir.mkdir()
    committee = pd.DataFrame(
        {
            "dok_id": ["H201AU1"],
            "ref_dok_ids": ['["H501UbU27", "H999999"]'],
        }
    )
    committee.to_parquet(bet_dir / "bet-2010-2013.json.parquet", index=False)

    refs = _load_betankande_motion_refs(str(bet_dir))
    assert refs["H201AU1"] == ["H501UbU27", "H999999"]

    motions = pd.DataFrame(
        {
            "motion_id": ["H501UbU27", "H999999"],
            "party": ["M", "S"],
            "category": ["centre_right", "far_left"],
            "motion_weight": [0.9, 0.8],
            "motion_date_parsed": pd.to_datetime(["2012-01-01", "2012-01-05"], utc=True),
        }
    )

    chosen = _select_candidate_motion(refs["H201AU1"], "M", ["centre_right"], pd.Timestamp("2012-01-03", tz="UTC"), motions)

    assert chosen is not None
    assert chosen["motion_id"] == "H501UbU27"
