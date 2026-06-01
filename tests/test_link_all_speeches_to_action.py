import pandas as pd

from scripts.link_all_speeches_to_action import (
    _build_graph_candidate_groups,
    _build_fallback_groups,
    _build_speech_link_table,
    _fallback_motion_for_speech,
    _graph_pick_candidate,
)


def test_fallback_prefers_vote_in_party_category():
    motions = pd.DataFrame(
        {
            "motion_id": ["m1", "m2", "m3"],
            "party": ["M", "M", "S"],
            "category": ["right", "right", "left"],
            "motion_weight": [0.7, 0.9, 0.8],
            "motion_date": pd.to_datetime(["2022-01-01", "2022-01-03", "2022-01-02"], utc=True),
            "has_vote": [False, True, True],
        }
    )
    speech = pd.Series(
        {
            "party": "M",
            "category": "right",
            "speech_date": pd.Timestamp("2022-01-04", tz="UTC"),
        }
    )

    groups = _build_fallback_groups(motions)
    best, source = _fallback_motion_for_speech(speech, groups)

    assert source == "party_category_vote"
    assert best["motion_id"] == "m2"
    assert bool(best["has_vote"]) is True


def test_build_speech_link_table_keeps_unclassified_speeches():
    speech_meta = pd.DataFrame(
        {
            "speech_id": ["s1", "s2", "s3"],
            "party": ["M", "S", "V"],
            "speech_date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"], utc=True),
        }
    )
    speech_top = pd.DataFrame(
        {
            "speech_id": ["s1", "s2"],
            "category": ["right", "left"],
            "speech_weight": [0.9, 0.8],
        }
    )

    out = _build_speech_link_table(speech_meta, speech_top)

    assert len(out) == 3
    assert set(out["speech_id"].tolist()) == {"s1", "s2", "s3"}
    assert out.loc[out["speech_id"] == "s3", "category"].iloc[0] == ""


def test_graph_pick_prefers_signatory_match_for_motion_channel():
    motions = pd.DataFrame(
        {
            "motion_id": ["m_vote", "m_motion"],
            "party": ["M", "M"],
            "category": ["right", "right"],
            "motion_weight": [0.7, 0.8],
            "motion_date": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "has_vote": [True, False],
            "intressent_id": ["other", "abc123"],
        }
    )

    groups = _build_graph_candidate_groups(motions)
    speech_row = pd.Series(
        {
            "party": "M",
            "category": "right",
            "speech_date": pd.Timestamp("2022-01-03", tz="UTC"),
            "intressent_id": "abc123",
        }
    )
    cand = _graph_pick_candidate(speech_row, groups, "motion")

    assert cand is not None
    assert cand["motion_id"] == "m_motion"
