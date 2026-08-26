"""Domain rules for comparing stated stance with recorded party choice."""

from __future__ import annotations

from enum import Enum


class SayDoTransition(str, Enum):
    """Mutually exclusive outcomes for one matched speech-decision pair."""

    SUPPORT_FOLLOW_THROUGH = "support_follow_through"
    SUPPORT_TO_NO = "support_to_no"
    OPPOSE_FOLLOW_THROUGH = "oppose_follow_through"
    OPPOSE_TO_YES = "oppose_to_yes"
    NON_COMMITMENT = "non_commitment"
    NO_BEHAVIORAL_EVIDENCE = "no_behavioral_evidence"
    SPLIT_PARTY_CHOICE = "split_party_choice"
    UNCLEAR_SPEECH_STANCE = "unclear_speech_stance"


def classify_say_do_transition(
    speech_stance: str,
    party_choice: str,
) -> SayDoTransition:
    """Classify one explicitly linked statement and party decision.

    A No vote is not treated as the ideological inverse of a proposal.  This
    function only evaluates whether the recorded choice follows the earlier
    support or opposition stance toward the same validated policy referent.
    """

    stance = str(speech_stance).strip().casefold()
    choice = str(party_choice).strip().casefold()

    if stance not in {"support", "oppose", "unclear"}:
        raise ValueError(f"Unsupported speech stance: {speech_stance!r}")
    if choice not in {"yes", "no", "abstain", "absent", "no_present", "split"}:
        raise ValueError(f"Unsupported party choice: {party_choice!r}")

    if choice in {"absent", "no_present"}:
        return SayDoTransition.NO_BEHAVIORAL_EVIDENCE
    if choice == "abstain":
        return SayDoTransition.NON_COMMITMENT
    if choice == "split":
        return SayDoTransition.SPLIT_PARTY_CHOICE
    if stance == "unclear":
        return SayDoTransition.UNCLEAR_SPEECH_STANCE

    transitions = {
        ("support", "yes"): SayDoTransition.SUPPORT_FOLLOW_THROUGH,
        ("support", "no"): SayDoTransition.SUPPORT_TO_NO,
        ("oppose", "no"): SayDoTransition.OPPOSE_FOLLOW_THROUGH,
        ("oppose", "yes"): SayDoTransition.OPPOSE_TO_YES,
    }
    return transitions[(stance, choice)]


__all__ = ["SayDoTransition", "classify_say_do_transition"]