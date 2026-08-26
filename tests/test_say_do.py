from swedish_parliament_policy_classifier.analysis.say_do import (
    SayDoTransition,
    classify_say_do_transition,
)


def test_classify_say_do_transition_keeps_mismatch_directions_separate() -> None:
    assert classify_say_do_transition("support", "yes") is SayDoTransition.SUPPORT_FOLLOW_THROUGH
    assert classify_say_do_transition("support", "no") is SayDoTransition.SUPPORT_TO_NO
    assert classify_say_do_transition("oppose", "no") is SayDoTransition.OPPOSE_FOLLOW_THROUGH
    assert classify_say_do_transition("oppose", "yes") is SayDoTransition.OPPOSE_TO_YES
    assert classify_say_do_transition("support", "abstain") is SayDoTransition.NON_COMMITMENT
    assert classify_say_do_transition("support", "no_present") is SayDoTransition.NO_BEHAVIORAL_EVIDENCE