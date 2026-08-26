from scripts.apply_rhetorical_adjustments import apply_rhetorical_to_probs, detect_rhetorical_patterns


def test_rhetorical_adjustment_script_uses_current_detector() -> None:
    adjustments = detect_rhetorical_patterns("Vi vill sänka skatter och stärka försvaret.")

    adjusted = apply_rhetorical_to_probs(
        {"left": 0.5, "right": 0.5},
        {"left": adjustments.get("left", 0.0), "right": adjustments["right"]},
    )

    assert adjustments["right"] > 0
    assert adjusted["right"] > adjusted["left"]
    assert sum(adjusted.values()) == 1.0