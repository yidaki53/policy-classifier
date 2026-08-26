from scripts.audit_update_manifest import audit_manifest


def test_audit_manifest_rejects_nested_failed_subprocess() -> None:
    manifest = {
        "run_ts": "20260730T120000Z",
        "steps": {
            "classify": {
                "classify_speeches": {
                    "step": "classify_speeches_parquet",
                    "ok": False,
                    "returncode": 1,
                },
                "classify_motions": {
                    "step": "classify_motions",
                    "ok": True,
                    "returncode": 0,
                },
            }
        },
        "completed_at": "20260730T130000Z",
    }

    report = audit_manifest(manifest)

    assert report["status"] == "failed"
    assert report["failed_steps"] == [
        {
            "path": "steps.classify.classify_speeches",
            "step": "classify_speeches_parquet",
            "returncode": 1,
        }
    ]


def test_audit_manifest_rejects_top_level_pipeline_error() -> None:
    report = audit_manifest(
        {
            "run_ts": "20260730T120000Z",
            "steps": {},
            "error": "analysis stage crashed",
            "completed_at": "20260730T120500Z",
        }
    )

    assert report["status"] == "failed"
    assert report["manifest_errors"] == ["analysis stage crashed"]