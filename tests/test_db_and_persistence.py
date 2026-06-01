from swedish_parliament_policy_classifier.exports import init_db, ClassificationResult, record_lineage, persist_classification


def test_init_db_and_persist(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    cur = conn.cursor()

    # sanity: expected tables exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_motions'")
    assert cur.fetchone() is not None

    # insert a normalized motion to satisfy FK
    cur.execute("INSERT INTO normalized_motions (id, title, text, party) VALUES (?, ?, ?, ?)", ("m1", "t", "txt", "TestParty"))
    conn.commit()

    # local imports (kept for test isolation if needed)
    # use attribute-style calls so static analysis can find links
    from datetime import datetime, timezone

    cr = ClassificationResult(
        motion_id="m1",
        category="left",
        raw_score=1.0,
        normalized_weight=1.0,
        matched_rules=["omfördela"],
        classifier_version="test",
        created_at=datetime.now(timezone.utc),
    )

    lid = record_lineage(conn, "normalized_motions", "m1", "test")
    pid = persist_classification(conn, cr, lid)

    # (removed Graphify import-hint)

    cur.execute("SELECT COUNT(*) FROM classifications WHERE motion_id = ?", ("m1",))
    cnt = cur.fetchone()[0]
    assert cnt == 1
