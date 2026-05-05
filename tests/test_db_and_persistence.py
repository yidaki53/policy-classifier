from swedish_parliament_policy_classifier.db.schema import init_db


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

    from swedish_parliament_policy_classifier.models.models import ClassificationResult
    from swedish_parliament_policy_classifier.classifier.persist import record_lineage, persist_classification
    from datetime import datetime

    cr = ClassificationResult(
        motion_id="m1",
        category="left",
        raw_score=1.0,
        normalized_weight=1.0,
        matched_rules=["omfördela"],
        classifier_version="test",
        created_at=datetime.utcnow(),
    )

    lid = record_lineage(conn, "normalized_motions", "m1", "test")
    pid = persist_classification(conn, cr, lid)

    cur.execute("SELECT COUNT(*) FROM classifications WHERE motion_id = ?", ("m1",))
    cnt = cur.fetchone()[0]
    assert cnt == 1
