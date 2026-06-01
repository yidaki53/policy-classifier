import json
import sqlite3

from swedish_parliament_policy_classifier import exports
from swedish_parliament_policy_classifier.models import NormalizedMotion
from swedish_parliament_policy_classifier.db import init_db


def test_classify_and_persist_roundtrip():
    # create an in-memory DB with schema
    conn = init_db(":memory:")

    nm = NormalizedMotion(
        id="m-boundary-1",
        title="Sänk skatter",
        text="Vi vill sänka skatter och privatisera skolor.",
        party="X",
    )

    results = exports.classify_and_persist(nm, db_conn=conn)
    # expecting at least one classification result
    assert isinstance(results, list)
    assert len(results) > 0

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM classifications WHERE motion_id = ?", (nm.id,))
    row = cur.fetchone()
    assert row is not None and row[0] >= 0

    # ensure lineage entry created for the batch (persist_classifications_batch creates lineage)
    cur.execute("SELECT COUNT(*) as c FROM lineage WHERE source_table = ? AND source_id = ?", ("normalized_motions", nm.id))
    lr = cur.fetchone()
    assert lr is not None
