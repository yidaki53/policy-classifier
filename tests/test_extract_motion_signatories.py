import pandas as pd

from scripts.extract_motion_signatories import _extract_rows_from_doc


def test_extract_rows_from_doc_handles_list_intressent():
    doc = {
        "dokumentstatus": {
            "dokument": {"dok_id": "H123", "rm": "2022/23"},
            "dokintressent": {
                "intressent": [
                    {
                        "intressent_id": "111",
                        "namn": "A A",
                        "partibet": "M",
                        "roll": "undertecknare",
                        "ordning": "1",
                    },
                    {
                        "intressent_id": "222",
                        "namn": "B B",
                        "partibet": "M",
                        "roll": "undertecknare",
                        "ordning": "2",
                    },
                ]
            },
        }
    }

    rows = _extract_rows_from_doc(doc)

    assert len(rows) == 2
    assert rows[0]["motion_id"] == "H123"
    assert rows[0]["intressent_id"] == "111"
    assert rows[1]["signatory_order"] == 2


def test_extract_rows_from_doc_handles_single_dict_intressent():
    doc = {
        "dokumentstatus": {
            "dokument": {"dok_id": "H999", "rm": "2018/19"},
            "dokintressent": {
                "intressent": {
                    "intressent_id": "333",
                    "namn": "C C",
                    "partibet": "SD",
                    "roll": "undertecknare",
                    "ordning": "1",
                }
            },
        }
    }

    rows = _extract_rows_from_doc(doc)

    assert len(rows) == 1
    assert rows[0]["motion_id"] == "H999"
    assert rows[0]["signatory_party"] == "SD"
    assert pd.notna(rows[0]["signatory_order"])
