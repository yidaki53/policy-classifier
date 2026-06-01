"""Extract party affiliation from Riksdagen bulk JSON structures.

Bulk dataset JSON stores party affiliation in the signatory list under
``dokumentstatus.dokintressent.intressent[].partibet`` rather than in the
flat ``dokumentstatus.dokument.parti`` field.  This helper handles both the
single-dict and list-of-dict shapes of the ``intressent`` field and returns
the ``partibet`` of the primary signatory (``ordning=1`` or ``roll=undertecknare``).
"""


def extract_party_from_intressent(data: dict) -> str | None:
    """Return the party abbreviation from the first signatory if available.

    The function looks inside ``data`` for a nested ``dokumentstatus`` →
    ``dokintressent`` → ``intressent`` structure.  The ``intressent`` value can
    be a single ``dict`` or a ``list`` of ``dict``s.  It returns ``partibet``
    of the first signatory, otherwise ``None``.
    """
    if not isinstance(data, dict):
        return None

    ds = data.get("dokumentstatus")
    if not isinstance(ds, dict):
        return None

    di = ds.get("dokintressent")
    if not isinstance(di, dict):
        return None

    intressent = di.get("intressent")
    if isinstance(intressent, dict):
        intressent = [intressent]
    if not isinstance(intressent, list) or not intressent:
        return None

    # Prefer the primary signatory (ordning=1); fall back to first undertecknare.
    for sig in intressent:
        if isinstance(sig, dict):
            if str(sig.get("ordning", "")).strip() == "1":
                return _clean_partibet(sig.get("partibet"))

    for sig in intressent:
        if isinstance(sig, dict) and str(sig.get("roll", "")).strip().lower() == "undertecknare":
            return _clean_partibet(sig.get("partibet"))

    # Fallback to first signatory with any partibet
    for sig in intressent:
        if isinstance(sig, dict):
            pb = sig.get("partibet")
            if pb:
                return _clean_partibet(pb)

    return None


def _clean_partibet(pb):
    """Strip whitespace and return None for empty strings."""
    if isinstance(pb, str):
        pb = pb.strip()
        return pb if pb else None
    return None
