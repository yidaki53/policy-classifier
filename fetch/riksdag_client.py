"""Small Riksdag client skeleton. Use `sample=True` for offline development.

This module provides a lightweight live fetch from the Riksdag open-data
JSON endpoint when `sample=False`. The remote schema can vary; the client
is defensive and falls back to the bundled sample data on any error.
"""

from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import requests
import logging

LOG = logging.getLogger(__name__)


_SAMPLE = [
    {
        "id": "sample-1",
        "title": "Sänk skatterna",
        "text": "Vi vill sänka skatter och privatisera tjänster",
        "date": "2025-01-01",
        "party": "Moderaterna",
    },
    {
        "id": "sample-2",
        "title": "Öka välfärden",
        "text": "Vi föreslår att öka resurser till vård och omsorg",
        "date": "2025-02-01",
        "party": "Vänsterpartiet",
    },
]


def _parse_riksdag_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    # Best-effort extraction of fields; the remote API has varied keys.
    docid = d.get("dok_id") or d.get("id") or d.get("dokument_id") or d.get("dok_id", None)
    title = d.get("titel") or d.get("rubrik") or d.get("titelengelska") or ""
    text = d.get("sammanfattning") or d.get("doktext") or d.get("text") or title or ""
    date = d.get("dok_datum") or d.get("datum") or d.get("publicerad") or None
    party = None
    # Some API variants store proposer/party info in nested structures
    if isinstance(d.get("parti"), str):
        party = d.get("parti")
    elif isinstance(d.get("fran"), str):
        party = d.get("fran")

    return {"id": str(docid), "title": title, "text": text, "date": date, "party": party}


def fetch_recent_motions(sample: bool = True, limit: int = 50, timeout: int = 10, query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch recent motions/proposals from the Riksdag API.

    If `sample` is True the bundled sample set is returned. When `sample` is
    False the function attempts to fetch JSON from the public Riksdag endpoint
    and returns a list of simplified motion dicts. On any error the function
    falls back to the sample data.
    """
    if sample:
        return _SAMPLE[:limit]

    base = "https://data.riksdagen.se/dokumentlista/"
    params = {"doktyp": "motion", "utformat": "json", "antal": limit}
    if query:
        params["sok"] = query

    url = f"{base}?{urlencode(params)}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        # Try several plausible paths to the document list
        docs = []
        if isinstance(payload, dict):
            if "dokumentlista" in payload:
                # typical shape: {"dokumentlista": {"dokument": [ ... ]}}
                dl = payload.get("dokumentlista") or {}
                docs = dl.get("dokument") or []
            elif "dokument" in payload:
                docs = payload.get("dokument") or []

        if not docs:
            LOG.debug("No documents found in Riksdag payload; returning sample")
            return _SAMPLE[:limit]

        out = []
        for d in docs[:limit]:
            out.append(_parse_riksdag_doc(d))

        return out
    except Exception as e:
        LOG.warning("Live Riksdag fetch failed: %s", e)
        return _SAMPLE[:limit]
