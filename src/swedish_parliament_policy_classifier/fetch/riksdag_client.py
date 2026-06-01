"""Small Riksdag client skeleton. Use `sample=True` for offline development.

This module provides a lightweight live fetch from the Riksdag open-data
JSON endpoint when `sample=False`. The remote schema can vary; the client
is defensive and falls back to the bundled sample data on any error.
"""

from typing import List, Dict, Any, Optional, Tuple
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


import re as _re_module

def _parse_riksdag_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    # Best-effort extraction of fields; the remote API has varied keys.
    docid = d.get("dok_id") or d.get("id") or d.get("dokument_id") or d.get("dok_id", None)
    title = d.get("titel") or d.get("rubrik") or d.get("titelengelska") or ""
    undertitel = d.get("undertitel") or ""
    # Use undertitel as text if no summary is available (often richer than titel)
    text = d.get("sammanfattning") or d.get("doktext") or d.get("text") or undertitel or title or ""
    date = d.get("dok_datum") or d.get("datum") or d.get("publicerad") or None
    party = None
    # Some API variants store proposer/party info in nested structures
    if isinstance(d.get("parti"), str):
        party = d.get("parti")
    elif isinstance(d.get("fran"), str):
        party = d.get("fran")

    # Fallback: parse party from undertitel (e.g. "av Rebecka Le Moine m.fl. (MP)")
    if party is None:
        m = _re_module.search(r'\(([A-Z]+)\)\s*$', undertitel)
        if m:
            party = m.group(1)

    # Full-text URL for fetching motion body text (XML with HTML content)
    url_text = d.get("dokument_url_text") or d.get("url_text") or ""
    # The API returns protocol-relative URLs like //data.riksdagen.se/... 
    if url_text.startswith("//"):
        url_text = "https:" + url_text

    return {"id": str(docid), "title": title, "text": text, "date": date, "party": party, "undertitel": undertitel, "full_text_url": url_text}


def fetch_page(
    doktyp: str = "mot",
    page: int = 1,
    query: Optional[str] = None,
    timeout: int = 10,
    retries: int = 2,
    retry_delay: float = 1.0,
    fr: Optional[str] = None,
    till: Optional[str] = None,
    sort: Optional[str] = None,
    sortorder: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch a single page from the Riksdag API.

    Returns (list_of_docs, has_more_pages).  Retries on transient errors.

    Args:
        fr: Start date (inclusive) as YYYY-MM-DD.
        till: End date (inclusive) as YYYY-MM-DD.
        sort: Sort field (e.g. 'datum').
        sortorder: 'asc' or 'desc'.
    """
    import time
    base = "https://data.riksdagen.se/dokumentlista/"
    params: Dict[str, Any] = {"doktyp": doktyp, "utformat": "json", "antal": 200, "p": page}
    if query:
        params["sok"] = query
    if fr:
        params["fr"] = fr
    if till:
        params["till"] = till
    if sort:
        params["sort"] = sort
    if sortorder:
        params["sortorder"] = sortorder

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(base, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            docs = []
            has_more = False
            if isinstance(payload, dict):
                if "dokumentlista" in payload:
                    dl = payload.get("dokumentlista") or {}
                    docs = dl.get("dokument") or []
                    # Check pagination metadata
                    sidor = dl.get("@sidor")
                    if sidor is not None:
                        try:
                            total_pages = int(sidor)
                            has_more = page < total_pages
                        except (ValueError, TypeError):
                            has_more = len(docs) > 0
                    else:
                        has_more = len(docs) > 0
                elif "dokument" in payload:
                    docs = payload.get("dokument") or []
                    has_more = len(docs) > 0

            out = []
            for d in docs:
                parsed = _parse_riksdag_doc(d)
                parsed["doc_type"] = doktyp
                out.append(parsed)
            return out, has_more
        except Exception as e:
            last_err = e
            LOG.warning("Live Riksdag fetch attempt %s/%s failed on page %s: %s", attempt + 1, retries + 1, page, e)
            if attempt < retries:
                time.sleep(retry_delay * (attempt + 1))

    LOG.warning("Live Riksdag fetch failed after %s retries on page %s: %s", retries + 1, page, last_err)
    return [], False


def fetch_recent_motions(
    sample: bool = True,
    limit: int = 50,
    timeout: int = 10,
    query: Optional[str] = None,
    doktyp: str = "mot",
    parties: Optional[list[str]] = None,
    pages: int = 1,
) -> List[Dict[str, Any]]:
    """Fetch recent motions/proposals from the Riksdag API.

    If `sample` is True the bundled sample set is returned. When `sample` is
    False the function attempts to fetch JSON from the public Riksdag endpoint
    and returns a list of simplified motion dicts. On any error the function
    falls back to the sample data.

    The API returns ~20 results per page regardless of `limit`. Use `pages`
    to fetch multiple pages.
    """
    if sample:
        return _SAMPLE[:limit]

    all_docs: List[Dict[str, Any]] = []
    for page in range(1, pages + 1):
        docs, has_more = fetch_page(doktyp=doktyp, page=page, query=query, timeout=timeout)
        for d in docs:
            if parties and d.get("party") not in parties:
                continue
            all_docs.append(d)
            if len(all_docs) >= limit:
                return all_docs[:limit]
        if not has_more or not docs:
            break

    if not all_docs:
        return _SAMPLE[:limit]
    return all_docs[:limit]
