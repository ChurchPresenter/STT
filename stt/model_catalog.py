"""Keeping the shipped model catalogue reachable once a machine has cached one.

The Model Manager's faster-whisper list is a catalogue of name -> repo. It is
cached to ``config/faster_whisper_models.json`` so the list survives an offline
start, and the Refresh button replaces that file with whatever it discovers on
the Hub.

The cache was absolute: ``get_faster_whisper_models_list`` returned it whenever
the file existed, with no expiry, so once a machine had written one the shipped
catalogue was never consulted again. That is fine until a repo moves — and one
did. ``Systran/faster-whisper-large-v3-turbo`` was withdrawn, and every install
that had run Refresh while it existed kept offering it: listed, selectable,
downloadable, and answering 401 on every attempt. No code change could correct
it, because the correction lived in a dict the machine had stopped reading.

So the shipped catalogue becomes a *floor* rather than a fallback. Whatever the
cache or a discovery run says, the entries the running code ships are present and
have the repo the running code names. Discovery may still add — that is what it
is for, and a model found on the Hub is real whether or not this release knew
about it — but it may no longer silently drop a model we ship, or pin one to an
address we have since corrected.

Stdlib only; the caller owns the file IO.
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping


def merge_catalog(
    shipped: Mapping[str, Mapping[str, str]],
    discovered: Mapping[str, Mapping[str, str]],
) -> Dict[str, Dict[str, str]]:
    """Combine a discovered/cached catalogue with the one this release ships.

    Entries only ``discovered`` knows about are kept — a newer Hub than this
    release is a good reason to offer more models, not fewer. Entries ``shipped``
    knows about are taken from ``shipped``, because that is the copy a code
    update can fix and the one whose repo address has been checked.

    Non-dict entries in ``discovered`` (a hand-edited or truncated cache file)
    are dropped rather than propagated into the UI.
    """
    merged: MutableMapping[str, Dict[str, str]] = {}
    for name, entry in discovered.items():
        if isinstance(entry, Mapping):
            merged[str(name)] = dict(entry)
    for name, entry in shipped.items():
        merged[str(name)] = dict(entry)
    return dict(merged)


def catalog_repos(catalog: Mapping[str, Mapping[str, str]]) -> Dict[str, str]:
    """``{model name: repo id}`` for the entries that name a repo.

    Used by the check that the shipped dict and the shipped seed file cannot
    drift apart again — they disagreed in both directions before this, which is
    how one of them ended up holding a dead address nobody noticed.
    """
    return {
        name: str(entry["repo"])
        for name, entry in catalog.items()
        if isinstance(entry, Mapping) and entry.get("repo")
    }
