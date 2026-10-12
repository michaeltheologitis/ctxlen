"""The latest release's _data.json for a live lookup: from memory, else this machine's cache,
else GitHub, else the bundled copy."""

from __future__ import annotations

import _thread
import contextlib
import os
import time
from typing import Any, NamedTuple

from ctxlen import _resolve

URL = "https://raw.githubusercontent.com/michaeltheologitis/ctxlen/main/src/ctxlen/_data.json"
USER_AGENT = "ctxlen (+https://github.com/michaeltheologitis/ctxlen)"
TTL = 3600.0  # seconds a fetch, or a failed fetch, stands for
DEADLINE = 2.0  # the most a lookup waits for a fetch, and httpx's timeout


class Memo(NamedTuple):
    at: float  # time.time() when the fetch this index stands for began
    index: _resolve.Index  # its data's index, or the bundled one


MEMO: Memo | None = None  # this process's answer; None in a new process
LOCK = _thread.allocate_lock()  # many threads, one fetch


def renew_lock() -> None:
    """A fresh LOCK: run in a forked child, where a fetch in the parent would never release it."""
    global LOCK
    LOCK = _thread.allocate_lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=renew_lock)


def index() -> _resolve.Index:
    """The index a live lookup resolves against. Never raises. Waits at most about DEADLINE,
    at most once per TTL per machine (once per TTL per process when the cache is unusable)."""
    global MEMO
    if offline():
        return _resolve.shipped_index()
    with LOCK:
        if MEMO is None or not fresh(MEMO.at):
            at, data = read_cache()
            if at is None or not fresh(at):
                at, data = time.time(), fetch()
                write_cache(at, data)
            MEMO = Memo(at, usable(data) or _resolve.shipped_index())
        return MEMO.index


def offline() -> bool:
    """CTXLEN_OFFLINE is set to anything but "" or "0". Read on every live lookup."""
    return os.environ.get("CTXLEN_OFFLINE", "") not in ("", "0")


def fresh(at: float) -> bool:
    """0 <= time.time() - at < TTL: under an hour old, and a clock set back counts as stale."""
    return 0 <= time.time() - at < TTL


def usable(data: Any) -> _resolve.Index | None:
    """build_index(data) when data is a dict whose "format" equals FORMAT and build_index
    accepts it; None otherwise, never raising."""
    if not isinstance(data, dict) or data.get("format") != _resolve.FORMAT:
        return None
    try:
        return _resolve.build_index(data)
    except Exception:  # noqa: BLE001 (data build_index rejects means the bundled copy)
        return None


def cache_path() -> str | None:
    """<base>/ctxlen/data-<FORMAT>.json, where <base> is $XDG_CACHE_HOME when it is not blank
    (used as given), else ~/.cache, on every OS. None when that falls back to a home
    directory that os.path.expanduser cannot find (it returns "~")."""
    base = os.environ.get("XDG_CACHE_HOME", "")
    if not base.strip():
        home = os.path.expanduser("~")
        if home == "~":
            return None
        base = os.path.join(home, ".cache")
    return os.path.join(base, "ctxlen", f"data-{_resolve.FORMAT}.json")


def read_cache() -> tuple[float | None, Any]:
    """(fetched_at, data) from the cache file. (None, None) when the path is None, or the file
    is missing, unreadable, not JSON, not an object, or lacks "fetched_at" (an int or float,
    not a bool) or "data". Never raises. fetched_at is returned as a float."""
    import json

    path = cache_path()
    if path is None:
        return None, None
    try:
        with open(path, encoding="utf-8") as file:
            cached = json.load(file)
    except Exception:  # noqa: BLE001 (an unreadable cache is fetched again)
        return None, None
    if not isinstance(cached, dict) or "data" not in cached:
        return None, None
    at = cached.get("fetched_at")
    if type(at) not in (int, float):
        return None, None
    return float(at), cached["data"]


def write_cache(at: float, data: Any) -> None:
    """Writes {"fetched_at": at, "data": data} to cache_path(): makedirs, a temp file
    .data-<FORMAT>-*.tmp from tempfile.mkstemp in the same directory, json.dump with
    separators (",", ":"), then os.replace. Any failure leaves the previous file as it was and
    removes the temp file if one was made. Never raises; a None path writes nothing."""
    path = cache_path()
    if path is None:
        return
    import json
    import tempfile

    directory, temp = os.path.dirname(path), None
    try:
        os.makedirs(directory, exist_ok=True)
        handle, temp = tempfile.mkstemp(
            prefix=f".data-{_resolve.FORMAT}-", suffix=".tmp", dir=directory
        )
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump({"fetched_at": at, "data": data}, file, separators=(",", ":"))
        os.replace(temp, path)
    except Exception:  # noqa: BLE001 (an unwritable cache means memory only)
        if temp is not None:
            with contextlib.suppress(OSError):
                os.remove(temp)


def fetch() -> Any:
    """download()'s answer, from a daemon worker thread joined for DEADLINE seconds; None if
    the worker has not answered by then, or cannot start (at interpreter shutdown, for
    one). A worker that misses the deadline runs on and its answer is dropped. Never raises."""
    import threading

    answers = []
    try:
        worker = threading.Thread(
            target=lambda: answers.append(download()), name="ctxlen-fetch", daemon=True
        )
        worker.start()
        worker.join(DEADLINE)
    except Exception:  # noqa: BLE001 (a worker that cannot start means the bundled copy)
        return None
    return answers[0] if answers else None


def download() -> Any:
    """URL's body as JSON, or None on any failure (httpx or certifi missing, DNS, refused,
    proxy, TLS, timeout, any status but 2xx after redirects, bad gzip, not JSON)."""
    try:
        import ssl

        import certifi
        import httpx

        trust = ssl.create_default_context()  # the system's CAs, and SSL_CERT_FILE/DIR
        trust.load_verify_locations(certifi.where())  # plus certifi's
        response = httpx.get(
            URL,
            headers={"User-Agent": USER_AGENT},
            timeout=DEADLINE,
            verify=trust,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.json()  # httpx un-gzips
    except Exception:  # noqa: BLE001 (every failure means the bundled copy)
        return None
