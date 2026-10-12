"""The live path, memory then the cache file then the fetch then the bundled copy, with nothing
leaving the machine: the fetch is faked, or served from 127.0.0.1."""

import contextlib
import json
import multiprocessing
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from gzip import compress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import certifi
import httpx  # imported here, so the first test's deadline does not pay for it
import pytest

import ctxlen
from ctxlen import _latest
from ctxlen._resolve import FORMAT
from pipeline.build import SITECUSTOMIZE
from tests.conftest import FIXTURE

FUNCTIONS = [ctxlen.ctxlen, ctxlen.max_input, ctxlen.max_output]
ONLY_LIVE = "ctxlen-test-only-live"  # an id the shipped data cannot have
LATEST = {
    "format": FORMAT,
    **FIXTURE,
    "rows": {**FIXTURE["rows"], ONLY_LIVE: ["openai", 900000, 128000, 1050000]},
}
BUNDLED = ctxlen.max_input("gpt-4o", live=False)
GZIP = {"Content-Encoding": "gzip"}
# argv: "live" or "bundled", then model ids, each looked up with the three functions.
GUARDED = """import json, socket, sys
import sitecustomize
import ctxlen
from ctxlen import _resolve
live, models = sys.argv[1:2] == ["live"], sys.argv[2:]
answers = [function(model, live=live) for function in (ctxlen.ctxlen, ctxlen.max_input, ctxlen.max_output)
           for model in models]
report = {"attempts": len(sitecustomize.ATTEMPTS), "answers": answers,
          "loaded": [name for name in ("ctxlen._latest", "httpx", "ssl", "tempfile", "threading")
                     if name in sys.modules],
          "bundled_read": _resolve.shipped_index.cache_info().currsize > 0}
try:
    socket.create_connection(("127.0.0.1", 9), timeout=1)
    report["guarded"] = False
except OSError:
    report["guarded"] = len(sitecustomize.ATTEMPTS) == report["attempts"] + 1
print(json.dumps(report))
"""


def closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(autouse=True)
def online(tmp_path, monkeypatch):
    """A new process with CTXLEN_OFFLINE unset and its cache in tmp_path, whose fetch is
    refused unless a test fakes it: nothing reaches GitHub."""
    monkeypatch.delenv("CTXLEN_OFFLINE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setattr(_latest, "MEMO", None)
    monkeypatch.setattr(_latest, "URL", f"http://127.0.0.1:{closed_port()}/_data.json")


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "cache" / "ctxlen" / "data-1.json"


@pytest.fixture
def clock(monkeypatch):
    """_latest's clock, moved by assignment: clock.now += 59 * 60."""
    fake = types.SimpleNamespace(now=1_791_769_787.5)
    fake.time = lambda: fake.now
    monkeypatch.setattr(_latest, "time", fake)
    return fake


@pytest.fixture
def downloads(monkeypatch):
    """download(), faked: a Mock answering LATEST, whose call_count counts the fetches."""
    fake = mock.Mock(return_value=LATEST)
    monkeypatch.setattr(_latest, "download", fake)
    return fake


class Server(ThreadingHTTPServer):
    """An HTTP server on 127.0.0.1. reply() sets what a GET of each path gets, and requests
    holds the handler of every request that arrived, with its .path and .headers."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.replies, self.requests = {}, []

    def reply(
        self, body, status=200, gzip=True, headers=None, delay=0.0, path="/_data.json"
    ):
        """`body` is bytes, or a value sent as its JSON; gzip=True compresses it and says so."""
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers = {**(GZIP if gzip else {}), **(headers or {})}
        payload = compress(payload) if gzip else payload
        self.replies[path] = (status, headers, payload, delay)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append(self)
        status, headers, payload, delay = self.server.replies[self.path]
        time.sleep(delay)
        with contextlib.suppress(OSError):  # the client hung up at its deadline
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(online, monkeypatch):
    """A Server, serving in a daemon thread, that _latest.URL points at."""
    fake = Server()
    serving = threading.Thread(
        target=fake.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    serving.start()
    monkeypatch.setattr(
        _latest, "URL", f"http://127.0.0.1:{fake.server_port}/_data.json"
    )
    yield fake
    fake.shutdown()
    fake.server_close()


def guarded(tmp_path, *argv, **environment):
    """GUARDED's report from a new interpreter that counts and refuses every connection."""
    guard = tmp_path / "guard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(SITECUSTOMIZE)
    path = os.pathsep.join(filter(None, [str(guard), os.environ.get("PYTHONPATH")]))
    done = subprocess.run(
        [sys.executable, "-c", GUARDED, *argv],
        env={**os.environ, "PYTHONPATH": path, **environment},
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(done.stdout)
    assert report.pop("guarded") is True
    return report


def bundled(*models):
    return [function(model, live=False) for function in FUNCTIONS for model in models]


def new_process():
    _latest.MEMO = None


# L1 · import and the off switches stay offline


# (argv, environment, modules it must not load); a live=False lookup reads the bundled copy
# through importlib.resources, which loads tempfile and threading
OFF_SWITCHES = [
    ((), {}, ["ctxlen._latest", "httpx", "ssl", "tempfile", "threading"]),
    (("bundled", "gpt-4o", "not-a-model"), {}, ["ctxlen._latest", "httpx", "ssl"]),
    (("live", "gpt-4o", "not-a-model"), {"CTXLEN_OFFLINE": "1"}, ["httpx", "ssl"]),
]


@pytest.mark.parametrize(
    "argv, environment, unloaded",
    OFF_SWITCHES,
    ids=["import", "live_false", "ctxlen_offline"],
)
def test_import_and_the_off_switches_open_no_socket_and_load_no_httpx(
    argv, environment, unloaded, tmp_path
):
    report = guarded(tmp_path, *argv, **environment)
    assert report["attempts"] == 0
    assert set(report["loaded"]).isdisjoint(unloaded)
    assert report["answers"] == bundled(*argv[1:])


def test_a_fresh_cache_answers_with_no_socket_and_no_httpx(tmp_path, cache_file):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(json.dumps({"fetched_at": time.time(), "data": LATEST}))
    report = guarded(tmp_path, "live", ONLY_LIVE)
    assert report == {
        "attempts": 0,
        "answers": [1050000, 900000, 128000],
        "loaded": ["ctxlen._latest"],
        "bundled_read": False,
    }


@pytest.mark.parametrize(
    "value, is_offline",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("1", True),
        ("true", True),
        ("no", True),
    ],
)
def test_ctxlen_offline_is_on_for_any_value_but_empty_or_zero(
    value, is_offline, monkeypatch
):
    if value is not None:
        monkeypatch.setenv("CTXLEN_OFFLINE", value)
    assert _latest.offline() is is_offline


# L2 · the default is live


def test_the_default_answers_from_the_fetched_data(downloads):
    assert ctxlen.max_input(ONLY_LIVE) == 900000
    assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    assert ctxlen.max_input(ONLY_LIVE, live=False) is None


def test_the_download_reads_the_urls_gzipped_json_with_ctxlens_user_agent(server):
    server.reply(LATEST)
    assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    [request] = server.requests
    assert request.headers["User-Agent"] == _latest.USER_AGENT


def test_the_download_trusts_certifis_cas_and_times_out_at_the_deadline(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "none.pem"))  # no system CAs
    asked = []

    def get(url, **options):
        asked.append(options)
        return httpx.Response(200, json=LATEST, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    [options] = asked
    assert options["timeout"] == _latest.DEADLINE
    certifis = ssl.create_default_context(cafile=certifi.where()).get_ca_certs()
    assert options["verify"].get_ca_certs() == certifis


def test_a_redirect_is_followed(server):
    server.reply(b"", status=301, gzip=False, headers={"Location": "/moved.json"})
    server.reply(LATEST, path="/moved.json")
    assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    assert [request.path for request in server.requests] == [
        "/_data.json",
        "/moved.json",
    ]


@pytest.mark.parametrize("function", FUNCTIONS)
def test_a_non_str_model_raises_before_any_request(function, downloads):
    with pytest.raises(TypeError, match=r"takes a model id as a str, got None"):
        function(None)
    assert downloads.call_count == 0
    assert _latest.MEMO is None


# L3 · one fetch an hour per machine


def test_one_process_fetches_once_an_hour(clock, downloads):
    for _ in range(100):
        assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    assert downloads.call_count == 1
    clock.now += 61 * 60
    ctxlen.ctxlen(ONLY_LIVE)
    assert downloads.call_count == 2


def test_a_second_process_within_the_hour_reads_the_cache(clock, downloads):
    ctxlen.ctxlen(ONLY_LIVE)
    clock.now += 59 * 60
    new_process()
    assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    assert downloads.call_count == 1
    clock.now += 2 * 60
    new_process()
    ctxlen.ctxlen(ONLY_LIVE)
    assert downloads.call_count == 2


def test_a_failed_fetch_is_remembered_for_the_hour_across_processes(clock, downloads):
    downloads.return_value = None
    assert ctxlen.max_input("gpt-4o") == BUNDLED
    clock.now += 30 * 60
    new_process()
    assert ctxlen.max_input("gpt-4o") == BUNDLED
    assert ctxlen.max_input(ONLY_LIVE) is None
    assert downloads.call_count == 1
    clock.now += 31 * 60
    new_process()
    ctxlen.max_input("gpt-4o")
    assert downloads.call_count == 2


def test_a_clock_set_back_counts_as_stale(clock, downloads):
    ctxlen.ctxlen(ONLY_LIVE)
    clock.now -= 60
    new_process()
    ctxlen.ctxlen(ONLY_LIVE)
    assert downloads.call_count == 2


def test_an_unusable_cache_directory_means_one_fetch_per_process(
    tmp_path, monkeypatch, clock, downloads
):
    (tmp_path / "a-file").write_text("")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "a-file" / "cache"))
    for _ in range(2):
        new_process()
        assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
        assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    assert downloads.call_count == 2


def test_threads_share_one_fetch(downloads):
    downloads.side_effect = lambda: time.sleep(0.2) or LATEST
    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(ctxlen.ctxlen, [ONLY_LIVE] * 8))
    assert downloads.call_count == 1
    assert answers == [1050000] * 8


@pytest.mark.parametrize(
    "broken",
    [
        "garbage\n",
        "[]",
        '{"data": null}',
        '{"fetched_at": 1791769787.5}',
        '{"fetched_at": "1791769787.5", "data": null}',
    ],
)
def test_a_broken_cache_file_is_ignored_and_fetched_again(
    broken, clock, downloads, cache_file
):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(broken)
    assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    assert downloads.call_count == 1
    assert json.loads(cache_file.read_text())["data"] == LATEST


def test_the_cache_file_holds_the_fetch_time_and_the_data(clock, downloads, cache_file):
    ctxlen.ctxlen(ONLY_LIVE)
    fetched = {"fetched_at": clock.now, "data": LATEST}
    assert cache_file.read_text() == json.dumps(fetched, separators=(",", ":"))
    downloads.return_value = None
    clock.now += 61 * 60
    new_process()
    ctxlen.ctxlen(ONLY_LIVE)
    assert json.loads(cache_file.read_text()) == {"fetched_at": clock.now, "data": None}


@pytest.mark.parametrize("xdg_cache_home", [None, "  "], ids=["unset", "blank"])
def test_the_cache_is_under_home_dot_cache_without_xdg_cache_home(
    xdg_cache_home, tmp_path, monkeypatch, downloads
):
    if xdg_cache_home is None:
        monkeypatch.delenv("XDG_CACHE_HOME")
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", xdg_cache_home)
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".cache" / "ctxlen" / "data-1.json"
    assert _latest.cache_path() == str(path)
    ctxlen.ctxlen(ONLY_LIVE)
    assert json.loads(path.read_text())["data"] == LATEST


def test_no_home_means_memory_only(tmp_path, monkeypatch, downloads):
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setattr(os.path, "expanduser", lambda path: path)
    monkeypatch.chdir(tmp_path)
    assert _latest.cache_path() is None
    assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    assert list(tmp_path.iterdir()) == []


# L4 · every failure is silent and bundled


def corrupt_cache(cache_file, monkeypatch):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("garbage\n")


def cache_path_is_a_file(cache_file, monkeypatch):
    cache_file.parent.parent.mkdir(parents=True)
    cache_file.parent.write_text("")


def slower_than_the_deadline(cache_file, monkeypatch):
    monkeypatch.setattr(_latest, "DEADLINE", 0.3)


def no_httpx(cache_file, monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)


def a_download_that_hangs(cache_file, monkeypatch):
    monkeypatch.setattr(_latest, "DEADLINE", 0.3)
    monkeypatch.setattr(_latest, "download", lambda: time.sleep(1.0) or LATEST)


# (case, what the server replies or None for the refused URL, any other setup)
FAILURES = [
    ("refused", None, None),
    (
        "slower_than_the_deadline",
        {"body": LATEST, "delay": 1.0},
        slower_than_the_deadline,
    ),
    ("http_404", {"body": LATEST, "status": 404}, None),
    ("http_500", {"body": LATEST, "status": 500}, None),
    ("bad_gzip", {"body": b"not gzip", "gzip": False, "headers": GZIP}, None),
    ("not_json", {"body": b"<html>", "gzip": False}, None),
    ("a_json_list", {"body": [1, 2]}, None),
    ("no_format", {"body": FIXTURE}, None),
    ("format_2", {"body": {**LATEST, "format": 2}}, None),
    ("a_bad_pattern", {"body": {**LATEST, "routing": [["(", "openai"]]}}, None),
    ("corrupt_cache", None, corrupt_cache),
    ("cache_path_is_a_file", None, cache_path_is_a_file),
    ("no_httpx", {"body": LATEST}, no_httpx),
    ("a_download_that_hangs", None, a_download_that_hangs),
]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "case, reply, setup", FAILURES, ids=[case for case, _, _ in FAILURES]
)
def test_every_failure_gives_the_bundled_answer_silently_within_the_deadline(
    case, reply, setup, request, monkeypatch, cache_file, recwarn
):
    server = request.getfixturevalue("server") if reply else None
    if server:
        server.reply(**reply)
    if setup:
        setup(cache_file, monkeypatch)
    start = time.perf_counter()
    answer = ctxlen.max_input("gpt-4o")
    took = time.perf_counter() - start
    only_live = ctxlen.max_input(ONLY_LIVE)
    assert (answer, only_live) == (BUNDLED, None)
    assert list(recwarn) == []  # recorded from every thread, the worker's too
    assert took < _latest.DEADLINE + 0.1
    if server:
        assert len(server.requests) == (0 if case == "no_httpx" else 1)


# L5 · a failed write leaves no damage


def test_a_failed_cache_write_leaves_the_previous_file_and_no_temp_file(
    clock, downloads, cache_file, monkeypatch
):
    cache_file.parent.mkdir(parents=True)
    stale = json.dumps({"fetched_at": clock.now - 2 * 3600, "data": LATEST}).encode()
    cache_file.write_bytes(stale)

    def refuse(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", refuse)
    assert ctxlen.ctxlen(ONLY_LIVE) == 1050000
    assert downloads.call_count == 1
    assert cache_file.read_bytes() == stale
    assert [path.name for path in cache_file.parent.iterdir()] == ["data-1.json"]


# The fork hook


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_child_forked_during_a_fetch_can_look_up(downloads):
    def look_up():
        answer = ctxlen.max_input("gpt-4o")
        assert type(answer) is int and answer > 0

    child = multiprocessing.get_context("fork").Process(target=look_up)
    with _latest.LOCK:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # 3.12+: a live worker
            child.start()
        child.join(5)
    child.kill()  # one stuck on the held lock it inherited; a no-op once it has exited
    child.join()
    assert child.exitcode == 0
