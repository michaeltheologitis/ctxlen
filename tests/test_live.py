"""The live tier, E5, E9 and E6: release.yml dispatched on CTXLEN_LIVE_REF against real PyPI, GitHub
and healthchecks.io. The tests run in file order, and only the first publishes: a pass publishes one dev
release. The later dispatches replay its inputs, so upstream moving mid-pass cannot publish another."""

import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from ctxlen import _latest
from ctxlen._resolve import FORMAT, build_index, find_row
from pipeline.build import PYPI_WAIT_SECONDS, current_ctxlen_json, gh, remote_tags
from pipeline.release import RELEASE_LABEL, is_dev

REF = os.environ.get("CTXLEN_LIVE_REF", "")
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (REF and os.environ.get("GH_TOKEN")),
        reason="the live tier needs GH_TOKEN and CTXLEN_LIVE_REF",
    ),
]
RUN_TIMEOUT = 60 * 60
POLL_SECONDS = 20
STATE = {}
SEEN_RUNS = set()
# argv: the URL to fetch. A live lookup, and whether it fell back to the bundled copy or loaded httpx.
E9_LOOKUP = """import json, sys
import ctxlen
from ctxlen import _latest, _resolve
_latest.URL = sys.argv[1]
answer = ctxlen.max_input("gpt-4o")
print(json.dumps({"answer": answer, "bundled_read": _resolve.shipped_index.cache_info().currsize > 0,
                  "httpx": "httpx" in sys.modules}))
"""


def when(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def until(produce, timeout, failure):
    deadline = time.monotonic() + timeout
    while True:
        value = produce()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(failure)
        time.sleep(POLL_SECONDS)


def dispatch(fault, replay_run=""):
    """Dispatches release.yml on REF and returns the finished run, with its jobs."""
    started = datetime.now(timezone.utc) - timedelta(seconds=30)
    inputs = ["-f", f"fault={fault}"] + (
        ["-f", f"replay_run={replay_run}"] if replay_run else []
    )
    gh("workflow", "run", "release.yml", "--ref", REF, *inputs)

    def new_run():
        filters = ["--workflow", "release.yml", "--event", "workflow_dispatch"]
        options = ["--branch", REF, "--limit", "10", "--json", "databaseId,createdAt"]
        listed = gh("run", "list", *filters, *options)
        fresh = [
            run
            for run in json.loads(listed)
            if run["databaseId"] not in SEEN_RUNS and when(run["createdAt"]) >= started
        ]
        return min(fresh, key=lambda run: run["createdAt"]) if fresh else None

    started_run = until(new_run, 300, f"no release.yml run started for fault={fault}")
    run_id = started_run["databaseId"]
    SEEN_RUNS.add(run_id)

    def finished():
        fields = "databaseId,status,conclusion,jobs,url"
        run = json.loads(gh("run", "view", str(run_id), "--json", fields))
        return run if run["status"] == "completed" else None

    return until(finished, RUN_TIMEOUT, f"run {run_id} did not finish")


def job(run, name):
    return next(job for job in run["jobs"] if job["name"] == name)


def job_log(run, name):
    run_id, job_id = str(run["databaseId"]), str(job(run, name)["databaseId"])
    return gh("run", "view", run_id, "--log", "--job", job_id)


def pypi_releases():
    """PyPI's releases, read by the build's own rule: once its JSON lists every tagged version."""
    return current_ctxlen_json().get("releases", {})


def tags():
    return set(remote_tags("origin"))


def release_issue():
    query = ["--label", RELEASE_LABEL, "--state", "open", "--json", "body,updatedAt"]
    issues = json.loads(gh("issue", "list", *query))
    return issues[0] if issues else {"body": "", "updatedAt": "1970-01-01T00:00:00Z"}


def install(version, scratch):
    """ctxlen==version from PyPI in a fresh venv: its _data.json bytes and max_input("gpt-4o")."""
    venv = scratch / "venv"
    subprocess.run(["uv", "venv", "--quiet", "--python", "3.13", str(venv)], check=True)
    python = str(venv / "bin" / "python")

    def installed():
        command = ["uv", "pip", "install", "--quiet", "--refresh", "--python", python]
        done = subprocess.run(
            [*command, f"ctxlen=={version}"], capture_output=True, check=False
        )
        return done.returncode == 0

    until(installed, PYPI_WAIT_SECONDS, f"ctxlen=={version} did not install from PyPI")

    def stdout(code):
        command = [python, "-c", code]
        return subprocess.run(
            command, check=True, capture_output=True, cwd=scratch
        ).stdout

    data = stdout(
        "import importlib.resources, sys; "
        "sys.stdout.buffer.write(importlib.resources.files('ctxlen').joinpath('_data.json').read_bytes())"
    )
    return data, int(
        stdout("import ctxlen; print(ctxlen.max_input('gpt-4o', live=False))")
    )


def test_dispatch_publishes_a_dev_release_that_installs_and_answers(tmp_path):
    before = set(pypi_releases())
    run = dispatch("force")
    assert run["conclusion"] == "success", run["url"]
    [version] = until(
        lambda: set(pypi_releases()) - before, PYPI_WAIT_SECONDS, "no new version"
    )
    assert is_dev(version)
    run_id = str(run["databaseId"])
    gh("run", "download", run_id, "--name", "record", "--dir", str(tmp_path / "record"))
    recorded = (tmp_path / "record" / "src" / "ctxlen" / "_data.json").read_bytes()
    installed, answer = install(version, tmp_path)
    assert installed == recorded
    expected = find_row("gpt-4o", build_index(json.loads(recorded)))[1][1]
    assert answer == expected > 0
    assert f"v{version}" in tags()
    STATE.update(force_run=run_id, version=version, recorded=recorded)


def test_the_dev_release_fetches_its_tags_data_once_and_a_second_process_reads_the_cache(
    tmp_path,
):
    version, recorded = STATE["version"], json.loads(STATE["recorded"])
    install(version, tmp_path)
    url = _latest.URL.replace("/main/", f"/v{version}/", 1)
    environment = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
    environment.pop("CTXLEN_OFFLINE", None)

    def look_up():
        python = str(tmp_path / "venv" / "bin" / "python")
        command = [python, "-c", E9_LOOKUP, url]
        done = subprocess.run(
            command, env=environment, capture_output=True, check=True, cwd=tmp_path
        )
        return json.loads(done.stdout)

    expected = find_row("gpt-4o", build_index(recorded))[1][1]
    assert look_up() == {"answer": expected, "bundled_read": False, "httpx": True}
    cache = json.loads((tmp_path / "cache" / "ctxlen" / "data-1.json").read_text())
    assert cache["data"] == recorded
    assert recorded["format"] == FORMAT
    assert look_up() == {"answer": expected, "bundled_read": False, "httpx": False}


def test_replayed_inputs_publish_nothing():
    before = set(pypi_releases())
    run = dispatch("none", STATE["force_run"])
    assert run["conclusion"] == "success", run["url"]
    assert job(run, "publish")["conclusion"] == "skipped"
    assert job(run, "record")["conclusion"] == "skipped"
    assert set(pypi_releases()) == before


def test_every_unyanked_version_has_its_tag_every_file_its_attestation_and_every_tag_its_version():
    releases = pypi_releases()
    version_tags = {tag for tag in tags() if tag.startswith("v")}
    assert STATE["version"] in releases
    for version, files in releases.items():
        if not all(file["yanked"] for file in files):
            assert f"v{version}" in version_tags, version
        for file in files:
            url = f"https://pypi.org/integrity/ctxlen/{version}/{file['filename']}/provenance"
            with urllib.request.urlopen(url, timeout=60) as response:
                assert response.status == 200
                assert json.load(response)["attestation_bundles"], url
    assert {tag[1:] for tag in version_tags} <= set(releases)


def test_shrunk_map_publishes_nothing_and_opens_an_issue():
    versions, version_tags = set(pypi_releases()), tags()
    started = datetime.now(timezone.utc) - timedelta(seconds=30)
    run = dispatch("shrink", STATE["force_run"])
    assert job(run, "build")["conclusion"] == "failure", run["url"]
    assert job(run, "publish")["conclusion"] == "skipped"
    assert job(run, "watchdog")["conclusion"] == "success"
    assert set(pypi_releases()) == versions and tags() == version_tags
    issue = release_issue()
    assert "the build job failed: gate: rows shrank" in issue["body"]
    assert when(issue["updatedAt"]) >= started


def test_refused_publish_leaves_no_version_or_tag_and_opens_an_issue():
    versions, version_tags = set(pypi_releases()), tags()
    started = datetime.now(timezone.utc) - timedelta(seconds=30)
    run = dispatch("refuse", STATE["force_run"])
    assert job(run, "publish")["conclusion"] == "failure", run["url"]
    assert "exchange failure" in job_log(run, "publish").lower()
    assert job(run, "record")["conclusion"] == "skipped"
    assert set(pypi_releases()) == versions and tags() == version_tags
    issue = release_issue()
    assert "the publish job failed" in issue["body"]
    assert when(issue["updatedAt"]) >= started


def test_alarm_fault_reaches_healthchecks():
    run = dispatch("alarm", STATE["force_run"])
    assert job(run, "watchdog")["conclusion"] == "success", run["url"]
    assert job(run, "publish")["conclusion"] == "skipped"
    assert "Pinged healthchecks.io: fail" in job_log(run, "watchdog")
