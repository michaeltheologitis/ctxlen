"""The release's I/O, one subcommand per step: `python -m pipeline.build fetch|data|package|watchdog`.

Every network call and subprocess of a release is here; what to do with the answers is decided
by the pure functions of pipeline.data and pipeline.release. Each subcommand appends its results
to $GITHUB_OUTPUT and its reasons to $GITHUB_STEP_SUMMARY. It exits 1 on any failure, with the
failure's one-line `reason` among its outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pipeline.data import (
    MODELS_DEV_PROVIDERS,
    RESERVED_KEYS,
    SHRINK_LIMIT,
    DataError,
    GateError,
    attach_fills,
    catalog_ids,
    check_gate,
    coverage,
    disagreements_csv,
    dump_data,
    models_dev_entries,
    project_litellm,
    read_provider_enum,
)
from pipeline.release import (
    channel_for,
    content_digest,
    latest_litellm_release,
    latest_release,
    missing_from_pypi,
    next_version,
    release_plan,
    watchdog_verdict,
)

LITELLM_REPO = "https://github.com/BerriAI/litellm"
MODELS_DEV_REPO = "https://github.com/anomalyco/models.dev"
LITELLM_JSON = "https://raw.githubusercontent.com/BerriAI/litellm/{commit}/model_prices_and_context_window.json"
MODELS_DEV_API = "https://models.dev/api.json"
PYPI_JSON = "https://pypi.org/pypi/{project}/json"
USER_AGENT = "ctxlen-release (+https://github.com/michaeltheologitis/ctxlen)"
HTTP_TRIES = 3
WHEEL_MEMBERS = {
    "litellm_types_utils.py": "litellm/types/utils.py",
    "litellm_providers.json": "litellm/llms/openai_like/providers.json",
}
DATA_FILE = Path("src/ctxlen/_data.json")
REPORTS = Path("reports")
SDIST_FILES = frozenset(
    {
        "pyproject.toml",
        "PKG-INFO",
        "README.md",
        "LICENSE",
        "src/ctxlen/__init__.py",
        "src/ctxlen/_resolve.py",
        "src/ctxlen/_data.json",
        "src/ctxlen/py.typed",
    }
)
# uv 0.13 keeps the original pyproject.toml beside the normalized copy it writes into the sdist.
SDIST_OPTIONAL = frozenset({"pyproject.toml.orig"})
FIRST_LOOKUP_LIMIT_MS = 100
PROBE_PYTHONS = ("3.13", "3.9")
PYPI_WAIT_SECONDS = 600
PYPI_POLL_SECONDS = 30
PING_TRIES = 3


class BuildError(Exception):
    """A fetch, package or watchdog step that cannot complete; the message is the reason."""


def output(**values: object) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as handle:
            handle.writelines(f"{key}={value}\n" for key, value in values.items())


def summary(text: str) -> None:
    print(text)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as handle:
            handle.write(text + "\n")


def run(command: Sequence[str], **kwargs: object) -> str:
    """The command's stdout; BuildError, with the end of its stderr, if it fails."""
    done = subprocess.run(
        list(command), capture_output=True, text=True, check=False, **kwargs
    )
    if done.returncode != 0:
        raise BuildError(f"{' '.join(command)} failed: {done.stderr.strip()[-500:]}")
    return done.stdout


def gh(*args: str) -> str:
    return run(["gh", *args])


def http_get(url: str) -> bytes:
    """GET `url`. A connection failure or a 5xx is tried HTTP_TRIES times, 10 s apart."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(HTTP_TRIES):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.URLError as error:
            refused = isinstance(error, urllib.error.HTTPError) and error.code < 500
            if refused or attempt == HTTP_TRIES - 1:
                raise
        time.sleep(10)
    raise AssertionError("unreachable")


def pypi_json(project: str) -> dict:
    """PyPI's JSON for `project`, or {} if PyPI has no such project."""
    try:
        return json.loads(http_get(PYPI_JSON.format(project=project)))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise


def remote_tags(repo: str) -> list[str]:
    out = run(["git", "ls-remote", "--tags", repo])
    return [
        line.split("refs/tags/", 1)[1]
        for line in out.splitlines()
        if not line.endswith("^{}")
    ]


def ls_remote(repo: str, ref: str) -> str:
    """The commit `ref` names in `repo`, or "" if there is none."""
    out = run(["git", "ls-remote", repo, ref])
    return out.split()[0] if out.strip() else ""


def utc_iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def fetch(args: argparse.Namespace) -> None:
    """Writes inputs/ (litellm at a commit, release R's two files, models.dev, meta.json) and pypi/.

    inputs/ can be replayed: with --pypi-only, an earlier run's inputs/ is already in place and
    only pypi/, which is always live, is fetched.
    """
    inputs, pypi = args.inputs, args.pypi
    if not args.pypi_only:
        inputs.mkdir(parents=True, exist_ok=True)
        fetched_at = datetime.now(timezone.utc)
        dev_head = ls_remote(MODELS_DEV_REPO, "refs/heads/dev")
        commit = ls_remote(LITELLM_REPO, "refs/heads/main")
        cost_map = http_get(LITELLM_JSON.format(commit=commit))
        json_object(cost_map, "litellm.json")
        (inputs / "litellm.json").write_bytes(cost_map)
        api = http_get(MODELS_DEV_API)
        json_object(api, "models_dev.json")
        (inputs / "models_dev.json").write_bytes(api)
        litellm = pypi_json("litellm")
        release = latest_litellm_release(litellm, fetched_at)
        unpack_litellm_wheel(release, litellm["releases"][release], inputs)
        json_object(
            (inputs / "litellm_providers.json").read_bytes(), "litellm_providers.json"
        )
        meta = {
            "fetched_at": utc_iso(fetched_at),
            "litellm_commit": commit,
            "litellm_release": release,
            "models_dev_dev_head": dev_head,
        }
        (inputs / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        summary(f"Fetched litellm {commit[:12]} (release {release}) and models.dev")
    pypi.mkdir(parents=True, exist_ok=True)
    ctxlen = current_ctxlen_json()
    (pypi / "ctxlen.json").write_text(json.dumps(ctxlen))
    latest = latest_release(ctxlen.get("releases", {}), channel_for(args.ref))
    if latest:
        (pypi / "latest.whl").write_bytes(
            http_get(release_wheel(ctxlen["releases"][latest])["url"])
        )
    summary(f"The latest {channel_for(args.ref)} release on PyPI is {latest or 'none'}")


def json_object(raw: bytes, name: str) -> dict:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise DataError(f"{name}: not a JSON object")
    return parsed


def unpack_litellm_wheel(release: str, files: Sequence[dict], inputs: Path) -> None:
    """Copies WHEEL_MEMBERS out of the litellm release's wheel for this runner, the one
    `uv pip install litellm==<release>` takes. uv picks it, checks it against PyPI's sha256s
    and unpacks it: nothing is built and nothing runs."""
    hashes = [
        f"--hash=sha256:{f['digests']['sha256']}" for f in files if not f["yanked"]
    ]
    with tempfile.TemporaryDirectory() as scratch:
        requirements = Path(scratch) / "requirements.txt"
        requirements.write_text(f"litellm=={release} {' '.join(hashes)}\n")
        site = Path(scratch) / "site"
        flags = ["--quiet", "--no-deps", "--only-binary", ":all:", "--require-hashes"]
        paths = ["--target", str(site), "-r", str(requirements)]
        run(["uv", "pip", "install", *flags, *paths])
        for name, member in WHEEL_MEMBERS.items():
            shutil.copyfile(site / member, inputs / name)


def current_ctxlen_json() -> dict:
    """PyPI's ctxlen JSON once it lists every tagged version. A version is tagged only after PyPI
    took it, and PyPI's CDN can serve a stale copy, which lacks the latest release."""
    tags = remote_tags("origin")
    deadline = time.monotonic() + PYPI_WAIT_SECONDS
    while True:
        ctxlen = pypi_json("ctxlen")
        missing = missing_from_pypi(tags, ctxlen.get("releases", {}))
        if not missing:
            return ctxlen
        if time.monotonic() > deadline:
            raise BuildError(
                f"PyPI's JSON is stale: it lacks {missing}, which are tagged"
            )
        summary(f"PyPI's JSON lacks the tagged {missing}; asking again")
        time.sleep(PYPI_POLL_SECONDS)


def release_wheel(files: Sequence[dict]) -> dict | None:
    """The first non-yanked wheel among a release's files."""
    return next(
        (f for f in files if not f["yanked"] and f["filename"].endswith(".whl")), None
    )


def data(args: argparse.Namespace) -> None:
    """Projects, fills and gates; writes src/ctxlen/_data.json and reports/."""
    raw_map = (args.inputs / "litellm.json").read_bytes()
    raw_api = (args.inputs / "models_dev.json").read_bytes()
    cost_map, api = json.loads(raw_map), json.loads(raw_api)
    meta = json.loads((args.inputs / "meta.json").read_text())
    if args.fault == "shrink":
        dropped = set([key for key in cost_map if key not in RESERVED_KEYS][1::2])
        cost_map = {key: value for key, value in cost_map.items() if key not in dropped}
    enum = read_provider_enum((args.inputs / "litellm_types_utils.py").read_text())
    registry = json.loads((args.inputs / "litellm_providers.json").read_text())
    projected = project_litellm(cost_map, enum, registry)
    entries, skipped = models_dev_entries(api, MODELS_DEV_PROVIDERS)
    merged, disagreements, counts = attach_fills(projected, entries)
    previous = previous_data(args.pypi / "latest.whl")
    check_gate(merged, previous)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    DATA_FILE.write_text(dump_data(merged))
    (REPORTS / "disagreements.csv").write_text(disagreements_csv(disagreements))
    rows = list(merged["rows"].values())
    provenance = {
        "built_at": utc_iso(datetime.now(timezone.utc)),
        "litellm": {
            "commit": meta["litellm_commit"],
            "json_sha256": hashlib.sha256(raw_map).hexdigest(),
            "release": meta["litellm_release"],
        },
        "models_dev": {
            "api_sha256": hashlib.sha256(raw_api).hexdigest(),
            "fetched_at": meta["fetched_at"],
            "dev_head": meta["models_dev_dev_head"],
        },
        "rows": {
            "litellm": len(rows),
            "with_input": sum(row[1] is not None for row in rows),
            "with_output": sum(row[2] is not None for row in rows),
            "with_context": sum(row[3] is not None for row in rows),
            "fills": len(merged["fills"]),
            "providers": len(merged["providers"]),
        },
        "field_fills": {
            "input": counts["field_fills_input"],
            "output": counts["field_fills_output"],
        },
        "dropped_padding": counts["dropped_padding"],
        "skipped_models_dev": skipped,
        "disagreements": len(disagreements),
        "coverage": coverage(projected, merged, catalog_ids(api)),
        "gate": {
            "previous": {
                part: len(previous[part]) for part in ("rows", "fills", "providers")
            }
            if previous
            else None,
            "shrink_limit": SHRINK_LIMIT,
        },
    }
    (REPORTS / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    summary(f"```json\n{json.dumps(provenance, indent=1)}\n```")
    output(litellm_release=meta["litellm_release"])


def previous_data(wheel: Path) -> dict | None:
    if not wheel.exists():
        return None
    with zipfile.ZipFile(wheel) as archive:
        return json.loads(archive.read("ctxlen/_data.json"))


def package(args: argparse.Namespace) -> None:
    """Stamps the version, builds, runs E4, and plans the publish and the record."""
    pypi, dist, record = args.pypi, args.dist, args.record
    meta = json.loads((args.inputs / "meta.json").read_text())
    releases = json.loads((pypi / "ctxlen.json").read_text()).get("releases", {})
    channel = channel_for(args.ref)
    day = datetime.fromisoformat(meta["fetched_at"].replace("Z", "+00:00")).date()
    version = next_version(releases, day, channel)
    run(["uv", "version", "--frozen", version])
    shutil.rmtree(dist, ignore_errors=True)
    run(["uv", "build", "--quiet", "--out-dir", str(dist)])
    [wheel], [sdist] = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    e4 = check_e4(wheel, sdist)
    digest = content_digest(wheel.read_bytes())
    latest = latest_release(releases, channel)
    latest_digest = (
        content_digest((pypi / "latest.whl").read_bytes()) if latest else None
    )
    latest_tagged = bool(latest) and bool(ls_remote("origin", f"refs/tags/v{latest}"))
    plan = release_plan(
        args.event,
        args.ref,
        args.fault,
        digest,
        latest_digest,
        latest,
        latest_tagged,
        version,
    )
    provenance = json.loads((REPORTS / "provenance.json").read_text())
    provenance.update(version=version, e4=e4)
    (REPORTS / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    shutil.rmtree(record, ignore_errors=True)
    for source in [DATA_FILE, *REPORTS.iterdir()]:
        (record / source).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, record / source)
    summary(
        f"Plan: {plan._asdict()}; content digest {digest}; latest {channel} release {latest} ({latest_digest})"
    )
    output(
        channel=plan.channel,
        version=plan.version,
        publish=str(plan.publish).lower(),
        record=str(plan.record).lower(),
        record_version=plan.record_version or "",
        wheel=wheel.name,
        sdist=sdist.name,
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        sdist_sha256=hashlib.sha256(sdist.read_bytes()).hexdigest(),
        digest=digest,
    )


def check_e4(wheel: Path, sdist: Path) -> dict:
    """E4: the wheel requires nothing, the sdist holds exactly SDIST_FILES, and the installed
    wheel answers with the network refused, its first lookup within FIRST_LOOKUP_LIMIT_MS."""
    with zipfile.ZipFile(wheel) as archive:
        metadata = next(
            archive.read(n).decode()
            for n in archive.namelist()
            if n.endswith(".dist-info/METADATA")
        )
    with tarfile.open(sdist) as archive:
        sdist_files = {
            name.split("/", 1)[1]
            for name in archive.getnames()
            if "/" in name and archive.getmember(name).isfile()
        }
    e4 = {
        "wheel_bytes": wheel.stat().st_size,
        "requires_dist": [
            line for line in metadata.splitlines() if line.startswith("Requires-Dist")
        ],
        "sdist_extra": sorted(sdist_files - SDIST_FILES - SDIST_OPTIONAL),
        "sdist_missing": sorted(SDIST_FILES - sdist_files),
        "probes": {python: probe(wheel, python) for python in PROBE_PYTHONS},
    }
    summary(f"E4: ```{json.dumps(e4)}```")
    if e4["requires_dist"]:
        raise BuildError(f"E4: the wheel requires {e4['requires_dist']}")
    if e4["sdist_extra"] or e4["sdist_missing"]:
        raise BuildError(
            f"E4: the sdist has extra {e4['sdist_extra']} and lacks {e4['sdist_missing']}"
        )
    for python, result in e4["probes"].items():
        if not result["network_refused"] or not (
            isinstance(result["gpt_4o"], int) and result["gpt_4o"] > 0
        ):
            raise BuildError(f"E4: the {python} probe got {result}")
    first_lookup = e4["probes"][PROBE_PYTHONS[0]]["first_lookup_ms"]
    if first_lookup > FIRST_LOOKUP_LIMIT_MS:
        raise BuildError(
            f"E4: the first lookup took {first_lookup:.0f} ms, over {FIRST_LOOKUP_LIMIT_MS} ms"
        )
    return e4


# Put on PYTHONPATH, this refuses every connection the probe's interpreter tries.
SITECUSTOMIZE = """import socket


def _refuse(*args, **kwargs):
    raise OSError("the E4 probe allows no network")


socket.socket.connect = _refuse
socket.create_connection = _refuse
socket.getaddrinfo = _refuse
"""
PROBE = """import json, socket, time
start = time.perf_counter()
import ctxlen
imported = time.perf_counter()
first = ctxlen.max_input("gpt-4o")
looked_up = time.perf_counter()
later = []
for model in ["gpt-4o", "openai/gpt-4o", "anthropic/claude-sonnet-4-5", "vertex_ai/gemini-2.5-pro",
              "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "gemini/gemini-2.5-pro-001",
              "ft:gpt-4o-2024-08-06:acme::AbC123", "GPT-4o", "dashscope/qwen3.6-plus", "not-a-model"]:
    before = time.perf_counter()
    ctxlen.max_output(model)
    later.append(time.perf_counter() - before)
try:
    socket.create_connection(("pypi.org", 443), timeout=5)
    guarded = False
except OSError:
    guarded = True
print(json.dumps({"import_ms": (imported - start) * 1000, "first_lookup_ms": (looked_up - imported) * 1000,
                  "slowest_later_ms": max(later) * 1000, "gpt_4o": first, "network_refused": guarded}))
"""


def probe(wheel: Path, python: str) -> dict:
    """PROBE's timings and answers, run on `python` in a fresh venv holding only `wheel`."""
    with tempfile.TemporaryDirectory() as scratch:
        venv, guard = Path(scratch) / "venv", Path(scratch) / "guard"
        interpreter = str(venv / "bin" / "python")
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(SITECUSTOMIZE)
        run(["uv", "venv", "--quiet", "--python", python, str(venv)])
        run(["uv", "pip", "install", "--quiet", "--python", interpreter, str(wheel)])
        environment = {"PATH": os.environ["PATH"], "PYTHONPATH": str(guard)}
        return json.loads(run([interpreter, "-c", PROBE], env=environment, cwd=scratch))


def watchdog(args: argparse.Namespace) -> None:
    """Checks PyPI carries this run's content, posts issues, and pings healthchecks.io."""
    needs = json.loads(args.results)
    results = {job: need["result"] for job, need in needs.items()}
    reason = needs.get("build", {}).get("outputs", {}).get("reason", "")
    is_delivered = delivered(args.channel, args.digest) if args.digest else None
    differences = int(args.differences) if args.differences else None
    verdict = watchdog_verdict(
        args.ref, args.fault, results, is_delivered, differences, reason
    )
    summary(
        f"Results {results}; delivered {is_delivered}; differences {differences}; ping {verdict.ping}"
    )
    for label, title, body in verdict.issues:
        summary(f"Issue {label}: {post_issue(label, title, body, args.run_url, gh)}")
    if verdict.ping:
        ping(verdict.ping)


def delivered(channel: str, digest: str) -> bool:
    """Whether PyPI's latest `channel` release carries content `digest`, asking again for up to
    PYPI_WAIT_SECONDS while it does not: PyPI's JSON is CDN-cached."""
    digests = {}
    deadline = time.monotonic() + PYPI_WAIT_SECONDS
    while True:
        releases = pypi_json("ctxlen").get("releases", {})
        latest = latest_release(releases, channel)
        if latest and latest not in digests:
            digests[latest] = content_digest(
                http_get(release_wheel(releases[latest])["url"])
            )
        if latest and digests[latest] == digest:
            return True
        if time.monotonic() > deadline:
            summary(
                f"PyPI's latest {channel} release {latest} does not carry content {digest}"
            )
            return False
        time.sleep(PYPI_POLL_SECONDS)


def post_issue(
    label: str, title: str, body: str, run_url: str, gh: Callable[..., str]
) -> str:
    """One open issue per label: create it, or edit its body and comment when the body changed.
    Returns "created", "updated" or "unchanged". Humans close issues."""
    query = ["--label", label, "--state", "open", "--json", "number,body"]
    found = json.loads(gh("issue", "list", *query))
    if not found:
        gh("label", "create", label, "--force")
        gh("issue", "create", "--title", title, "--body", body, "--label", label)
        return "created"
    issue = found[0]
    if issue["body"].strip() == body.strip():
        return "unchanged"
    number = str(issue["number"])
    gh("issue", "edit", number, "--body", body)
    gh("issue", "comment", number, "--body", f"changed in {run_url}")
    return "updated"


def ping(kind: str) -> None:
    """Pings healthchecks.io: "success", or "fail", up to PING_TRIES times."""
    url = os.environ.get("HEALTHCHECK_PING_URL", "")
    if not url:
        raise BuildError("a ping is due and HEALTHCHECK_PING_URL is empty")
    target = url if kind == "success" else f"{url.rstrip('/')}/fail"
    for attempt in range(PING_TRIES):
        try:
            http_get(target)
            summary(f"Pinged healthchecks.io: {kind}")
            return
        except urllib.error.URLError as error:
            summary(f"Ping {attempt + 1} of {PING_TRIES} failed: {error}")
    raise BuildError(f"every {kind} ping failed")


COMMANDS = {"fetch": fetch, "data": data, "package": package, "watchdog": watchdog}


def main(argv: list[str] | None = None) -> int:
    """Runs one subcommand; 0 on success, 1 on any failure."""
    parser = argparse.ArgumentParser(prog="python -m pipeline.build")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_args = commands.add_parser("fetch")
    data_args = commands.add_parser("data")
    package_args = commands.add_parser("package")
    watchdog_args = commands.add_parser("watchdog")
    for command in (fetch_args, data_args, package_args):
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--pypi", type=Path, required=True)
    for command in (fetch_args, package_args, watchdog_args):
        command.add_argument("--ref", required=True)
    for command in (data_args, package_args, watchdog_args):
        command.add_argument("--fault", default="none")
    fetch_args.add_argument("--pypi-only", action="store_true")
    package_args.add_argument("--event", required=True)
    package_args.add_argument("--dist", type=Path, required=True)
    package_args.add_argument("--record", type=Path, required=True)
    for name in ("--channel", "--digest", "--results", "--differences", "--run-url"):
        watchdog_args.add_argument(name, default="")
    args = parser.parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except Exception as error:  # noqa: BLE001 (every failure outputs its reason)
        reason = " ".join(str(error).split())
        if isinstance(error, (BuildError, DataError, GateError, ValueError, OSError)):
            print(reason, file=sys.stderr)
        else:
            traceback.print_exc()
            reason = f"{type(error).__name__}: {reason}"
        summary(f"**{args.command} failed**: {reason}")
        output(reason=reason)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
