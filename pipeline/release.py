"""The release's pure logic: channels, versions, the plan, the content digest, the watchdog's verdict."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import NamedTuple

MAIN_REF = "refs/heads/main"
FAULTS = ("none", "force", "shrink", "refuse", "alarm")
BROKEN_RESULTS = ("failure", "cancelled")
LITELLM_COOLDOWN = timedelta(days=3)
RELEASE_LABEL = "ctxlen-release"
CONFORMANCE_LABEL = "ctxlen-conformance"
VERSION_FORMS = {
    "final": re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.(\d+)"),
    "dev": re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.dev(\d+)"),
}


class Plan(NamedTuple):
    channel: str
    version: str
    publish: bool
    record: bool
    record_version: str | None


class Verdict(NamedTuple):
    issues: list[tuple[str, str, str]]  # (label, title, body)
    ping: str | None  # "success", "fail" or None


def channel_for(ref: str) -> str:
    """The ref's channel: "final" only for MAIN_REF, "dev" for every other ref."""
    return "final" if ref == MAIN_REF else "dev"


def release_plan(
    event: str,
    ref: str,
    fault: str,
    built_digest: str,
    latest_digest: str | None,
    latest_version: str | None,
    latest_tagged: bool,
    new_version: str,
) -> Plan:
    """Whether this run publishes and records.

    Only a dispatch, or a schedule on main, can. It publishes when the built content differs
    from the channel's latest release, or when the fault is force or refuse. It records what
    it publishes, or else a latest release whose tag is missing. Raises on a fault on main,
    and on a version not of the ref's channel, so no ref but main can plan a final version.
    """
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    if fault != "none" and ref == MAIN_REF:
        raise ValueError("faults run only off main")
    channel = channel_for(ref)
    if _parts(new_version, channel) is None:
        raise ValueError(
            f"{new_version} is not a {channel} version, so {ref} cannot publish it"
        )
    if event not in ("schedule", "workflow_dispatch") or (
        event == "schedule" and ref != MAIN_REF
    ):
        return Plan(channel, new_version, False, False, None)
    publish = built_digest != latest_digest or fault in ("force", "refuse")
    record = publish or (latest_version is not None and not latest_tagged)
    record_version = new_version if publish else (latest_version if record else None)
    return Plan(channel, new_version, publish, record, record_version)


def next_version(existing: Iterable[str], day: date, channel: str) -> str:
    """The next version of `channel` on `day`: N is 1 plus the highest N already used, yanked included."""
    today = (day.year, day.month, day.day)
    used = [
        parts[3]
        for parts in (_parts(v, channel) for v in existing)
        if parts and parts[:3] == today
    ]
    n = max(used, default=0) + 1
    return f"{day.year}.{day.month}.{day.day}.{'dev' if channel == 'dev' else ''}{n}"


def is_dev(version: str) -> bool:
    """True for a YYYY.M.D.devN version."""
    return VERSION_FORMS["dev"].fullmatch(version) is not None


def _parts(version: str, channel: str) -> tuple[int, ...] | None:
    match = VERSION_FORMS[channel].fullmatch(version)
    return tuple(int(part) for part in match.groups()) if match else None


def latest_release(releases: Mapping[str, Sequence[dict]], channel: str) -> str | None:
    """The highest version of `channel`'s form in PyPI's "releases" that has a non-yanked file."""
    live = [
        v
        for v, files in releases.items()
        if _parts(v, channel) and any(not f["yanked"] for f in files)
    ]
    return max(live, key=lambda v: _parts(v, channel), default=None)


def content_digest(wheel: bytes) -> str:
    """sha256 over the wheel's ctxlen/ members by name, each as name, NUL, bytes; dist-info excluded."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in sorted(n for n in archive.namelist() if n.startswith("ctxlen/")):
            digest.update(name.encode() + b"\0" + archive.read(name))
    return digest.hexdigest()


def missing_from_pypi(tags: Iterable[str], releases: Mapping[str, object]) -> list[str]:
    """The tagged versions PyPI's JSON does not list. record tags a version only after PyPI took it,
    so a non-empty answer means the JSON is stale."""
    return sorted({tag[1:] for tag in tags if tag.startswith("v")} - set(releases))


def latest_litellm_release(pypi: dict, now: datetime) -> str:
    """The newest all-numeric litellm release with a non-yanked file uploaded at least LITELLM_COOLDOWN ago."""

    def old_enough(files: Sequence[dict]) -> bool:
        uploads = [
            datetime.fromisoformat(f["upload_time_iso_8601"].replace("Z", "+00:00"))
            for f in files
        ]
        return bool(uploads) and now - min(uploads) >= LITELLM_COOLDOWN

    eligible = [
        version
        for version, files in pypi["releases"].items()
        if re.fullmatch(r"\d+(\.\d+)*", version)
        and old_enough([f for f in files if not f["yanked"]])
    ]
    return max(eligible, key=lambda v: tuple(int(part) for part in v.split(".")))


def watchdog_verdict(
    ref: str,
    fault: str,
    results: Mapping[str, str],
    delivered: bool | None,
    differences: int | None,
    reason: str,
) -> Verdict:
    """The issues to open or bump and the ping to send, from the jobs' results and PyPI's state.

    `delivered` says whether PyPI's latest release carries this run's content (None: there is
    nothing to check), and `reason` is why build failed. A body names the problem only, never
    the run, so it stays the same while the problem does.

    A timeout cancels a job. Conformance never pings: it has its own issue. Off main only the
    alarm fault pings, because the live tier makes jobs fail on purpose.
    """
    issues = []
    failed = [
        job
        for job in ("build", "publish", "record")
        if results.get(job) in BROKEN_RESULTS
    ]
    broken = bool(failed) or delivered is False
    if broken:
        lines = [f"The release on `{ref}` did not deliver."]
        for job in failed:
            cause = f": {reason}" if job == "build" and reason else ""
            outcome = f"failed{cause}" if results[job] == "failure" else "was cancelled"
            lines.append(f"- the {job} job {outcome}")
        if delivered is False:
            lines.append("- PyPI's latest release does not carry this run's content")
        issues.append(
            (RELEASE_LABEL, "ctxlen's release is not delivering", "\n".join(lines))
        )
    could_not_run = results.get("conformance") == "failure"
    if could_not_run or differences:
        detail = "could not run" if could_not_run else f"differs on {differences} ids"
        body = f"Conformance on `{ref}` {detail}: ctxlen and litellm find different rows. See the run's conformance report."
        issues.append(
            (CONFORMANCE_LABEL, "ctxlen and litellm find different rows", body)
        )
    if fault == "alarm" or (ref == MAIN_REF and broken):
        ping = "fail"
    elif ref == MAIN_REF and delivered:
        ping = "success"
    else:
        ping = None
    return Verdict(issues, ping)
