import io
import zipfile
from datetime import date, datetime, timedelta, timezone

import pytest

from pipeline.release import (
    CONFORMANCE_LABEL,
    FAULTS,
    MAIN_REF,
    RELEASE_LABEL,
    Plan,
    channel_for,
    content_digest,
    is_dev,
    latest_litellm_release,
    latest_release,
    missing_from_pypi,
    next_version,
    release_plan,
    watchdog_verdict,
)

TASK_REF = "refs/heads/claude/hopeful-brown-gw7aqa"
DAY = date(2026, 10, 11)
EVENTS = ("push", "pull_request", "schedule", "workflow_dispatch")
ALL_GREEN = dict.fromkeys(("build", "conformance", "publish", "record"), "success")
BROKEN_RUNS = [
    ({**ALL_GREEN, "build": "failure"}, None),
    ({**ALL_GREEN, "conformance": "failure"}, True),
    ({**ALL_GREEN, "publish": "failure", "record": "skipped"}, True),
    ({**ALL_GREEN, "record": "failure"}, True),
    ({**ALL_GREEN, "record": "cancelled"}, True),
    (ALL_GREEN, False),
]


def plan(
    event="workflow_dispatch",
    ref=TASK_REF,
    fault="none",
    built="a",
    latest="a",
    tagged=True,
    **kw,
):
    """release_plan for a dispatch off main whose built content equals the latest release's."""
    version = kw.pop("version", next_version([], DAY, channel_for(ref)))
    latest_version = kw.pop("latest_version", "2026.10.10.dev1" if latest else None)
    return release_plan(
        event, ref, fault, built, latest, latest_version, tagged, version
    )


def wheel(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_channel_is_final_only_on_main():
    assert channel_for(MAIN_REF) == "final"
    for ref in (
        TASK_REF,
        "refs/pull/1/merge",
        "refs/heads/other",
        "refs/heads/main-2",
        "refs/tags/v1",
    ):
        assert channel_for(ref) == "dev"


@pytest.mark.parametrize(
    "existing, day, channel, expected",
    [
        ([], DAY, "final", "2026.10.11.1"),
        (["2026.10.11.1", "2026.10.11.2"], DAY, "final", "2026.10.11.3"),
        (["2026.10.10.7"], DAY, "final", "2026.10.11.1"),
        (["2026.10.11.dev4"], DAY, "final", "2026.10.11.1"),
        ([], DAY, "dev", "2026.10.11.dev1"),
        (["2026.10.11.dev1", "2026.10.11.3"], DAY, "dev", "2026.10.11.dev2"),
        (["2026.10.11.dev9", "2026.10.11.dev10"], DAY, "dev", "2026.10.11.dev11"),
        ([], date(2027, 1, 5), "final", "2027.1.5.1"),
    ],
)
def test_next_version(existing, day, channel, expected):
    assert next_version(existing, day, channel) == expected


def test_next_version_counts_yanked_versions():
    releases = {
        "2026.10.11.1": [{"yanked": True}],
        "2026.10.11.dev1": [{"yanked": True}],
    }
    assert next_version(releases, DAY, "final") == "2026.10.11.2"
    assert next_version(releases, DAY, "dev") == "2026.10.11.dev2"


def test_latest_release_ignores_yanked_and_the_other_channel():
    releases = {
        "2026.10.9.1": [{"yanked": False}],
        "2026.10.10.1": [{"yanked": False}],
        "2026.10.10.2": [{"yanked": True}],
        "2026.10.10.3": [],
        "2026.10.11.dev2": [{"yanked": False}],
        "2026.10.11.dev10": [{"yanked": False}],
    }
    assert latest_release(releases, "final") == "2026.10.10.1"
    assert latest_release(releases, "dev") == "2026.10.11.dev10"
    assert latest_release({}, "final") is None


def test_pypi_is_stale_while_it_lacks_a_tagged_version():
    tags = ["v2026.10.11.dev1", "v2026.10.11.1", "not-a-version-tag"]
    releases = {"2026.10.11.1": [{"yanked": True}]}
    assert missing_from_pypi(tags, releases) == ["2026.10.11.dev1"]
    assert missing_from_pypi(tags, {**releases, "2026.10.11.dev1": []}) == []
    assert missing_from_pypi([], {}) == []


def test_content_digest_ignores_dist_info():
    code = {"ctxlen/__init__.py": "x", "ctxlen/_data.json": "{}"}
    first = wheel({**code, "ctxlen-1.dist-info/METADATA": "Version: 1"})
    second = wheel({**code, "ctxlen-2.dist-info/METADATA": "Version: 2"})
    assert content_digest(first) == content_digest(second)


def test_content_digest_changes_with_the_data():
    first = wheel({"ctxlen/__init__.py": "x", "ctxlen/_data.json": '{"rows":{}}'})
    second = wheel(
        {"ctxlen/__init__.py": "x", "ctxlen/_data.json": '{"rows":{"a":[]}}'}
    )
    assert content_digest(first) != content_digest(second)


@pytest.mark.parametrize("ref", [TASK_REF, "refs/pull/1/merge", "refs/heads/other"])
@pytest.mark.parametrize("event", EVENTS)
def test_non_main_ref_never_plans_a_final_version(event, ref):
    for fault in FAULTS:
        for built, latest in (("a", "a"), ("a", "b"), ("a", None)):
            result = plan(event, ref, fault, built, latest)
            assert not result.publish or is_dev(result.version)
            assert result.channel == "dev"
            with pytest.raises(ValueError):
                plan(event, ref, fault, built, latest, version="2026.10.11.1")


def test_scheduled_run_off_main_plans_nothing():
    assert plan("schedule", built="a", latest="b", tagged=False) == Plan(
        "dev", "2026.10.11.dev1", False, False, None
    )


@pytest.mark.parametrize("event", ["push", "pull_request"])
@pytest.mark.parametrize("ref", [TASK_REF, MAIN_REF])
def test_push_and_pull_request_never_publish(event, ref):
    result = plan(event, ref, built="a", latest="b", tagged=False)
    assert (result.publish, result.record) == (False, False)


def test_unchanged_content_does_not_publish():
    assert plan(built="a", latest="a").publish is False
    assert plan("schedule", MAIN_REF, built="a", latest="a").publish is False


def test_changed_content_publishes_and_records_its_version():
    assert plan(built="a", latest="b") == Plan(
        "dev", "2026.10.11.dev1", True, True, "2026.10.11.dev1"
    )
    assert plan("schedule", MAIN_REF, built="a", latest=None).version == "2026.10.11.1"


@pytest.mark.parametrize("fault", ["force", "refuse"])
def test_force_and_refuse_publish_unchanged_content(fault):
    assert plan(fault=fault, built="a", latest="a").publish is True


def test_a_missing_tag_is_recorded_by_the_next_run():
    result = plan(built="a", latest="a", tagged=False, latest_version="2026.10.10.dev3")
    assert result == Plan("dev", "2026.10.11.dev1", False, True, "2026.10.10.dev3")


@pytest.mark.parametrize("fault", [f for f in FAULTS if f != "none"])
def test_faults_are_rejected_on_main(fault):
    with pytest.raises(ValueError, match="off main"):
        plan("workflow_dispatch", MAIN_REF, fault)


def test_an_unknown_fault_is_rejected():
    with pytest.raises(ValueError):
        plan(fault="explode")


def test_latest_litellm_release_waits_three_days_and_skips_prereleases_and_yanked():
    now = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)

    def files(days_ago, yanked=False):
        uploaded = (now - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return [{"yanked": yanked, "upload_time_iso_8601": uploaded}]

    releases = {
        "1.104.2": files(3.5),
        "1.104.10": files(4, yanked=True),
        "1.103.9": files(10),
        "1.105.0": files(0.5),
        "1.105.0rc1": files(9),
        "1.104.3.dev1": files(9),
    }
    assert latest_litellm_release({"releases": releases}, now) == "1.104.2"


def test_watchdog_pings_success_only_on_main_when_delivered():
    assert watchdog_verdict(MAIN_REF, "none", ALL_GREEN, True, 0, "").ping == "success"
    assert watchdog_verdict(MAIN_REF, "none", ALL_GREEN, True, 3, "").ping == "success"
    assert watchdog_verdict(TASK_REF, "none", ALL_GREEN, True, 0, "").ping is None


@pytest.mark.parametrize("results, delivered", BROKEN_RUNS)
def test_watchdog_pings_fail_on_main_for_a_failed_job_or_undelivered_content(
    results, delivered
):
    assert watchdog_verdict(MAIN_REF, "none", results, delivered, 0, "").ping == "fail"


@pytest.mark.parametrize("results, delivered", BROKEN_RUNS)
@pytest.mark.parametrize("fault", [f for f in FAULTS if f != "alarm"])
def test_watchdog_never_pings_off_main_without_the_alarm_fault(
    fault, results, delivered
):
    assert watchdog_verdict(TASK_REF, fault, results, delivered, 0, "").ping is None


def test_watchdog_pings_fail_for_the_alarm_fault():
    assert watchdog_verdict(TASK_REF, "alarm", ALL_GREEN, True, 0, "").ping == "fail"


@pytest.mark.parametrize(
    "results, delivered, reason, named",
    [
        (
            {**ALL_GREEN, "build": "failure"},
            None,
            "gate: rows shrank",
            "gate: rows shrank",
        ),
        ({**ALL_GREEN, "publish": "failure"}, True, "", "publish"),
        ({**ALL_GREEN, "record": "failure"}, True, "", "record"),
        (ALL_GREEN, False, "", "PyPI"),
    ],
)
def test_watchdog_opens_the_release_issue_for_a_failed_job_or_undelivered_content(
    results, delivered, reason, named
):
    verdict = watchdog_verdict(TASK_REF, "none", results, delivered, 0, reason)
    [(label, _, body)] = verdict.issues
    assert label == RELEASE_LABEL
    assert named in body and TASK_REF in body


def test_watchdog_opens_no_issue_when_all_is_well():
    assert watchdog_verdict(TASK_REF, "none", ALL_GREEN, True, 0, "").issues == []


@pytest.mark.parametrize(
    "results, differences",
    [(ALL_GREEN, 3), ({**ALL_GREEN, "conformance": "failure"}, None)],
)
def test_watchdog_opens_the_conformance_issue_for_differences(results, differences):
    verdict = watchdog_verdict(MAIN_REF, "none", results, True, differences, "")
    assert [label for label, _, _ in verdict.issues] == [CONFORMANCE_LABEL]
