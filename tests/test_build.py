import functools
import json
import zipfile

import pytest

from ctxlen._resolve import FORMAT
from pipeline.build import main, post_issue
from tests.conftest import litellm_row, models_dev_model

ENUM_SOURCE = """
class LlmProviders(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    DASHSCOPE = "dashscope"
"""
COST_MAP = {
    "sample_spec": litellm_row("one of the providers"),
    "gpt-4o": litellm_row("openai", 128000, 16384),
    "gpt-4o-mini": litellm_row("openai", 128000, 16384),
    "claude-opus-5-5": litellm_row("anthropic", 1000000, 128000),
    "claude-haiku-4-5": litellm_row("anthropic", 200000),
    "gemini/gemini-2.5-pro": litellm_row("gemini", 1048576, max_tokens=65536),
    "gemini/gemini-2.5-flash": litellm_row("gemini", 1048576),
    "mistral/mistral-large-4": litellm_row("mistral", 1048576),
    "mistral/mistral-small-4": litellm_row("mistral", 262144),
    "fallback_generalizations": {
        "rules": [
            {
                "pattern": "^claude-[a-z]+-\\d+$",
                "model_info": {"litellm_provider": "anthropic"},
            }
        ]
    },
}
MODELS_DEV = {
    "alibaba": {"models": {"qwen3.6-plus": models_dev_model(1000000, 65536)}},
    "mistral": {"models": {"mistral-large-4": models_dev_model(1048576, 262144)}},
    "openai": {"models": {"gpt-4o": models_dev_model(200000, 16384)}},
}
META = {
    "fetched_at": "2026-10-11T03:17:00Z",
    "litellm_commit": "c0ca4a5",
    "litellm_release": "1.104.2",
    "models_dev_dev_head": "1accc98",
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    inputs, pypi = tmp_path / "inputs", tmp_path / "pypi"
    inputs.mkdir()
    pypi.mkdir()
    (inputs / "litellm.json").write_text(json.dumps(COST_MAP))
    (inputs / "litellm_types_utils.py").write_text(ENUM_SOURCE)
    (inputs / "litellm_providers.json").write_text(
        json.dumps({"nano-gpt": {"base_url": "https://nano-gpt.com/api/v1"}})
    )
    (inputs / "models_dev.json").write_text(json.dumps(MODELS_DEV))
    (inputs / "meta.json").write_text(json.dumps(META))
    (pypi / "ctxlen.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    return tmp_path


def outputs(workspace):
    lines = (workspace / "output").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines)


def run_data(*extra):
    return main(["data", "--inputs", "inputs", "--pypi", "pypi", *extra])


def test_data_command_writes_the_data_and_reports_from_inputs(workspace):
    assert run_data() == 0
    data = json.loads((workspace / "src/ctxlen/_data.json").read_text())
    assert next(iter(data)) == "format"
    assert data["format"] == FORMAT
    assert list(data["rows"]) == [
        key
        for key in COST_MAP
        if key not in ("sample_spec", "fallback_generalizations")
    ]
    assert data["rows"]["mistral/mistral-large-4"] == [
        "mistral",
        1048576,
        262144,
        1048576,
    ]
    assert data["rows"]["gemini/gemini-2.5-pro"] == ["gemini", 1048576, None, None]
    assert data["rows"]["gpt-4o"] == ["openai", 128000, 16384, 200000]
    assert data["fills"] == {
        "dashscope/qwen3.6-plus": ["dashscope", 1000000, 65536, 1000000]
    }
    assert data["providers"] == [
        "anthropic",
        "dashscope",
        "gemini",
        "mistral",
        "nano-gpt",
        "openai",
    ]
    provenance = json.loads((workspace / "reports/provenance.json").read_text())
    assert provenance["litellm"]["release"] == "1.104.2"
    assert provenance["rows"]["litellm"] == 8
    assert provenance["field_fills"] == {"input": 0, "output": 1}
    assert provenance["disagreements"] == 1
    assert provenance["gate"]["previous"] is None
    csv = (workspace / "reports/disagreements.csv").read_text().splitlines()
    assert csv == [
        "key,field,litellm,models_dev,source",
        "gpt-4o,max_input,128000,200000,openai/gpt-4o",
    ]
    assert outputs(workspace)["litellm_release"] == "1.104.2"


def test_data_command_fails_the_gate_on_a_shrunk_map(workspace):
    assert run_data() == 0
    shipped = (workspace / "src/ctxlen/_data.json").read_text()
    with zipfile.ZipFile(workspace / "pypi/latest.whl", "w") as wheel:
        wheel.writestr("ctxlen/_data.json", shipped)
    assert run_data("--fault", "shrink") == 1
    assert (workspace / "src/ctxlen/_data.json").read_text() == shipped
    assert outputs(workspace)["reason"].startswith("gate: rows")
    assert "gate: rows" in (workspace / "summary").read_text()


def test_an_unexpected_failure_still_outputs_its_reason(workspace):
    (workspace / "inputs/meta.json").write_text("[]")
    assert run_data() == 1
    assert outputs(workspace)["reason"].startswith("TypeError: list indices")


class FakeGh:
    def __init__(self):
        self.issues = {}
        self.calls = []

    def __call__(self, *args):
        args = list(args)
        self.calls.append(args)
        options = (
            dict(zip(args[2::2], args[3::2])) if args[:2] != ["issue", "edit"] else {}
        )
        if args[:2] == ["issue", "list"]:
            found = [
                {"number": n, "body": i["body"]}
                for n, i in self.issues.items()
                if i["label"] == options["--label"]
            ]
            return json.dumps(found)
        if args[:2] == ["issue", "create"]:
            number = len(self.issues) + 1
            self.issues[number] = {
                "label": options["--label"],
                "body": options["--body"],
                "comments": [],
            }
            return f"https://github.com/o/r/issues/{number}"
        if args[:2] == ["issue", "edit"]:
            self.issues[int(args[2])]["body"] = args[4]
        if args[:2] == ["issue", "comment"]:
            self.issues[int(args[2])]["comments"].append(args[4])
        return ""


def test_post_issue_creates_then_updates_only_on_change():
    gh = FakeGh()
    post = functools.partial(post_issue, "ctxlen-release", "Release failing")
    assert post("build failed", "https://run/1", gh) == "created"
    assert post("build failed", "https://run/2", gh) == "unchanged"
    assert post("publish failed", "https://run/3", gh) == "updated"
    assert gh.issues == {
        1: {
            "label": "ctxlen-release",
            "body": "publish failed",
            "comments": ["changed in https://run/3"],
        }
    }
    assert ["label", "create", "ctxlen-release", "--force"] in gh.calls
