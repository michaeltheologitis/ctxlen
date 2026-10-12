import json

import pytest

from ctxlen import _resolve
from ctxlen._resolve import FORMAT, build_index, find_row
from pipeline.data import (
    MODELS_DEV_PROVIDERS,
    NEVER_FILL,
    DataError,
    Disagreement,
    Entry,
    GateError,
    attach_fills,
    catalog_ids,
    check_gate,
    coverage,
    dump_data,
    models_dev_entries,
    positive_int,
    project_litellm,
    read_provider_enum,
    same_family,
)
from tests.conftest import FIXTURE, litellm_row, models_dev_model

ENUM_SOURCE = """
from enum import Enum

class LlmProviders(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NOT_A_NAME = 5

    def label(self):
        return "x"
"""


def entry(model, max_input=None, max_output=None, source=None, context=None):
    slug = model.split("/")[0]
    return Entry(source or model, model, slug, max_input, max_output, context, False)


def answers(model, data):
    hit = find_row(model, build_index(data))
    return tuple(hit[1][1:]) if hit else (None, None)


def test_read_provider_enum_takes_the_string_members_only():
    assert read_provider_enum(ENUM_SOURCE) == ["openai", "anthropic"]


@pytest.mark.parametrize(
    "source", ["class Other:\n    A = 'a'\n", "class LlmProviders:\n    A = 1\n"]
)
def test_read_provider_enum_without_the_class_is_a_data_error(source):
    with pytest.raises(DataError):
        read_provider_enum(source)


def test_projection_drops_reserved_keys_and_keeps_litellm_order():
    cost_map = {
        "sample_spec": litellm_row("one of the providers"),
        "zeta": litellm_row(max_input=10),
        "alpha": litellm_row("anthropic", max_output=20),
        "fallback_generalizations": {"rules": []},
    }
    projected = project_litellm(cost_map, ["openai"], {"nano-gpt": {}})
    assert list(projected["rows"].items()) == [
        ("zeta", ["openai", 10, None, None]),
        ("alpha", ["anthropic", None, 20, None]),
    ]
    assert projected["providers"] == ["nano-gpt", "openai"]
    assert projected["fills"] == {}


def test_projection_never_reads_max_tokens():
    projected = project_litellm({"m": litellm_row(max_tokens=1048576)}, [], {})
    assert projected["rows"]["m"] == ["openai", None, None, None]


def test_zero_limit_is_none():
    assert positive_int(0, "m.max_input_tokens") is None
    assert positive_int(None, "m.max_input_tokens") is None
    assert positive_int(128000.0, "m.max_input_tokens") == 128000


@pytest.mark.parametrize("value", ["x", -1, 1.5, True])
def test_malformed_litellm_limit_is_a_data_error(value):
    with pytest.raises(DataError, match="gpt-4o.*max_input_tokens"):
        project_litellm({"gpt-4o": litellm_row(max_input=value)}, [], {})


def test_projection_keeps_the_ai21_chat_rows():
    cost_map = {
        "jamba-1.5-large": litellm_row("ai21", mode="chat"),
        "j2-light": litellm_row("ai21", mode="completion"),
        "gpt-4o": litellm_row(mode="chat"),
    }
    assert project_litellm(cost_map, [], {})["ai21_chat"] == ["jamba-1.5-large"]


def test_routing_rules_are_the_provider_only_rules():
    rules = [
        {"pattern": "^claude-", "model_info": {"litellm_provider": "anthropic"}},
        {"pattern": "claude-[a-z]+", "model_info": {"max_input_tokens": 200000}},
        {
            "pattern": "^legacy-",
            "model_info": {"litellm_provider": "openai", "max_input_tokens": 1},
        },
        {"pattern": "^broken-", "model_info": {"litellm_provider": None}},
    ]
    projected = project_litellm({"fallback_generalizations": {"rules": rules}}, [], {})
    assert projected["routing"] == [["^claude-", "anthropic"], ["^legacy-", "openai"]]


def test_models_dev_input_cap_wins_over_context():
    api = {
        "openai": {"models": {"gpt-5": models_dev_model(400000, 128000, input=272000)}}
    }
    entries, skipped = models_dev_entries(api, {"openai": "openai"})
    assert entries == [
        Entry("openai/gpt-5", "openai/gpt-5", "openai", 272000, 128000, 400000, False)
    ]
    assert skipped == 0


def test_models_dev_output_of_at_least_the_context_is_dropped_as_padding():
    api = {"togetherai": {"models": {"m": models_dev_model(131072, 131072)}}}
    entries, _ = models_dev_entries(api, {"togetherai": "together_ai"})
    assert entries == [
        Entry(
            "togetherai/m", "together_ai/m", "together_ai", 131072, None, 131072, True
        )
    ]


def test_models_dev_embedding_output_is_a_vector_size_not_a_limit():
    embedding = models_dev_model(8191, 1536)
    api = {
        "openai": {
            "models": {
                "text-embedding-3-small": embedding,
                "bge-m3": {**embedding, "type": "embedding"},
                "e5": {**embedding, "family": "text-embedding"},
                "gpt-5": models_dev_model(400000, 128000),
            }
        }
    }
    entries, _ = models_dev_entries(api, {"openai": "openai"})
    assert {e.model: (e.max_input, e.max_output) for e in entries} == {
        "openai/text-embedding-3-small": (8191, None),
        "openai/bge-m3": (8191, None),
        "openai/e5": (8191, None),
        "openai/gpt-5": (400000, 128000),
    }


def test_vertex_default_suffix_is_dropped():
    api = {
        "google-vertex": {"models": {"gemini-9@default": models_dev_model(10, 5)}},
        "google-vertex-anthropic": {
            "models": {"claude-opus-9@default": models_dev_model(10, 5)}
        },
    }
    entries, _ = models_dev_entries(api, MODELS_DEV_PROVIDERS)
    models = sorted(e.model for e in entries)
    assert models == ["vertex_ai/claude-opus-9", "vertex_ai/gemini-9"]


def test_excluded_providers_never_fill():
    assert set(NEVER_FILL) == {
        "openrouter",
        "vercel",
        "huggingface",
        "venice",
        "github-copilot",
    }
    assert not set(NEVER_FILL) & set(MODELS_DEV_PROVIDERS)
    api = {
        provider: {"models": {"m": models_dev_model(1000, 100)}}
        for provider in NEVER_FILL
    }
    assert models_dev_entries(api, MODELS_DEV_PROVIDERS) == ([], 0)


def test_copilot_total_is_its_prompt_cap_not_the_labs_window(monkeypatch):
    model = models_dev_model(1000000, 128000)
    api = {"github-copilot": {"models": {"claude-opus-5.5": model, "gpt-9": model}}}
    entries, _ = models_dev_entries(api, MODELS_DEV_PROVIDERS)
    merged, _, _ = attach_fills(FIXTURE, entries)
    assert "github_copilot/gpt-9" not in merged["fills"]
    monkeypatch.setattr(_resolve, "shipped_index", lambda: build_index(merged))
    copilot = _resolve.limits("github_copilot/claude-opus-5.5", "ctxlen")
    assert copilot == (200000, 128000, 200000)


@pytest.mark.parametrize(
    "status, fills", [("deprecated", False), ("beta", True), (None, True)]
)
def test_deprecated_models_dev_entries_never_fill(status, fills):
    model = {**models_dev_model(1000, 100), **({"status": status} if status else {})}
    api = {"openai": {"models": {"gpt-9": model}}}
    entries, skipped = models_dev_entries(api, {"openai": "openai"})
    assert [e.model for e in entries] == (["openai/gpt-9"] if fills else [])
    assert skipped == 0


def test_non_text_models_never_fill():
    image = models_dev_model(1000, 100, modalities=("image",))
    api = {"openai": {"models": {"image-1": image}}}
    assert models_dev_entries(api, {"openai": "openai"}) == ([], 0)


def test_malformed_models_dev_entry_is_skipped_not_fatal():
    models = {"bad": models_dev_model("big", 100), "good": models_dev_model(1000, 100)}
    entries, skipped = models_dev_entries(
        {"openai": {"models": models}}, {"openai": "openai"}
    )
    assert [e.model for e in entries] == ["openai/good"]
    assert skipped == 1


def test_fill_fills_only_missing_fields_of_a_same_family_row():
    key = "mistral/mistral-large-4"
    merged, _, counts = attach_fills(FIXTURE, [entry(key, 1000, 262144)])
    assert merged["rows"][key] == ["mistral", 1048576, 262144, 1048576]
    assert (counts["field_fills_input"], counts["field_fills_output"]) == (0, 1)
    assert FIXTURE["rows"][key] == ["mistral", 1048576, None, 1048576]


def test_lenient_only_hit_fills_nothing():
    lenient = entry("azure_ai/text-embedding-3-small", 8191, 3072, context=8192)
    merged, _, _ = attach_fills(FIXTURE, [lenient])
    assert merged["rows"]["text-embedding-3-small"] == ["openai", 8191, None, None]
    assert "azure_ai/text-embedding-3-small" not in merged["fills"]


def test_unmatched_entry_becomes_a_fill_row():
    data = {**FIXTURE, "fills": {}}
    entries = [
        entry("dashscope/qwen3.6-plus", 1000000, 65536, context=1000000),
        entry("zai/glm-9"),
    ]
    merged, _, _ = attach_fills(data, entries)
    assert merged["fills"] == {
        "dashscope/qwen3.6-plus": ["dashscope", 1000000, 65536, 1000000]
    }


def test_models_dev_context_becomes_the_rows_total():
    merged, disagreements, _ = attach_fills(
        FIXTURE, [entry("openai/gpt-4.1", context=1047576)]
    )
    assert merged["rows"]["gpt-4.1"] == ["openai", 1047576, 32768, 1047576]
    assert disagreements == []


def test_a_context_below_the_rows_own_limits_is_dropped_and_reported():
    entries = [
        entry("openai/gpt-4.1", context=1000000),
        entry("zai/glm-9", 300000, 8192, context=200000),
    ]
    merged, disagreements, _ = attach_fills(FIXTURE, entries)
    assert merged["rows"]["gpt-4.1"][3] is None
    assert merged["fills"]["zai/glm-9"] == ["zai", 300000, 8192, None]
    assert disagreements == [
        Disagreement("gpt-4.1", "context", 1047576, 1000000, "openai/gpt-4.1"),
        Disagreement("zai/glm-9", "context", 300000, 200000, "zai/glm-9"),
    ]


def test_first_entry_wins_for_a_field():
    entries = [
        entry("mistral/mistral-large-4", max_output=1, source="a"),
        entry("mistral/mistral-large-4", max_output=2, source="b"),
        entry("zai/glm-9", 10, 1, source="c"),
        entry("zai/glm-9", 20, 2, source="d"),
    ]
    merged, _, _ = attach_fills(FIXTURE, entries)
    assert merged["rows"]["mistral/mistral-large-4"][2] == 1
    assert merged["fills"]["zai/glm-9"] == ["zai", 10, 1, None]


def test_difference_over_five_percent_is_a_disagreement():
    entries = [
        entry("openai/gpt-4o", 140000, 16384 * 104 // 100),
        entry("openai/gpt-5", 272000 * 104 // 100),
    ]
    merged, disagreements, _ = attach_fills(FIXTURE, entries)
    assert disagreements == [
        Disagreement("gpt-4o", "max_input", 128000, 140000, "openai/gpt-4o")
    ]
    assert merged["rows"]["gpt-4o"] == ["openai", 128000, 16384, 128000]


def test_fills_never_change_a_number_litellm_gives():
    ids = [
        *FIXTURE["rows"],
        *(f"{row[0]}/{key}" for key, row in FIXTURE["rows"].items()),
    ]
    entries = [
        entry(f"{slug}/{model_id}", 7, 7)
        for slug in FIXTURE["providers"]
        for model_id in ids
    ]
    merged, _, _ = attach_fills(FIXTURE, entries)
    for model in ids:
        for before, after in zip(answers(model, FIXTURE), answers(model, merged)):
            assert before is None or after == before, model


@pytest.mark.parametrize(
    "row_provider, slug, expected",
    [
        ("openai", "openai", True),
        ("vertex_ai-language-models", "vertex_ai", True),
        ("fireworks_ai-embedding-models", "fireworks_ai", True),
        ("bedrock_converse", "bedrock", True),
        ("bedrock", "bedrock_converse", True),
        ("openai", "azure_ai", False),
        ("cohere_chat", "cohere", False),
        ("gemini", "vertex_ai", False),
        ("vertex_ai", "vertex_ai-language-models", False),
    ],
)
def test_same_family(row_provider, slug, expected):
    assert same_family(row_provider, slug) is expected


def shrunk(data, part, keep):
    items = (
        data[part].items() if isinstance(data[part], dict) else enumerate(data[part])
    )
    kept = [item for n, item in enumerate(items) if n < keep]
    return {
        **data,
        part: dict(kept) if isinstance(data[part], dict) else [v for _, v in kept],
    }


@pytest.mark.parametrize("part", ["rows", "fills", "providers"])
def test_gate_rejects_a_shrink_over_25_percent(part):
    previous = {
        **FIXTURE,
        "rows": {**FIXTURE["rows"], "extra": ["openai", 1, 1]},
        "fills": {**FIXTURE["fills"]},
    }
    previous["fills"].update({f"zai/m{n}": ["zai", 1, 1] for n in range(2)})
    size = len(previous[part])
    check_gate(previous, previous)
    with pytest.raises(GateError, match=part):
        check_gate(shrunk(previous, part, size * 3 // 4 - 1), previous)


@pytest.mark.parametrize(
    "change", [{"gpt-4o": None}, {"claude-opus-5-5": ["anthropic", None, 128000]}]
)
def test_gate_rejects_a_missing_anchor(change):
    rows = {
        key: value
        for key, value in {**FIXTURE["rows"], **change}.items()
        if value is not None
    }
    with pytest.raises(GateError, match="anchor"):
        check_gate({**FIXTURE, "rows": rows}, FIXTURE)


def test_gate_without_a_previous_release_runs_the_absolute_checks():
    check_gate(shrunk(FIXTURE, "providers", 1), None)
    with pytest.raises(GateError, match="value"):
        check_gate(
            {**FIXTURE, "rows": {**FIXTURE["rows"], "bad": ["openai", 0, None]}}, None
        )
    with pytest.raises(GateError, match="pattern"):
        check_gate({**FIXTURE, "routing": [["(unclosed", "openai"]]}, None)


def test_dump_data_starts_with_the_format_and_is_one_row_per_line():
    text = dump_data(FIXTURE)
    assert json.loads(text) == {"format": FORMAT, **FIXTURE}
    lines = text.splitlines()
    assert lines[0] == '{"format":1,'
    assert '"gpt-4o":["openai",128000,16384,128000],' in lines
    assert '"dashscope/qwen3.6-plus":["dashscope",1000000,65536,1000000],' in lines
    assert len(lines) == len(FIXTURE["rows"]) + len(FIXTURE["fills"]) + 6


def test_coverage_counts_litellm_alone_and_with_fills():
    projected = {**FIXTURE, "fills": {}}
    catalog = [
        "dashscope/qwen3.6-plus",
        "openai/gpt-4o",
        "openai/text-embedding-3-small",
        "zai/glm-9",
        "fireworks_ai/routers/auto",
    ]
    assert coverage(projected, FIXTURE, catalog) == {
        "catalog": 5,
        "litellm_alone": 2,
        "ctxlen": 3,
        "total_from_models_dev": 2,
        "total_from_litellm": 1,
        "total_none": 2,
    }


def test_catalog_is_the_fill_providers_and_vercel_as_litellm_ids():
    api = {
        "alibaba": {"models": {"qwen3.6-plus": models_dev_model(10, 5)}},
        "vercel": {"models": {"openai/gpt-5": models_dev_model(10, 5)}},
        "openrouter": {"models": {"openai/gpt-5": models_dev_model(10, 5)}},
        "github-copilot": {"models": {"gpt-5": models_dev_model(10, 5)}},
    }
    assert catalog_ids(api) == [
        "dashscope/qwen3.6-plus",
        "github_copilot/gpt-5",
        "vercel_ai_gateway/openai/gpt-5",
    ]
