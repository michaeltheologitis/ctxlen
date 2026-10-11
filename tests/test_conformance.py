from types import SimpleNamespace

import pytest

from ctxlen import _resolve
from pipeline.conformance import (
    NON_MODELS,
    check_loaded,
    compare,
    corpus,
    drift,
    litellm_answer,
    prefixed_form,
    sample_keys,
    variants,
)
from tests.conftest import litellm_row, models_dev_model

COST_MAP = {
    "sample_spec": litellm_row("x"),
    "gpt-4o": litellm_row("openai", 128000, 16384),
    "openai/gpt-4o-mini": litellm_row("openai", 128000),
    "claude-opus-5-5": litellm_row("anthropic", 1000000),
    "fallback_generalizations": {"rules": []},
}
API = {"alibaba": {"models": {"qwen3.6-plus": models_dev_model(10, 5)}}}


class NotMappedError(Exception):
    pass


def fake_get_model_info(answers):
    def get_model_info(model):
        if model not in answers:
            raise NotMappedError(f"This model isn't mapped yet. model={model}")
        return {"key": answers[model]}

    return get_model_info


def litellm_module(**changes):
    common_utils = SimpleNamespace(
        get_bedrock_cross_region_inference_regions=lambda: sorted(
            _resolve.BEDROCK_CROSS_REGION
        ),
        AmazonBedrockGlobalConfig=lambda: SimpleNamespace(
            get_all_regions=lambda: sorted(_resolve.BEDROCK_REGIONS)
        ),
    )
    fireworks = SimpleNamespace(
        FIREROUTER="firerouter", AZURE_FOUNDRY_FIREWORKS_MODEL_ID_PREFIX="FW-"
    )
    module = SimpleNamespace(
        azure_llms={k: v for k, v in _resolve.AZURE_ALIASES.items() if k != "ada"},
        azure_embedding_models={"ada": "azure/ada"},
        constants=SimpleNamespace(
            PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO=frozenset(
                {"github_copilot", "chatgpt"}
            ),
            REPLICATE_MODEL_NAME_WITH_ID_LENGTH=64,
        ),
        llms=SimpleNamespace(
            bedrock=SimpleNamespace(common_utils=common_utils),
            fireworks_ai=SimpleNamespace(common_utils=fireworks),
        ),
    )
    for name, value in changes.items():
        setattr(module, name, value)
    return module


def test_corpus_groups_dedup_and_order():
    items = corpus(COST_MAP, API, {"openai", "anthropic"})
    groups = {}
    for group, model in items:
        groups.setdefault(group, []).append(model)
    assert list(groups) == ["key", "prefixed", "catalog", "variant", "non_model"]
    assert groups["key"] == ["gpt-4o", "openai/gpt-4o-mini", "claude-opus-5-5"]
    assert groups["prefixed"] == ["openai/gpt-4o", "anthropic/claude-opus-5-5"]
    assert groups["catalog"] == ["dashscope/qwen3.6-plus"]
    assert "gpt-4o-2099-01-01" in groups["variant"]
    assert len({model for _, model in items}) == len(items)
    assert not {"sample_spec", "fallback_generalizations"} & {
        model for _, model in items
    }


def test_corpus_is_deterministic():
    assert corpus(COST_MAP, API, {"openai"}) == corpus(COST_MAP, API, {"openai"})
    keys = [f"model-{n}" for n in range(1000)]
    assert sample_keys(keys, 10) == sample_keys(reversed(keys), 10)
    assert len(set(sample_keys(keys, 10))) == 10


def test_non_models_are_fifty_ids_no_row_answers():
    assert len(NON_MODELS) == 50
    some = {"", " ", "gpt-4o ", "/", "openai/", "not-a-model", "azure/my-prod-gpt4"}
    assert some <= set(NON_MODELS)


@pytest.mark.parametrize(
    "key, row_provider, expected",
    [
        ("gpt-4o", "openai", "openai/gpt-4o"),
        ("gemini-2.5-pro", "vertex_ai-language-models", "vertex_ai/gemini-2.5-pro"),
        ("accounts/x", "fireworks_ai-embedding-models", "fireworks_ai/accounts/x"),
        (
            "anthropic.claude-opus-4-6-v1",
            "bedrock_converse",
            "bedrock/anthropic.claude-opus-4-6-v1",
        ),
        ("jamba-1.5-large", "ai21", "ai21/jamba-1.5-large"),
        ("some-model", "not-a-provider", None),
    ],
)
def test_prefixed_form(key, row_provider, expected):
    providers = {"openai", "vertex_ai", "fireworks_ai", "bedrock", "ai21"}
    assert prefixed_form(key, row_provider, providers) == expected


@pytest.mark.parametrize(
    "key, row_provider, expected",
    [
        (
            "claude-opus-5-5",
            "anthropic",
            ["claude-opus-5-5-2099-01-01", "claude-opus-5-5-001", "CLAUDE-OPUS-5-5"],
        ),
        (
            "gpt-4o",
            "openai",
            ["gpt-4o-2099-01-01", "gpt-4o-001", "GPT-4O", "ft:gpt-4o:acme::abc123"],
        ),
        (
            "anthropic.claude-x-v1:0",
            "bedrock_converse",
            [
                "anthropic.claude-x-v1:0-2099-01-01",
                "anthropic.claude-x-v1:0-001",
                "ANTHROPIC.CLAUDE-X-V1:0",
                "bedrock/converse/anthropic.claude-x-v1:0",
                "bedrock/us-west-2/anthropic.claude-x-v1:0",
            ],
        ),
    ],
)
def test_variants(key, row_provider, expected):
    assert variants(key, row_provider) == expected


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.pop("gpt-4o"),
        lambda m: m.update({"gpt-4o": {**m["gpt-4o"], "max_input_tokens": 1}}),
        lambda m: m.update({"extra": {"litellm_provider": "openai"}}),
    ],
)
def test_check_loaded_detects_a_fallback_map(change):
    loaded = {
        key: dict(value)
        for key, value in COST_MAP.items()
        if key != "fallback_generalizations"
    }
    assert check_loaded(loaded, COST_MAP) == []
    change(loaded)
    assert check_loaded(loaded, COST_MAP) != []


def test_pattern_rule_answers_are_marked_rule():
    get_model_info = fake_get_model_info(
        {"claude-opus-9": "claude-opus-9", "GPT-4o": "gpt-4o"}
    )
    assert litellm_answer(get_model_info, {"gpt-4o"}, "claude-opus-9") == "rule"
    assert litellm_answer(get_model_info, {"gpt-4o"}, "GPT-4o") == "gpt-4o"


def test_litellm_exceptions_are_none():
    assert litellm_answer(fake_get_model_info({}), {"gpt-4o"}, "not-a-model") is None


def test_compare_lists_differences_and_the_e1_verdict():
    items = [
        ("key", "a"),
        ("key", "b"),
        ("prefixed", "p/a"),
        ("catalog", "c"),
        ("variant", "v"),
        ("non_model", "x"),
    ]
    ours = {"a": "a", "b": "b", "p/a": "a", "c": "c", "v": "w"}
    theirs = {"a": "a", "b": "b", "p/a": "a", "c": "rule", "v": "v"}
    report = compare(items, ours.get, theirs.get, lambda model: False)
    assert report["groups"]["key"] == {"n": 2, "equal": 2}
    assert report["groups"]["variant"] == {"n": 1, "equal": 0}
    assert report["excluded"] == {"pattern_rule": 1, "network": 0}
    assert report["differences"] == [
        {"id": "v", "group": "variant", "ctxlen": "w", "litellm": "v"}
    ]
    assert report["e1"] == {
        "keys_and_prefixed_equal": True,
        "rest_equal_ratio": 0.5,
        "holds": False,
    }


def test_compare_counts_ids_only_ctxlen_answers():
    report = compare(
        [("key", "a"), ("catalog", "b")],
        {"a": "a", "b": "b"}.get,
        {"a": "a"}.get,
        lambda m: False,
    )
    assert report["ctxlen_only"] == 1
    assert report["e1"]["holds"] is False


def test_compare_skips_network_providers():
    asked = []

    def litellm_key(model):
        asked.append(model)

    report = compare(
        [("catalog", "huggingface/x"), ("catalog", "y")],
        {}.get,
        litellm_key,
        lambda m: "/" in m,
    )
    assert asked == ["y"]
    assert report["excluded"]["network"] == 1
    assert report["e1"]["holds"] is True


def test_drift_reports_changed_and_unavailable_constants():
    assert drift(litellm_module()) == []
    changed = litellm_module(
        constants=SimpleNamespace(REPLICATE_MODEL_NAME_WITH_ID_LENGTH=65)
    )
    report = drift(changed)
    assert any(line.startswith("changed: REPLICATE_VERSION_LENGTH") for line in report)
    assert "unavailable: AUTHENTICATING_PROVIDERS" in report
