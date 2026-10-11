import pytest

from ctxlen import _resolve
from ctxlen._resolve import (
    bedrock_base_model,
    build_index,
    find_row,
    fireworks_resource,
    infer_provider,
    provider_matches,
    strip_version,
)
from tests.conftest import FIXTURE


def found_key(model, index):
    hit = find_row(model, index)
    return hit[0] if hit else None


@pytest.mark.parametrize(
    "model, key",
    [
        # the examples ctxlen was specified with
        ("gpt-4o", "gpt-4o"),
        ("openai/gpt-4o", "gpt-4o"),
        ("anthropic/claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("vertex_ai/gemini-2.5-pro", "gemini-2.5-pro"),
        (
            "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-pro"),
        ("gemini/gemini-2.5-pro-001", "gemini/gemini-2.5-pro"),
        ("ft:gpt-4o-2024-08-06:acme::AbC123", "ft:gpt-4o-2024-08-06"),
        ("GPT-4o", "gpt-4o"),
        ("azure/gpt-4.1", "azure/gpt-4.1"),
        ("not-a-model", None),
        ("azure/my-prod-gpt4", None),
        ("claude-opus-9", None),
        ("openrouter/anthropic/claude-sonnet-4.5", None),
        # quirks litellm has, reproduced on purpose
        ("openai/gpt-4o-2099-01-01", None),
        ("gpt-4o-2099-01-01", "gpt-4o"),
        ("jamba-1.5-large", None),
        ("ai21/jamba-1.5-large", None),
        ("ai21/j2-light", "j2-light"),
        ("gemini-2.5-pro-001", None),
        ("bedrock/us.amazon.nova-premier-v1:0", "amazon.nova-premier-v1:0"),
        ("fireworks_ai/routers/auto", "fireworks_ai/accounts/fireworks/routers/auto"),
        ("fireworks_ai/auto", None),
        ("CLAUDE-SONNET-4-5", "claude-sonnet-4-5"),
        ("Bedrock/anthropic.claude-opus-4-6-v1", None),
        # aliases, special cases and stripping
        ("gpt-35-turbo", "azure/gpt-35-turbo"),
        ("azure/gpt-41", "gpt-4.1"),
        ("cohere/command-r-08-2024", "command-r-08-2024"),
        (
            "bedrock_mantle/us-east-1/anthropic.claude-opus-5-5",
            "bedrock_mantle/anthropic.claude-opus-5-5",
        ),
        (
            "bedrock/converse/anthropic.claude-opus-4-6-v1",
            "anthropic.claude-opus-4-6-v1",
        ),
        ("us.anthropic.claude-opus-4-6-v1[1m]", "anthropic.claude-opus-4-6-v1"),
        (
            "anthropic.claude-sonnet-4-5-20250929-v1:0:18k",
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        (
            "fireworks_ai/deepseek-r1",
            "fireworks_ai/accounts/fireworks/models/deepseek-r1",
        ),
        ("azure_ai/gpt-4o", "gpt-4o"),
        ("litellm_proxy/gpt-4o", "gpt-4o"),
        ("github/gpt-4o", "gpt-4o"),
        ("github_copilot/gpt-4o", "github_copilot/gpt-4o"),
    ],
)
def test_find_row_matches_litellm_lookup(model, key, index):
    assert found_key(model, index) == key


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-4o", (128000, 16384)),
        ("text-embedding-3-small", (8191, None)),
        ("mistral/mistral-large-4", (1048576, None)),
        ("fireworks_ai/routers/auto", (None, None)),
        ("not-a-model", (None, None)),
        ("dashscope/qwen3.6-plus", (1000000, 65536)),
    ],
)
def test_limits_come_from_the_row(model, expected, index, monkeypatch):
    monkeypatch.setattr(_resolve, "shipped_index", lambda: index)
    assert _resolve.limits(model, "max_input")[:2] == expected


@pytest.mark.parametrize(
    "model, total",
    [
        ("gpt-4o", 128000),
        ("openai/gpt-5", 400000),
        ("claude-opus-5-5", 1000000),
        ("dashscope/qwen3.6-plus", 1000000),
        ("nano-gpt/anthropic/claude-opus-4.6", 1000000),
        ("text-embedding-3-small", 8191),
        ("gemini/gemini-2.5-pro-preview-tts", 16384),
        ("fireworks_ai/routers/auto", None),
        ("not-a-model", None),
    ],
)
def test_total_is_models_dev_context_else_the_larger_limit(
    model, total, index, monkeypatch
):
    monkeypatch.setattr(_resolve, "shipped_index", lambda: index)
    assert _resolve.limits(model, "ctxlen")[2] == total


def test_fill_is_reached_only_by_its_prefixed_id_after_litellm_misses(index):
    assert find_row("dashscope/qwen3.6-plus", index) == (
        "dashscope/qwen3.6-plus",
        ["dashscope", 1000000, 65536, 1000000],
    )
    assert find_row("qwen3.6-plus", index) is None
    fill = "nano-gpt/anthropic/claude-opus-4.6"
    assert found_key(fill, index) == fill
    assert found_key(f"openrouter/{fill}", index) is None


def test_exact_key_wins_then_the_later_case_variant():
    index = build_index(
        {**FIXTURE, "rows": {"foo": ["openai", 1, 1], "Foo": ["openai", 2, 2]}}
    )
    assert found_key("foo", index) == "foo"
    assert found_key("Foo", index) == "Foo"
    assert found_key("FOO", index) == "Foo"


@pytest.mark.parametrize(
    "model, expected",
    [
        ("openai/gpt-4o", ("openai", "gpt-4o")),
        ("azure/command-r-08-2024", ("openai", "azure/command-r-08-2024")),
        ("azure/mistral-large-4", ("openai", "azure/mistral-large-4")),
        ("azure/gpt-4.1", ("azure", "gpt-4.1")),
        ("cohere/command-r-08-2024", ("cohere_chat", "command-r-08-2024")),
        ("cohere/command-a", ("cohere", "command-a")),
        (
            "bedrock_mantle/us-east-1/anthropic.claude-opus-5-5",
            ("bedrock_mantle", "anthropic.claude-opus-5-5"),
        ),
        ("ai21/jamba-1.5-large", ("ai21_chat", "jamba-1.5-large")),
        ("ai21/j2-light", ("ai21", "j2-light")),
        (
            "nano-gpt/anthropic/claude-opus-4.6",
            ("nano-gpt", "anthropic/claude-opus-4.6"),
        ),
        (
            "ft:gpt-4o-mini-2024-07-18:acme::x1",
            ("openai", "ft:gpt-4o-mini-2024-07-18:acme::x1"),
        ),
        ("gpt-image-2", ("openai", "gpt-image-2")),
        ("gpt-4o", ("openai", "gpt-4o")),
        ("claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
        ("command-r-08-2024", ("cohere_chat", "command-r-08-2024")),
        ("a" * 30 + ":" + "f" * 64, ("replicate", "a" * 30 + ":" + "f" * 64)),
        (
            "anthropic.claude-x:" + "1" * 50,
            ("bedrock", "anthropic.claude-x:" + "1" * 50),
        ),
        ("gpt-4o:" + "1" * 60, (None, "gpt-4o:" + "1" * 60)),
        ("gemini-2.5-pro", ("vertex_ai", "gemini-2.5-pro")),
        ("jamba-1.5-large", ("ai21_chat", "jamba-1.5-large")),
        ("anthropic.claude-opus-4-6-v1", ("bedrock", "anthropic.claude-opus-4-6-v1")),
        ("*", ("openai", "*")),
        ("bytez/some-model", ("bytez", "bytez/some-model")),
        ("amazon_nova-lite", ("amazon_nova", "amazon_nova-lite")),
        ("claude-opus-9", ("anthropic", "claude-opus-9")),
        (
            "us.anthropic.claude-sonnet-9-v1:0",
            ("bedrock", "us.anthropic.claude-sonnet-9-v1:0"),
        ),
        ("deepseek-chat", (None, "deepseek-chat")),
        ("not-a-model", (None, "not-a-model")),
        ("", (None, "")),
    ],
)
def test_infer_provider(model, expected, index):
    assert infer_provider(model, index) == expected


@pytest.mark.parametrize(
    "row_provider, provider, expected",
    [
        ("openai", None, True),
        (None, "anthropic", True),
        ("openai", "openai", True),
        ("vertex_ai-language-models", "vertex_ai", True),
        ("fireworks_ai-embedding-models", "fireworks_ai", True),
        ("bedrock_converse", "bedrock", True),
        ("bedrock", "bedrock_mantle", True),
        ("openai", "azure_ai", True),
        ("azure", "azure_ai", True),
        ("anthropic", "azure_ai", False),
        ("anthropic", "litellm_proxy", True),
        ("anthropic", "github", True),
        ("anthropic", "nadir", True),
        ("ai21", "ai21_chat", False),
        ("gemini", "vertex_ai", False),
        ("vertex_ai-language-models", "gemini", False),
        ("openai", "anthropic", False),
    ],
)
def test_provider_matches(row_provider, provider, expected):
    assert provider_matches(row_provider, provider) is expected


@pytest.mark.parametrize(
    "model, provider, expected",
    [
        ("bedrock/us.anthropic.claude-x-v1:0", "bedrock", "anthropic.claude-x-v1:0"),
        ("us.anthropic.claude-x-v1:0", "bedrock_converse", "anthropic.claude-x-v1:0"),
        ("gemini-2.5-pro-001", "vertex_ai", "gemini-2.5-pro"),
        ("gemini/gemini-2.5-pro-001", "gemini", "gemini/gemini-2.5-pro"),
        ("databricks/dbrx-2", "databricks", "databricks/dbrx"),
        ("ft:gpt-4o-2024-08-06:acme::AbC123", "openai", "ft:gpt-4o-2024-08-06"),
        ("ft:gpt-4o-2024-08-06:acme::AbC123", None, "ft:gpt-4o-2024-08-06"),
        ("gpt-4o-2024-08-06", "openai", "gpt-4o"),
        ("gpt-4o-2024-08-06", None, "gpt-4o"),
        ("gemini-2.5-pro-001", None, "gemini-2.5-pro-001"),
        ("gemini-2.5-pro-001", "openai", "gemini-2.5-pro-001"),
    ],
)
def test_strip_version(model, provider, expected):
    assert strip_version(model, provider) == expected


@pytest.mark.parametrize(
    "model, expected",
    [
        ("bedrock/converse/us.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0"),
        (
            "bedrock/nova-2/arn:aws:bedrock:us-east-1:1:custom-model/x",
            "amazon.nova-2-custom",
        ),
        ("nova/arn:aws:bedrock:us-east-1:1:custom-model/x", "amazon.nova-custom"),
        ("bedrock/us-west-2/anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0"),
        (
            "arn:aws:bedrock:us-east-1:1:inference-profile/us.anthropic.claude-x-v1:0",
            "anthropic.claude-x-v1:0",
        ),
        ("anthropic.claude-x-v2:0:51k", "anthropic.claude-x-v2:0"),
        ("us.anthropic.claude-opus-4-6-v1[1m]", "anthropic.claude-opus-4-6-v1"),
        ("amazon.nova-premier-v1:0", "amazon.nova-premier-v1:0"),
        ("global.anthropic.claude-x", "anthropic.claude-x"),
        ("invoke/mantle/anthropic.claude-x", "anthropic.claude-x"),
        ("chat_completions/anthropic.claude-x", "chat_completions/anthropic.claude-x"),
        ("mars-1/anthropic.claude-x", "mars-1/anthropic.claude-x"),
    ],
)
def test_bedrock_base_model(model, expected):
    assert bedrock_base_model(model) == expected


@pytest.mark.parametrize(
    "model, expected",
    [
        ("deepseek-r1", "accounts/fireworks/models/deepseek-r1"),
        ("fireworks_ai/deepseek-r1", "accounts/fireworks/models/deepseek-r1"),
        ("accounts/acme/models/x", "accounts/acme/models/x"),
        ("FW-abc123", "FW-abc123"),
        ("acme#x", "acme#x"),
        ("routers/auto", "accounts/fireworks/routers/auto"),
        ("models/x", "accounts/fireworks/models/x"),
        ("glm-5-fast", "accounts/fireworks/routers/glm-5-fast"),
        ("firerouter", "accounts/fireworks/routers/firerouter"),
        ("firerouter/cheap", "accounts/fireworks/routers/firerouter/cheap"),
        ("auto", "accounts/fireworks/models/auto"),
    ],
)
def test_fireworks_resource(model, expected):
    assert fireworks_resource(model) == expected


def test_routing_is_case_insensitive_and_prefixes_are_not(index):
    assert found_key("CLAUDE-SONNET-4-5", index) == "claude-sonnet-4-5"
    assert (
        found_key("bedrock/anthropic.claude-opus-4-6-v1", index)
        == "anthropic.claude-opus-4-6-v1"
    )
    assert found_key("Bedrock/anthropic.claude-opus-4-6-v1", index) is None
