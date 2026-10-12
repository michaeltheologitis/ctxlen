"""litellm's static lookup of a model id, ported from litellm's source.

Each function names the litellm function it ports. "A row" is one of litellm's rows: models.dev's
fill rows are tried only after every candidate has missed litellm's rows.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from typing import NamedTuple

# [litellm_provider: str | None, max_input: int | None, max_output: int | None,
#  context: int | None], where context is models.dev's total window for the model
Row = list
# The version of _data.json's layout that build_index reads. Bump it only when an older
# build_index would misread new data.
FORMAT = 1


class Index(NamedTuple):
    rows: dict[str, Row]  # every litellm key, in litellm's order
    lower: dict[str, str]  # key.lower() -> key; the later key wins, as in litellm
    fills: dict[str, Row]  # models.dev's models litellm has no row for, by litellm id
    fill_lower: dict[str, str]
    providers: frozenset[str]  # the prefixes get_llm_provider splits off an id
    routing: tuple[tuple[re.Pattern[str], str], ...]  # (pattern, provider), in order
    ai21_chat: frozenset[str]  # the chat-mode ai21 keys: `ai21/<key>` goes to ai21_chat


AZURE_ALIASES = {
    "gpt-35-turbo": "azure/gpt-35-turbo",
    "gpt-35-turbo-16k": "azure/gpt-35-turbo-16k",
    "gpt-35-turbo-instruct": "azure/gpt-35-turbo-instruct",
    "azure/gpt-41": "gpt-4.1",
    "azure/gpt-41-mini": "gpt-4.1-mini",
    "azure/gpt-41-nano": "gpt-4.1-nano",
    "ada": "azure/ada",
}
AUTHENTICATING_PROVIDERS = frozenset({"github_copilot", "chatgpt"})
VERTEX_BARE_SETS = (
    "vertex_ai-chat-models",
    "vertex_ai-code-chat-models",
    "vertex_ai-text-models",
    "vertex_ai-code-text-models",
    "vertex_ai-language-models",
    "vertex_ai-embedding-models",
    "vertex_ai-vision-models",
    "vertex_ai-image-models",
    "vertex_ai-video-models",
)
BARE_PROVIDER = {
    "openai": "openai",
    "text-completion-openai": "text-completion-openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "cohere_chat": "cohere_chat",
    "openrouter": "openrouter",
    **{row_provider: "vertex_ai" for row_provider in VERTEX_BARE_SETS},
    "ai21": "ai21_chat",
    "aleph_alpha": "aleph_alpha",
    "nlp_cloud": "nlp_cloud",
    "bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "watsonx": "watsonx",
    "empower": "empower",
    "gradient_ai": "gradient_ai",
}
BEFORE_REPLICATE = frozenset(
    {"openai", "text-completion-openai", "anthropic", "cohere", "cohere_chat"}
)
REPLICATE_VERSION_LENGTH = 64
BARE_PREFIX_PROVIDERS = (
    "bytez/",
    "gdc/",
    "lemonade/",
    "heroku/",
    "cometapi/",
    "oci/",
    "compactifai/",
    "ovhcloud/",
    "clarifai/",
    "amazon_nova",
    "sap/",
)
BEDROCK_ROUTING_PREFIXES = (
    "bedrock/",
    "converse/",
    "invoke/",
    "openai/",
    "mantle/",
    "nova-2/",
    "nova/",
)
BEDROCK_CROSS_REGION = frozenset({"global", "us", "eu", "apac", "jp", "au", "us-gov"})
BEDROCK_REGIONS = frozenset(
    {
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
        "us-gov-east-1",
        "us-gov-west-1",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "eu-central-1",
        "eu-central-2",
        "eu-south-1",
        "eu-south-2",
        "eu-north-1",
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-northeast-3",
        "ap-south-1",
        "ap-south-2",
        "ap-southeast-1",
        "ap-southeast-2",
        "ca-central-1",
        "sa-east-1",
    }
)
FIREROUTER = "firerouter"
FIREWORKS_KEPT_PREFIXES = ("accounts/", "FW-")
MATCH_ANY_PROVIDERS = frozenset({"litellm_proxy", "github", "nadir"})
DATED_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def limits(model: str, caller: str) -> tuple[int | None, int | None, int | None]:
    """(max_input, max_output, total) of the row litellm's get_model_info reaches for `model`.

    The total is the row's models.dev context; without one, its max_input, raised to its
    max_output where that is larger, so the total is never below either limit.
    Raises TypeError unless `model` is a str; `caller` names the public function in the message.
    """
    if not isinstance(model, str):
        raise TypeError(f"ctxlen.{caller}() takes a model id as a str, got {model!r}")
    hit = find_row(model, shipped_index())
    if hit is None:
        return None, None, None
    _, max_input, max_output, context = hit[1]
    if context is not None or max_input is None:
        return max_input, max_output, context
    return max_input, max_output, max(max_input, max_output or 0)


@functools.cache
def shipped_index() -> Index:
    """The index of the bundled _data.json, read on the first call."""
    import json
    from importlib.resources import files

    return build_index(
        json.loads(files("ctxlen").joinpath("_data.json").read_text("utf-8"))
    )


def build_index(data: dict) -> Index:
    """An Index over _data.json's layout; routing patterns compile with re.IGNORECASE."""
    rows, fills = data["rows"], data["fills"]
    return Index(
        rows=rows,
        lower={key.lower(): key for key in rows},
        fills=fills,
        fill_lower={key.lower(): key for key in fills},
        providers=frozenset(data["providers"]),
        routing=tuple(
            (re.compile(pattern, re.IGNORECASE), provider)
            for pattern, provider in data["routing"]
        ),
        ai21_chat=frozenset(data["ai21_chat"]),
    )


def find_row(model: str, index: Index) -> tuple[str, Row] | None:
    """The (key, row) litellm reaches for `model`: litellm's rows first, then the fills."""
    model = AZURE_ALIASES.get(model, model)
    candidates, provider = candidate_keys(model, index)
    return first_match(candidates, provider, index.rows, index.lower) or first_match(
        candidates, provider, index.fills, index.fill_lower
    )


def candidate_keys(model: str, index: Index) -> tuple[tuple[str, ...], str | None]:
    """_get_potential_model_names: the six names get_model_info tries, in order, and the provider."""
    first, slash, rest = model.partition("/")
    if slash and first in AUTHENTICATING_PROVIDERS:
        stripped = strip_version(rest, first)
        return (
            model,
            model,
            rest,
            f"{first}/{stripped}",
            stripped,
            f"{first}/{model}",
        ), first
    provider, split = infer_provider(model, index)
    stripped = strip_version(model, provider)
    combined_stripped = stripped
    prefixed = model
    if provider in ("bedrock", "bedrock_converse"):
        split = strip_bedrock_routing(split)
    if provider == "bedrock_mantle":
        split = strip_mantle_region(split)
        combined_stripped = f"bedrock_mantle/{strip_version(split, provider)}"
    if provider == "fireworks_ai":
        prefixed = f"fireworks_ai/{fireworks_resource(split)}"
    return (model, model, split, combined_stripped, stripped, prefixed), provider


def infer_provider(model: str, index: Index) -> tuple[str | None, str]:
    """get_llm_provider(model): (provider, model with the prefix split off); None where litellm raises."""
    first, slash, rest = model.partition("/")
    if slash and first == "azure" and _is_non_openai_azure_model(rest, index):
        return "openai", model
    if slash and first == "cohere" and _row_provider(rest, index) == "cohere_chat":
        return "cohere_chat", rest
    if slash and first in index.providers:
        if first == "bedrock_mantle":
            return first, strip_mantle_region(rest)
        if first == "ai21" and rest in index.ai21_chat:
            return "ai21_chat", rest
        return first, rest
    if (
        "ft:gpt-3.5-turbo" in model
        or "ft:gpt-4" in model
        or model.startswith("gpt-image")
    ):
        return "openai", model
    bare = BARE_PROVIDER.get(_row_provider(model, index))
    if bare in BEFORE_REPLICATE:
        return bare, model
    if ":" in model and len(model) > REPLICATE_VERSION_LENGTH:
        if len(model.split(":")[1]) == REPLICATE_VERSION_LENGTH:
            return "replicate", model
        return routed_provider(model, index), model
    if bare:
        return bare, model
    if model == "*":
        return "openai", model
    for prefix in BARE_PREFIX_PROVIDERS:
        if model.startswith(prefix):
            return prefix.rstrip("/"), model
    return routed_provider(model, index), model


def _row_provider(key: str, index: Index) -> str | None:
    row = index.rows.get(key)
    return row[0] if row else None


def _is_non_openai_azure_model(name: str, index: Index) -> bool:
    return (
        _row_provider(name, index) == "cohere_chat"
        or _row_provider(f"mistral/{name}", index) == "mistral"
    )


def first_match(
    candidates: Sequence[str],
    provider: str | None,
    rows: dict[str, Row],
    lower: dict[str, str],
) -> tuple[str, Row] | None:
    """The first candidate whose key (exact, else case-insensitive) has a matching provider."""
    for candidate in candidates:
        key = candidate if candidate in rows else lower.get(candidate.lower())
        if key is not None and provider_matches(rows[key][0], provider):
            return key, rows[key]
    return None


def provider_matches(row_provider: str | None, provider: str | None) -> bool:
    """_check_provider_match."""
    if not provider or row_provider is None or row_provider == provider:
        return True
    if provider in ("vertex_ai", "fireworks_ai") and row_provider.startswith(provider):
        return True
    if provider.startswith("bedrock") and row_provider.startswith("bedrock"):
        return True
    if provider == "azure_ai":
        return row_provider in ("azure", "openai")
    return provider in MATCH_ANY_PROVIDERS


def routed_provider(model: str, index: Index) -> str | None:
    """The provider of the first routing rule whose pattern re.search finds in `model`."""
    return next(
        (provider for pattern, provider in index.routing if pattern.search(model)), None
    )


def strip_version(model: str, provider: str | None) -> str:
    """_strip_model_name."""
    if provider in ("bedrock", "bedrock_converse"):
        return bedrock_base_model(model)
    if provider in ("vertex_ai", "gemini", "databricks"):
        return re.sub(r"-\d+$", "", model)
    if "ft:" in model:
        return re.sub(r"(:[^:]*){3}$", "", model)
    return DATED_SUFFIX.sub("", model)


def bedrock_base_model(model: str) -> str:
    """get_bedrock_base_model."""
    head = model
    for prefix in ("bedrock/converse/", "bedrock/", "converse/"):
        if head.startswith(prefix):
            head = head.removeprefix(prefix)
            break
    if head.startswith("nova-2/"):
        return "amazon.nova-2-custom"
    if head.startswith("nova/"):
        return "amazon.nova-custom"
    model = strip_bedrock_routing(model)
    if "arn" in model.lower():
        model = model.split("/")[-1]
    model = re.sub(r"\[\w+\]$", "", re.sub(r"(:\d+):\d+k$", r"\1", model))
    region, dot, rest = model.partition(".")
    if dot and region in BEDROCK_CROSS_REGION:
        return rest
    region, slash, rest = model.partition("/")
    return rest if slash and region in BEDROCK_REGIONS else model


def strip_bedrock_routing(model: str) -> str:
    """strip_bedrock_routing_prefix: each routing prefix, in order, removed once if present."""
    for prefix in BEDROCK_ROUTING_PREFIXES:
        model = model.removeprefix(prefix)
    return model


def strip_mantle_region(model: str) -> str:
    """split_mantle_region_prefix(model)[1]."""
    region, slash, rest = model.partition("/")
    return rest if slash and region in BEDROCK_REGIONS else model


def fireworks_resource(model: str) -> str:
    """resolve_fireworks_resource_name."""
    name = model.removeprefix("fireworks_ai/")
    if name.startswith(FIREWORKS_KEPT_PREFIXES) or "#" in name:
        return name
    if name.startswith(("routers/", "models/")):
        return f"accounts/fireworks/{name}"
    if (
        name.endswith("-fast")
        or name == FIREROUTER
        or name.startswith(f"{FIREROUTER}/")
    ):
        return f"accounts/fireworks/routers/{name}"
    return f"accounts/fireworks/models/{name}"
