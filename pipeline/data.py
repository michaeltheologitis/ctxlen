"""The build's pure transforms: litellm's map projected, models.dev's fills attached, the gate."""

from __future__ import annotations

import ast
import csv
import io
import json
import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from ctxlen._resolve import build_index, find_row

RESERVED_KEYS = frozenset({"sample_spec", "fallback_generalizations"})
SHRINK_LIMIT = 0.25
ANCHORS = ("gpt-4o", "claude-opus-5-5", "gemini/gemini-2.5-pro")
DISAGREEMENT_RATIO = 0.05
EMBEDDING_TYPES = frozenset({"embedding", "reranking"})

MODELS_DEV_PROVIDERS = {
    "abliteration-ai": "abliteration",
    "ai21": "ai21",
    "aihubmix": "aihubmix",
    "alibaba": "dashscope",
    "amazon-bedrock": "bedrock",
    "anthropic": "anthropic",
    "azure": "azure",
    "azure-cognitive-services": "azure_ai",
    "baseten": "baseten",
    "cerebras": "cerebras",
    "chutes": "chutes",
    "clarifai": "clarifai",
    "cloudflare-workers-ai": "cloudflare",
    "cohere": "cohere",
    "cortecs": "cortecs",
    "crusoe": "crusoe",
    "databricks": "databricks",
    "deepinfra": "deepinfra",
    "deepseek": "deepseek",
    "digitalocean": "gradient_ai",
    "empiriolabs": "empiriolabs",
    "fireworks-ai": "fireworks_ai",
    "friendli": "friendliai",
    "gmicloud": "gmi",
    "google": "gemini",
    "google-vertex": "vertex_ai",
    "google-vertex-anthropic": "vertex_ai",
    "groq": "groq",
    "helicone": "helicone",
    "inception": "inception",
    "llama": "meta_llama",
    "meta": "meta",
    "minimax": "minimax",
    "mistral": "mistral",
    "modelscope": "modelscope",
    "moonshotai": "moonshot",
    "morph": "morph",
    "nano-gpt": "nano-gpt",
    "nebius": "nebius",
    "nova": "amazon_nova",
    "novita-ai": "novita",
    "nvidia": "nvidia_nim",
    "oci": "oci",
    "openai": "openai",
    "ovhcloud": "ovhcloud",
    "perplexity": "perplexity",
    "poe": "poe",
    "sap-ai-core": "sap",
    "sarvam": "sarvam",
    "scaleway": "scaleway",
    "scx-ai": "scx-ai",
    "snowflake-cortex": "snowflake",
    "synthetic": "synthetic",
    "tencent-tokenhub": "tencent",
    "togetherai": "together_ai",
    "volcengine": "volcengine",
    "wandb": "wandb",
    "watsonx": "watsonx",
    "xai": "xai",
    "xiaomi": "xiaomi_mimo",
    "zai": "zai",
}
CATALOG_ONLY = {"vercel": "vercel_ai_gateway", "github-copilot": "github_copilot"}
NEVER_FILL = {
    "openrouter": "its terms (§7(5), §12) forbid scripted copying, and models.dev's entries are such copies",
    "vercel": "its terms license use for internal purposes only and bar making the Services available to others",
    "huggingface": "the router picks the host, and hosts serve different windows for one model",
    "venice": 'its terms bar use of its "compilation, information, data" and access by any script or scraper',
    "github-copilot": "it gives the lab's full window, while Copilot caps the prompt lower",
}


class DataError(Exception):
    """Upstream data ctxlen will not ship; the message names the key and the field."""


class GateError(Exception):
    """Built data that fails the gate; the message is the reason."""


class Entry(NamedTuple):
    source: str
    model: str
    slug: str
    max_input: int | None
    max_output: int | None
    context: int | None
    padding: bool


class Disagreement(NamedTuple):
    key: str
    field: str
    litellm: int
    models_dev: int
    source: str


def read_provider_enum(source: str) -> list[str]:
    """The str constants assigned in `class LlmProviders`, read with ast and never imported."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "LlmProviders":
            members = [
                statement.value.value
                for statement in node.body
                if isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ]
            if members:
                return members
    raise DataError("litellm/types/utils.py: no str members in class LlmProviders")


def positive_int(value: object, where: str) -> int | None:
    """None or 0 -> None; a positive int, or a float with an integral value -> int; else DataError."""
    if value is None or (type(value) in (int, float) and value == 0):
        return None
    if type(value) is int and value > 0:
        return value
    if type(value) is float and value > 0 and value.is_integer():
        return int(value)
    raise DataError(f"{where}: {value!r} is not a positive whole number")


def project_litellm(
    cost_map: dict, enum: Sequence[str], registry: Mapping[str, object]
) -> dict:
    """litellm's map in _data.json's layout, with no fills yet. Never reads max_tokens."""
    rules = cost_map.get("fallback_generalizations", {}).get("rules", [])
    rows = {}
    for key, value in cost_map.items():
        if key in RESERVED_KEYS:
            continue
        if not isinstance(value, dict) or not isinstance(
            value.get("litellm_provider"), str
        ):
            raise DataError(
                f"{key}.litellm_provider: the row has no str litellm_provider"
            )
        rows[key] = [
            value["litellm_provider"],
            positive_int(value.get("max_input_tokens"), f"{key}.max_input_tokens"),
            positive_int(value.get("max_output_tokens"), f"{key}.max_output_tokens"),
            None,
        ]
    return {
        "providers": sorted(set(enum) | set(registry)),
        "routing": [
            [rule["pattern"], rule["model_info"]["litellm_provider"]]
            for rule in rules
            if isinstance(rule, dict)
            and isinstance(rule.get("pattern"), str)
            and isinstance(rule.get("model_info", {}).get("litellm_provider"), str)
        ],
        "ai21_chat": [
            key
            for key, value in cost_map.items()
            if key not in RESERVED_KEYS
            and value["litellm_provider"] == "ai21"
            and value.get("mode") == "chat"
        ],
        "rows": rows,
        "fills": {},
    }


def models_dev_entries(
    api: dict, providers: Mapping[str, str]
) -> tuple[list[Entry], int]:
    """models.dev's text models of `providers`, as litellm ids, sorted by source; and how many were skipped."""
    entries, skipped = [], 0
    for provider, slug in providers.items():
        for model_id, model in api.get(provider, {}).get("models", {}).items():
            if "text" not in model.get("modalities", {}).get("output", []):
                continue
            if model.get("status") == "deprecated":
                continue
            source = f"{provider}/{model_id}"
            if provider.startswith("google-vertex"):
                model_id = re.sub(r"@default$", "", model_id)
            limit = model.get("limit", {})
            try:
                context = positive_int(limit.get("context"), f"{source}.limit.context")
                max_input = (
                    positive_int(limit.get("input"), f"{source}.limit.input") or context
                )
                max_output = positive_int(limit.get("output"), f"{source}.limit.output")
                if is_embedding(model_id, model):
                    max_output = None
            except DataError:
                skipped += 1
                continue
            padding = bool(max_output and context and max_output >= context)
            if padding:
                max_output = None
            litellm_id = f"{slug}/{model_id}"
            entries.append(
                Entry(source, litellm_id, slug, max_input, max_output, context, padding)
            )
    return sorted(entries), skipped


def is_embedding(model_id: str, model: dict) -> bool:
    """An embedding or reranking model, whose models.dev limit.output is a vector size, not tokens."""
    named = f"{model_id} {model.get('family') or ''}".lower()
    return model.get("type") in EMBEDDING_TYPES or "embed" in named


def same_family(row_provider: str, slug: str) -> bool:
    """Equal; or the row's provider is slug + "-..."; or the pair bedrock and bedrock_converse."""
    return (
        row_provider == slug
        or row_provider.startswith(f"{slug}-")
        or {row_provider, slug} == {"bedrock", "bedrock_converse"}
    )


def attach_fills(
    data: dict, entries: Sequence[Entry]
) -> tuple[dict, list[Disagreement], dict]:
    """models.dev's entries merged into litellm's rows, the disagreements, and counts.

    Each entry is resolved as ctxlen resolves an id, over litellm's rows only. A hit on a row
    of the same family fills the row's missing fields and its context; where both have a
    number and they differ by more than DISAGREEMENT_RATIO, litellm's stays and the pair is
    a disagreement. A hit on another family adds nothing, so a host's number never lands on
    a lab's row. A miss becomes a fill row. The first entry wins each field. Last, a context
    below its row's own limits is dropped and reported. So a fill never changes a number
    litellm gives.
    """
    litellm_only = build_index({**data, "fills": {}})
    rows = {key: list(row) for key, row in data["rows"].items()}
    fills, disagreements, context_source = {}, [], {}
    counts = {"field_fills_input": 0, "field_fills_output": 0, "dropped_padding": 0}
    for entry in entries:
        counts["dropped_padding"] += entry.padding
        hit = find_row(entry.model, litellm_only)
        if hit is None:
            if entry.model not in fills and (entry.max_input or entry.max_output):
                # slug, max_input, max_output, context: the entry's row
                fills[entry.model] = list(entry[2:6])
                context_source[entry.model] = entry.source
            continue
        key, litellm_row = hit
        if not same_family(litellm_row[0], entry.slug):
            continue
        row = rows[key]
        if entry.context is not None and row[3] is None:
            row[3] = entry.context
            context_source[key] = entry.source
        for position, field in ((1, "max_input"), (2, "max_output")):
            value, theirs = getattr(entry, field), litellm_row[position]
            if value is None:
                continue
            if theirs is None and row[position] is None:
                row[position] = value
                counts[f"field_fills_{field.removeprefix('max_')}"] += 1
            elif theirs and abs(value - theirs) > DISAGREEMENT_RATIO * theirs:
                disagreements.append(
                    Disagreement(key, field, theirs, value, entry.source)
                )
    for key, row in [*rows.items(), *fills.items()]:
        bound = max((limit for limit in row[1:3] if limit is not None), default=0)
        if row[3] is not None and row[3] < bound:
            source = context_source.get(key, "")
            disagreements.append(Disagreement(key, "context", bound, row[3], source))
            row[3] = None
    disagreements.sort(key=lambda d: (d.key, d.field, d.source))
    merged = {**data, "rows": rows, "fills": dict(sorted(fills.items()))}
    return merged, disagreements, counts


def check_gate(data: dict, previous: dict | None) -> None:
    """GateError on a bad value, a shrink past SHRINK_LIMIT, an anchor without max_input, or a bad pattern."""
    for part in ("rows", "fills"):
        for key, row in data[part].items():
            if not all(
                value is None or (type(value) is int and value > 0) for value in row[1:]
            ):
                raise GateError(
                    f"gate: {key} has a value that is not a positive int or null: {row}"
                )
    for part in ("rows", "fills", "providers"):
        if previous is not None and len(data[part]) < (1 - SHRINK_LIMIT) * len(
            previous[part]
        ):
            raise GateError(
                f"gate: {part} shrank from {len(previous[part])} in the last release to {len(data[part])}, "
                f"more than {SHRINK_LIMIT:.0%}"
            )
    try:
        index = build_index(data)
    except re.error as error:
        raise GateError(f"gate: a routing pattern does not compile: {error}") from error
    for anchor in ANCHORS:
        hit = find_row(anchor, index)
        if hit is None or hit[1][1] is None:
            raise GateError(f"gate: the anchor {anchor} has no max_input")


def catalog_ids(api: dict) -> list[str]:
    """models.dev's text models of MODELS_DEV_PROVIDERS and CATALOG_ONLY, as litellm ids; E2's catalog."""
    entries, _ = models_dev_entries(api, {**MODELS_DEV_PROVIDERS, **CATALOG_ONLY})
    return sorted({entry.model for entry in entries})


def coverage(projected: dict, merged: dict, catalog: Sequence[str]) -> dict:
    """How many catalog ids get a max_input from litellm alone and from ctxlen, and where
    ctxlen's total comes from: models.dev's context, litellm's limits, or nowhere."""

    def rows(data: dict) -> list[list]:
        index = build_index(data)
        return [(find_row(model, index) or (None, [None] * 4))[1] for model in catalog]

    ctxlen = rows(merged)
    totals = ["md" if row[3] else "litellm" if row[1] else "none" for row in ctxlen]
    return {
        "catalog": len(catalog),
        "litellm_alone": sum(row[1] is not None for row in rows(projected)),
        "ctxlen": sum(row[1] is not None for row in ctxlen),
        "total_from_models_dev": totals.count("md"),
        "total_from_litellm": totals.count("litellm"),
        "total_none": totals.count("none"),
    }


def dump_data(data: dict) -> str:
    """_data.json's layout: one row per line, so a release's diff reads like litellm's."""

    def compact(value: object) -> str:
        return json.dumps(value, separators=(",", ":"))

    def table(rows: dict) -> str:
        return ",\n".join(f"{compact(key)}:{compact(row)}" for key, row in rows.items())

    head = ",\n".join(
        f"{compact(name)}:{compact(data[name])}"
        for name in ("providers", "routing", "ai21_chat")
    )
    return f'{{{head},\n"rows":{{\n{table(data["rows"])}}},\n"fills":{{\n{table(data["fills"])}}}}}\n'


def disagreements_csv(rows: Sequence[Disagreement]) -> str:
    """CSV key,field,litellm,models_dev,source, sorted."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(Disagreement._fields)
    writer.writerows(sorted(rows))
    return out.getvalue()
