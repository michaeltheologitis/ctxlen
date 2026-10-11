"""E1: ctxlen's built wheel and litellm release R find the same row, over the same JSON.

`python -m pipeline.conformance`, in a venv holding litellm==R and the wheel. litellm reads its
map from a local server that serves this run's inputs/litellm.json, and check_loaded proves that
it did: litellm falls back to its bundled copy silently.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import threading
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from pathlib import Path
from types import ModuleType

from ctxlen import _resolve
from pipeline.build import output
from pipeline.data import RESERVED_KEYS, catalog_ids

NON_MODELS: tuple[str, ...] = (
    "",
    " ",
    "gpt-4o ",
    " gpt-4o",
    "/",
    "//",
    "openai/",
    "/gpt-4o",
    "anthropic/",
    "bedrock/",
    "not-a-model",
    "gpt",
    "gpt-4o-mini-but-not-really",
    "azure/my-prod-gpt4",
    "azure/",
    "azure_ai/my-deployment",
    "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/openai/gpt-4o",
    "claude-opus-9",
    "claude-sonnet-99-9",
    "claude",
    "gemini-99-ultra",
    "vertex_ai/not-a-model",
    "bedrock/not-a-model",
    "bedrock/us-east-1/not-a-model",
    "bedrock/us.anthropic.claude-opus-99-v1:0",
    "fireworks_ai/not-a-model",
    "unknownprovider/gpt-4o",
    "acme/gpt-4o",
    "my-company/llama-3-70b",
    "ollama",
    "huggingface",
    "openai",
    "*",
    "**",
    ":",
    "gpt-4o:",
    "ft:",
    "ft:gpt-4o",
    "ft:gpt-4o:::",
    "a" * 300,
    "openai/" + "x" * 200,
    "gpt-4o\n",
    "gpt-4o\t",
    "GPT-4O-2099-01-01",
    "модель",
    "模型/gpt-4o",
    "gpt-4o\u200b",
    "e\u0301",
    "\U0001f916",
)
SAMPLE_SIZE = 300
NETWORK_PROVIDERS = frozenset({"huggingface"})
# Each constant _resolve ports by hand, and how to read litellm's own value of it.
DRIFT_CHECKS = {
    "AZURE_ALIASES": lambda m: {**m.azure_llms, **m.azure_embedding_models},
    "AUTHENTICATING_PROVIDERS": lambda m: frozenset(
        m.constants.PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO
    ),
    "REPLICATE_VERSION_LENGTH": lambda m: (
        m.constants.REPLICATE_MODEL_NAME_WITH_ID_LENGTH
    ),
    "BEDROCK_CROSS_REGION": lambda m: frozenset(
        m.llms.bedrock.common_utils.get_bedrock_cross_region_inference_regions()
    ),
    "BEDROCK_REGIONS": lambda m: frozenset(
        m.llms.bedrock.common_utils.AmazonBedrockGlobalConfig().get_all_regions()
    ),
    "FIREROUTER": lambda m: m.llms.fireworks_ai.common_utils.FIREROUTER,
    "FIREWORKS_KEPT_PREFIXES": lambda m: (
        "accounts/",
        m.llms.fireworks_ai.common_utils.AZURE_FOUNDRY_FIREWORKS_MODEL_ID_PREFIX,
    ),
}
GROUPS = ("key", "prefixed", "catalog", "variant", "non_model")
E1_RATIO = 0.995


def corpus(
    cost_map: dict, api: dict, providers: Collection[str]
) -> list[tuple[str, str]]:
    """(group, id), deduplicated keeping the first group: key, prefixed, catalog, variant, non_model."""
    keys = [key for key in cost_map if key not in RESERVED_KEYS]
    provider = {key: cost_map[key]["litellm_provider"] for key in keys}
    groups = {
        "key": keys,
        "prefixed": [
            prefixed_form(key, provider[key], providers)
            for key in keys
            if "/" not in key
        ],
        "catalog": catalog_ids(api),
        "variant": [
            variant
            for key in sample_keys(keys)
            for variant in variants(key, provider[key])
        ],
        "non_model": NON_MODELS,
    }
    seen, items = set(), []
    for group, models in groups.items():
        for model in models:
            if model is not None and model not in seen:
                seen.add(model)
                items.append((group, model))
    return items


def prefixed_form(
    key: str, row_provider: str, providers: Collection[str]
) -> str | None:
    """`key` behind the prefix a caller writes for its provider, or None if litellm knows no such prefix."""
    prefix = row_provider
    for family in ("vertex_ai", "fireworks_ai"):
        if row_provider.startswith(f"{family}-"):
            prefix = family
    if row_provider == "bedrock_converse":
        prefix = "bedrock"
    return f"{prefix}/{key}" if prefix in providers else None


def sample_keys(keys: Iterable[str], n: int = SAMPLE_SIZE) -> list[str]:
    """The n keys with the lowest sha256: a sample that changes only as the keys do."""
    return sorted(keys, key=lambda key: hashlib.sha256(key.encode()).hexdigest())[:n]


def variants(key: str, row_provider: str) -> list[str]:
    """Spellings of `key` that litellm strips back to it: a date, -001, upper case, Bedrock routes, ft:."""
    found = [f"{key}-2099-01-01", f"{key}-001", key.upper()]
    if row_provider.startswith("bedrock"):
        found += [f"bedrock/converse/{key}", f"bedrock/us-west-2/{key}"]
    if row_provider == "openai":
        found.append(f"ft:{key}:acme::abc123")
    return found


def check_loaded(model_cost: Mapping, cost_map: Mapping) -> list[str]:
    """The proof that litellm loaded `cost_map`: [] when it did, else up to 20 mismatches."""
    expected = {
        key: value
        for key, value in cost_map.items()
        if key != "fallback_generalizations"
    }
    fields = ("litellm_provider", "max_input_tokens", "max_output_tokens")
    mismatches = [f"missing: {key}" for key in expected if key not in model_cost]
    mismatches += [f"extra: {key}" for key in model_cost if key not in expected]
    mismatches += [
        f"changed: {key}.{field}: {model_cost[key].get(field)!r} != {value.get(field)!r}"
        for key, value in expected.items()
        if key in model_cost
        for field in fields
        if model_cost[key].get(field) != value.get(field)
    ]
    return mismatches[:20]


def litellm_answer(
    get_model_info: Callable, keys_lower: Collection[str], model: str
) -> str | None:
    """litellm's key for `model`; "rule" when a pattern rule answered; None when it raised."""
    try:
        key = get_model_info(model)["key"]
    except Exception:  # noqa: BLE001 (litellm raises a bare Exception for every unmapped id: no row)
        return None
    return key if isinstance(key, str) and key.lower() in keys_lower else "rule"


def compare(
    corpus: Sequence[tuple[str, str]],
    ctxlen_key: Callable[[str], str | None],
    litellm_key: Callable[[str], str | None],
    skip: Callable[[str], bool],
) -> dict:
    """Per-group agreement, the exclusions, the differences, and E1's verdict: it holds when
    every key and prefixed id agrees, E1_RATIO of the rest do, and none is ctxlen's alone."""
    groups = {group: {"n": 0, "equal": 0} for group in GROUPS}
    excluded = {"pattern_rule": 0, "network": 0}
    differences, ctxlen_only = [], 0
    for group, model in corpus:
        if skip(model):
            excluded["network"] += 1
            continue
        theirs = litellm_key(model)
        if theirs == "rule":
            excluded["pattern_rule"] += 1
            continue
        ours = ctxlen_key(model)
        groups[group]["n"] += 1
        if ours == theirs:
            groups[group]["equal"] += 1
            continue
        differences.append(
            {"id": model, "group": group, "ctxlen": ours, "litellm": theirs}
        )
        ctxlen_only += ours is not None and theirs is None
    rest = [groups[group] for group in ("catalog", "variant", "non_model")]
    rest_n = sum(counts["n"] for counts in rest)
    ratio = sum(counts["equal"] for counts in rest) / rest_n if rest_n else 1.0
    keys_equal = all(groups[g]["n"] == groups[g]["equal"] for g in ("key", "prefixed"))
    return {
        "corpus": len(corpus),
        "groups": groups,
        "excluded": excluded,
        "differences": differences,
        "ctxlen_only": ctxlen_only,
        "e1": {
            "keys_and_prefixed_equal": keys_equal,
            "rest_equal_ratio": ratio,
            "holds": keys_equal and ratio >= E1_RATIO and ctxlen_only == 0,
        },
    }


def drift(litellm: ModuleType) -> list[str]:
    """Each hand-ported constant that differs from litellm's live value, or is unavailable."""
    report = []
    for name, read in DRIFT_CHECKS.items():
        ours = getattr(_resolve, name)
        try:
            theirs = read(litellm)
        except (AttributeError, ImportError, TypeError):
            report.append(f"unavailable: {name}")
            continue
        if theirs != ours:
            report.append(
                f"changed: {name}: ctxlen has {ours!r}, litellm has {theirs!r}"
            )
    return report


def serve(directory: Path) -> str:
    """Serves `directory` at a local URL from a daemon thread; returns the URL of its litellm.json."""
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/litellm.json"


def main(argv: list[str] | None = None) -> int:
    """Serves inputs/litellm.json locally, imports litellm, proves the load, compares, reports."""
    parser = argparse.ArgumentParser(prog="python -m pipeline.conformance")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    cost_map = json.loads((args.inputs / "litellm.json").read_bytes())
    os.environ["LITELLM_MODEL_COST_MAP_URL"] = serve(args.inputs)
    # Imported only now, so that litellm reads the served map; drift reads the submodules.
    import litellm
    import litellm.constants
    import litellm.llms.bedrock.common_utils
    import litellm.llms.fireworks_ai.common_utils

    litellm.suppress_debug_info = True
    mismatches = check_loaded(litellm.model_cost, cost_map)
    if mismatches:
        args.report.write_text(
            json.dumps({"loaded": False, "mismatches": mismatches}, indent=1)
        )
        print(
            "litellm did not load the pinned map:",
            *mismatches,
            sep="\n  ",
            file=sys.stderr,
        )
        return 2
    index = _resolve.shipped_index()._replace(fills={}, fill_lower={})
    api = json.loads((args.inputs / "models_dev.json").read_text())
    keys_lower = {key.lower() for key in litellm.model_cost}

    def ctxlen_key(model: str) -> str | None:
        hit = _resolve.find_row(model, index)
        return hit[0] if hit else None

    def skip(model: str) -> bool:
        return _resolve.candidate_keys(model, index)[1] in NETWORK_PROVIDERS

    report = compare(
        corpus(cost_map, api, index.providers),
        ctxlen_key,
        lambda model: litellm_answer(litellm.get_model_info, keys_lower, model),
        skip,
    )
    report = {
        "loaded": True,
        "litellm": metadata.version("litellm"),
        **report,
        "drift": drift(litellm),
    }
    args.report.write_text(json.dumps(report, indent=1))
    shown = ("groups", "excluded", "ctxlen_only", "drift", "e1")
    print(json.dumps({key: report[key] for key in shown}, indent=1))
    for difference in report["differences"][:50]:
        print("differs:", json.dumps(difference))
    output(differences=len(report["differences"]))
    return 1 if args.strict and (report["differences"] or report["drift"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
