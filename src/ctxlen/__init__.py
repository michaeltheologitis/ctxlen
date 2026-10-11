"""The context limits of any litellm model id, from bundled data, with no network."""

import typing as _typing

from ctxlen import _resolve

__all__ = ["ctxlen", "max_input", "max_output"]


def ctxlen(model: str) -> _typing.Optional[int]:  # noqa: FA100
    """The total context window, input and output tokens together, or None.

    `model` is any id you would pass to litellm.completion ("gpt-4o",
    "anthropic/claude-sonnet-4-5", "bedrock/us.anthropic..."), resolved as
    litellm resolves it. The number is models.dev's for the same provider's
    model; where models.dev has none, or one below this model's limits, it is
    max_input(model), raised to max_output(model) where that is larger,
    which is never larger than the true total.
    """
    return _resolve.limits(model, "ctxlen")[2]


def max_input(model: str) -> _typing.Optional[int]:  # noqa: FA100 (help() shows Optional[int] on 3.9)
    """The most input tokens `model` accepts, or None if no source knows it.

    `model` is any id you would pass to litellm.completion ("gpt-4o",
    "anthropic/claude-sonnet-4-5", "bedrock/us.anthropic..."), resolved as
    litellm resolves it. The number is litellm's; where litellm has none,
    models.dev's for the same provider's model.

    The sources store what each provider publishes: for some models the input
    cap (gpt-5: 272,000 of a 400,000 window), for others the whole window,
    the reply included (Claude). ctxlen does not reconcile them: leave room
    for the reply.
    """
    return _resolve.limits(model, "max_input")[0]


def max_output(model: str) -> _typing.Optional[int]:  # noqa: FA100
    """The most tokens `model` can generate in one reply, or None if no source knows it.

    `model` is any id you would pass to litellm.completion ("gpt-4o",
    "anthropic/claude-sonnet-4-5", "bedrock/us.anthropic..."), resolved as
    litellm resolves it. The number is litellm's max_output_tokens for that
    model; where litellm has none, models.dev's for the same provider's model.
    An embedding model has no output limit, so it gives None.
    """
    return _resolve.limits(model, "max_output")[1]
