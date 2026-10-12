"""The context limits of any litellm model id: the latest release's numbers, fetched from GitHub
at most once an hour, else the ones installed with this version."""

import typing as _typing

from ctxlen import _resolve

__all__ = ["ctxlen", "max_input", "max_output"]


def ctxlen(model: str, *, live: bool = True) -> _typing.Optional[int]:  # noqa: FA100
    """The total context window, input and output tokens together, or None.

    `model` is any id you would pass to litellm.completion ("gpt-4o",
    "anthropic/claude-sonnet-4-5", "bedrock/us.anthropic..."), resolved as
    litellm resolves it. The number is models.dev's for the same provider's
    model; where models.dev has none, or one below this model's limits, it is
    max_input(model), raised to max_output(model) where that is larger,
    which is never larger than the true total.

    With live=True, the default, the numbers are the latest release's, fetched
    from GitHub at most once an hour and cached in ~/.cache/ctxlen/. Offline,
    with live=False, or with CTXLEN_OFFLINE=1 set, they are the ones installed
    with this version.
    """
    return _resolve.limits(model, "ctxlen", live)[2]


def max_input(model: str, *, live: bool = True) -> _typing.Optional[int]:  # noqa: FA100 (help() shows Optional[int] on 3.9)
    """The most input tokens `model` accepts, or None if no source knows it.

    `model` is any id you would pass to litellm.completion ("gpt-4o",
    "anthropic/claude-sonnet-4-5", "bedrock/us.anthropic..."), resolved as
    litellm resolves it. The number is litellm's; where litellm has none,
    models.dev's for the same provider's model.

    The sources store what each provider publishes: for some models the input
    cap (gpt-5: 272,000 of a 400,000 window), for others the whole window,
    the reply included (Claude). ctxlen does not reconcile them: leave room
    for the reply.

    With live=True, the default, the numbers are the latest release's, fetched
    from GitHub at most once an hour and cached in ~/.cache/ctxlen/. Offline,
    with live=False, or with CTXLEN_OFFLINE=1 set, they are the ones installed
    with this version.
    """
    return _resolve.limits(model, "max_input", live)[0]


def max_output(model: str, *, live: bool = True) -> _typing.Optional[int]:  # noqa: FA100
    """The most tokens `model` can generate in one reply, or None if no source knows it.

    `model` is any id you would pass to litellm.completion ("gpt-4o",
    "anthropic/claude-sonnet-4-5", "bedrock/us.anthropic..."), resolved as
    litellm resolves it. The number is litellm's max_output_tokens for that
    model; where litellm has none, models.dev's for the same provider's model.
    An embedding model has no output limit, so it gives None.

    With live=True, the default, the numbers are the latest release's, fetched
    from GitHub at most once an hour and cached in ~/.cache/ctxlen/. Offline,
    with live=False, or with CTXLEN_OFFLINE=1 set, they are the ones installed
    with this version.
    """
    return _resolve.limits(model, "max_output", live)[1]
