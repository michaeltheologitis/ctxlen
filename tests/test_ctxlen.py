import inspect
import subprocess
import sys

import pytest

import ctxlen
from ctxlen import _resolve
from pipeline.data import ANCHORS

FUNCTIONS = [ctxlen.ctxlen, ctxlen.max_input, ctxlen.max_output]
LIVE_PARAGRAPH = """
With live=True, the default, the numbers are the latest release's, fetched
from GitHub at most once an hour and cached in ~/.cache/ctxlen/. Offline,
with live=False, or with CTXLEN_OFFLINE=1 set, they are the ones installed
with this version."""


def test_public_surface_is_three_functions():
    assert [name for name in dir(ctxlen) if not name.startswith("_")] == [
        "ctxlen",
        "max_input",
        "max_output",
    ]


def test_import_reads_no_data():
    probe = "import ctxlen; print(ctxlen._resolve.shipped_index.cache_info().currsize)"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "0"


@pytest.mark.parametrize("function", FUNCTIONS)
def test_live_is_keyword_only(function):
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        function("gpt-4o", False)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_help_shows_the_live_keyword(function):
    assert (
        str(inspect.signature(function))
        == "(model: str, *, live: bool = True) -> Optional[int]"
    )
    assert inspect.getdoc(function).endswith(LIVE_PARAGRAPH)


@pytest.mark.parametrize("anchor", ANCHORS)
def test_anchors_have_positive_limits(anchor):
    assert ctxlen.max_input(anchor) > 0
    assert ctxlen.ctxlen(anchor) >= ctxlen.max_input(anchor)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("model", ["", "not-a-model", " gpt-4o"])
def test_unknown_and_empty_ids_give_none(function, model):
    assert function(model) is None


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("model", [None, 5, b"gpt-4o"])
def test_non_str_model_raises_type_error_naming_the_function(function, model):
    with pytest.raises(TypeError) as raised:
        function(model)
    assert (
        str(raised.value)
        == f"ctxlen.{function.__name__}() takes a model id as a str, got {model!r}"
    )


def test_a_str_subclass_is_a_model_id():
    class ModelId(str):
        pass

    assert ctxlen.max_input(ModelId("gpt-4o")) == ctxlen.max_input("gpt-4o")


def test_every_shipped_key_answers_an_int_or_none():
    index = _resolve.shipped_index()
    for key in [*index.rows, *index.fills]:
        for function in FUNCTIONS:
            answer = function(key)
            assert answer is None or (type(answer) is int and answer > 0), (key, answer)


def test_total_is_never_below_either_limit():
    index = _resolve.shipped_index()
    for key in [*index.rows, *index.fills]:
        total = ctxlen.ctxlen(key)
        for limit in (ctxlen.max_input(key), ctxlen.max_output(key)):
            assert None in (limit, total) or total >= limit, (key, total, limit)
