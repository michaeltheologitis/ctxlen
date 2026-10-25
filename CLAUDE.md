# ctxlen

A Python library, on PyPI as `ctxlen`, that gives the context limits of any model id litellm accepts: the whole window, the max input and the max output.

Read `../coding-skills/GUIDE.md` and follow it; in the main session a hook has already printed it.
This project's Notion page (todos, changelog, notes): https://app.notion.com/p/40462fb222378151babdd78cc045c6e4

## Goals and non-goals

- Three functions, `ctxlen`, `max_input` and `max_output`, that resolve a model id the way litellm's static lookup does (`src/ctxlen/_resolve.py` ports it) and find the same row as a real litellm release.
- Always fresh: `release.yml` rebuilds the numbers from litellm's map and models.dev every 6 hours and publishes when they change; an install fetches the latest `_data.json` from GitHub at most once an hour, with no upgrade.
- Works offline with the numbers it shipped with, on Python 3.9 and up, with only httpx and certifi as dependencies.
- Not a token counter, a pricing table or a model client. A new runtime dependency or public function needs Michael's yes.

## Checks

- `uv run pytest`: the unit tier, offline; live tests are skipped unless asked for. CI also runs it on 3.9: `uv run --python 3.9 pytest`.
- `uvx ruff check src pipeline tests && uvx ruff format --check src pipeline tests`: lint and format, as CI runs them.
- Every push runs `release.yml` without publishing: it builds the wheel and checks it finds the same rows as a real litellm release (E1, strict on pushes and PRs).
- `pytest -m live`: the live tier against real PyPI, GitHub and healthchecks.io, and it publishes one dev release. Run it by dispatching `ci.yml` on a branch other than main; that sets `GH_TOKEN` and `CTXLEN_LIVE_REF`.

## Conventions

- `src/ctxlen/_data.json` and `reports/` are written by the release workflow; never edit them by hand. A wrong number is fixed upstream or in `pipeline/`.
