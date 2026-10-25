# ctxlen

A Python library, on PyPI as `ctxlen`, that gives the context limits of any model id litellm accepts: the whole window, the max input and the max output.

Read `../coding-skills/GUIDE.md` and follow it; in the main session a hook has already printed it.
This project's Notion page (todos, changelog, notes): https://app.notion.com/p/40462fb222378151babdd78cc045c6e4

## Goals and non-goals

- Three functions, `ctxlen`, `max_input` and `max_output`, that resolve a model id the way litellm's static lookup does (`src/ctxlen/_resolve.py` ports it) and find the same row as a real litellm release.
- Always fresh: every 6 hours `release.yml` rebuilds the numbers from litellm's map and models.dev, and publishes when the built package (numbers or code) differs from the latest release; an install fetches the latest `_data.json` from GitHub at most once an hour, with no upgrade.
- Works offline with the numbers it shipped with, on Python 3.9 and up, with only httpx and certifi as dependencies.
- Not a token counter, a pricing table or a model client. A new runtime dependency or public function needs Michael's yes.

## Checks

- `uv run pytest`: the unit tier, offline; live tests are skipped unless asked for. CI also runs it on 3.9: `uv run --python 3.9 pytest`.
- `uvx ruff check src pipeline tests && uvx ruff format --check src pipeline tests`: lint and format, as CI runs them.
- Every push runs `release.yml` without publishing: it builds the wheel from live litellm and models.dev data and checks it finds the same rows as a real litellm release (E1, strict on pushes and PRs), so it can go red from an upstream change.
- `gh workflow run ci.yml --ref <branch>`, never on main: the live tier, `pytest -m live` against real PyPI, GitHub and healthchecks.io. It publishes a dev release and tag, updates the `ctxlen-release` issue, and sends healthchecks.io a fail ping.

## Conventions

- Pushes never publish, so a change merged to main ships as a final version at the next scheduled run.
- `src/ctxlen/_data.json`, `reports/` and the version are written by the release workflow; never edit them by hand. `pyproject.toml` stays at `0.0.0`, and a wrong number is fixed upstream or in `pipeline/`.
