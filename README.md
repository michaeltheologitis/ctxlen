# ctxlen 📏

A super minimal project to look up the context limits (input and output tokens) of LLMs. 🤔

- 🪶 Zero dependencies and no network calls: drop it into any project.
- 🔄 Checked for new models every 6 hours.
- 🧰 Just three functions.

## Install

```shell
uv add ctxlen
```

Or with pip:

```shell
pip install ctxlen
```

## Usage

Ask for a model's whole context window:

```python
import ctxlen

ctxlen.ctxlen("claude-opus-5-5")  # 1_000_000
ctxlen.ctxlen("gpt-6-sol")        # 1_050_000
```

You can also put the provider in front, like litellm does:

```python
ctxlen.ctxlen("anthropic/claude-sonnet-5-5")  # 1_000_000
ctxlen.ctxlen("openai/gpt-6-luna")            # 1_050_000
```

Or ask for the input and output limits on their own:

```python
ctxlen.max_input("gpt-6-sol")   # 922_000
ctxlen.max_output("gpt-6-sol")  # 128_000
```

The numbers come from litellm and models.dev; their notices are in [LICENSE](https://github.com/michaeltheologitis/ctxlen/blob/main/LICENSE).
