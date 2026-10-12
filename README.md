# ctxlen 📏

How many tokens fit? 🤔 Look up the context limits (input and output tokens) of LLMs.

## Install

```shell
uv add ctxlen
```

## Usage

Ask for a model's whole context window:

```python
import ctxlen

ctxlen.ctxlen("claude-opus-5-5")  # 1_000_000
ctxlen.ctxlen("gpt-6-sol")        # 1_050_000
```

litellm ids, provider first, work too:

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
