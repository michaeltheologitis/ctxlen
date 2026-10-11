# ctxlen

A simple library to look up the context limits (input and output tokens) of LLMs.

```shell
uv add ctxlen
```

```pycon
>>> import ctxlen
>>> ctxlen.ctxlen("claude-opus-5-5")
1000000
>>> ctxlen.ctxlen("anthropic/claude-sonnet-5-5")
1000000
>>> ctxlen.ctxlen("gpt-6-sol")
1050000
>>> ctxlen.ctxlen("openai/gpt-6-luna")
1050000
>>> ctxlen.max_input("gpt-6-sol"), ctxlen.max_output("gpt-6-sol")
(922000, 128000)

```

The numbers come from litellm and models.dev; their notices are in [LICENSE](LICENSE).
