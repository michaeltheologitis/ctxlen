import pytest

from ctxlen._resolve import build_index

# A _data.json in miniature: litellm's rows and its two routing rules, with models.dev contexts
# and two fills. A row is [litellm_provider, max_input, max_output, models.dev context].
# fmt: off
FIXTURE = {
    "providers": [
        "openai", "anthropic", "vertex_ai", "gemini", "bedrock", "bedrock_converse",
        "bedrock_mantle", "mistral", "azure", "azure_ai", "ai21", "fireworks_ai",
        "github_copilot", "together_ai", "dashscope", "deepseek", "cohere", "cohere_chat",
        "perplexity", "litellm_proxy", "github", "nano-gpt", "openrouter",
    ],
    "routing": [
        ["^(?:[a-z-]+\\.)?anthropic\\.claude-", "bedrock"],
        ["^claude-[a-z]+-\\d+(?:[-.]\\d+)?(?:-\\d{8})?$", "anthropic"],
    ],
    "ai21_chat": ["jamba-1.5-large"],
    "rows": {
        "gpt-4o": ["openai", 128000, 16384, 128000],
        "gpt-4o-2024-08-06": ["openai", 128000, 16384, None],
        "ft:gpt-4o-2024-08-06": ["openai", 128000, 16384, None],
        "gpt-5": ["openai", 272000, 128000, 400000],
        "gpt-5-pro": ["openai", 400000, 272000, None],
        "gpt-4.1": ["openai", 1047576, 32768, None],
        "text-embedding-3-small": ["openai", 8191, None, None],
        "claude-sonnet-4-5": ["anthropic", 200000, 64000, None],
        "claude-sonnet-4-5-20250929": ["anthropic", 200000, 64000, None],
        "claude-opus-5-5": ["anthropic", 1000000, 128000, 1000000],
        "gemini-2.5-pro": ["vertex_ai-language-models", 1048576, 65535, None],
        "gemini/gemini-2.5-pro": ["gemini", 1048576, 65536, 1048576],
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": ["bedrock_converse", 200000, 64000, None],
        "anthropic.claude-sonnet-4-5-20250929-v1:0": ["bedrock_converse", 200000, 64000, None],
        "anthropic.claude-opus-4-6-v1": ["bedrock_converse", 1000000, 128000, None],
        "amazon.nova-premier-v1:0": ["bedrock_converse", 1000000, 10000, None],
        "bedrock_mantle/anthropic.claude-opus-5-5": ["bedrock_mantle", 1000000, 128000, None],
        "vertex_ai/claude-sonnet-4-5@20250929": ["vertex_ai-anthropic_models", 200000, 64000, None],
        "mistral/mistral-large-4": ["mistral", 1048576, None, 1048576],
        "azure/gpt-4.1": ["azure", 1047576, 32768, None],
        "azure/gpt-35-turbo": ["azure", 4097, 4096, None],
        "jamba-1.5-large": ["ai21", 256000, 256000, None],
        "j2-light": ["ai21", 8192, 8192, None],
        "fireworks_ai/accounts/fireworks/models/deepseek-r1": ["fireworks_ai", 128000, 20480, None],
        "fireworks_ai/accounts/fireworks/routers/auto": ["fireworks_ai", None, None, None],
        "github_copilot/gpt-4o": ["github_copilot", 64000, 4096, None],
        "github_copilot/claude-opus-5.5": ["github_copilot", 200000, 128000, None],
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": ["together_ai", 131072, None, None],
        "dashscope/qwen-flash": ["dashscope", 997952, 32768, None],
        "deepseek-chat": ["deepseek", 131072, 8192, None],
        "command-r-08-2024": ["cohere_chat", 128000, 4096, None],
        "perplexity/sonar": ["perplexity", 128000, None, None],
        "gemini/gemini-2.5-pro-preview-tts": ["gemini", 8192, 16384, None],
    },
    "fills": {
        "dashscope/qwen3.6-plus": ["dashscope", 1000000, 65536, 1000000],
        "nano-gpt/anthropic/claude-opus-4.6": ["nano-gpt", 1000000, 128000, None],
    },
}
# fmt: on


@pytest.fixture
def index():
    return build_index(FIXTURE)
