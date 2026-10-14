"""Centralized model initialization"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

# --- Default: OpenAI, direct ---
# model = init_chat_model("openai:gpt-4.1-mini")

# --- OpenAI via the LangSmith LLM Gateway ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# MODEL_CONFIG is the single source the frontend's Gateway pane reads.
MODEL_CONFIG = {
    "model": "claude-sonnet-4-6",
    "provider": "anthropic",
    "base_url": "https://gateway.smith.langchain.com/anthropic",
}


def model_id() -> str:
    """Model ID to serve: CHAT_LANGCHAIN_LITE_MODEL if set, else MODEL_CONFIG's default."""
    return os.getenv("CHAT_LANGCHAIN_LITE_MODEL") or MODEL_CONFIG["model"]


# Built per call (not at import) because scripts/setup.py switches
# CHAT_LANGCHAIN_LITE_MODEL between baseline experiments in one process.
def get_model():
    """Build the chat model for the current model_id() via the Gateway."""
    return init_chat_model(
        model=model_id(),
        model_provider=MODEL_CONFIG["provider"],
        base_url=MODEL_CONFIG["base_url"],
        api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
        max_tokens=300,
        temperature=0,
    )

# --- Anthropic ---
# model = init_chat_model("anthropic:claude-sonnet-4-5")

# --- Azure OpenAI ---
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# --- AWS Bedrock ---
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(
#     provider="anthropic",
#     model_id="anthropic.claude-sonnet-4-20250514-v1:0",
# )