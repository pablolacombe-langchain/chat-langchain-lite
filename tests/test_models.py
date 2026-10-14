"""Model selection: served model and run metadata must follow CHAT_LANGCHAIN_LITE_MODEL."""

import os
import unittest
from unittest import mock

os.environ.setdefault("LANGSMITH_API_KEY_GATEWAY", "test-key")

with mock.patch("context.get_prompt", return_value=""):
    from agent import agent

from utils.models import MODEL_CONFIG

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"


def _built_model():
    """Return the model object build_agent() hands to create_agent."""
    with mock.patch.object(agent, "create_agent") as create_agent, \
            mock.patch.object(agent, "_readonly_context_hub_fs"):
        agent.build_agent()
    return create_agent.call_args.kwargs["model"]


class ModelSelectionTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"LANGSMITH_API_KEY_GATEWAY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("CHAT_LANGCHAIN_LITE_MODEL", None)

    def _assert_serves(self, model_id):
        self.assertEqual(_built_model().model, model_id)
        self.assertEqual(agent._config()["metadata"]["model"], model_id)

    def test_default_when_env_unset(self):
        self._assert_serves(MODEL_CONFIG["model"])

    def test_haiku_override(self):
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = HAIKU
        self._assert_serves(HAIKU)

    def test_sonnet_override(self):
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = SONNET
        self._assert_serves(SONNET)

    def test_env_change_between_builds(self):
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = HAIKU
        self._assert_serves(HAIKU)
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = SONNET
        self._assert_serves(SONNET)

    def test_gateway_and_max_tokens_preserved(self):
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = HAIKU
        built = _built_model()
        self.assertEqual(built.anthropic_api_url, MODEL_CONFIG["base_url"])
        self.assertEqual(built.max_tokens, 300)
        self.assertEqual(built.temperature, 0)


if __name__ == "__main__":
    unittest.main()
