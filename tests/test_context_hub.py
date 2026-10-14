import unittest

from agent.tools import TOOLS
from utils.context_hub import _SEED_AGENTS_MD

_SEED_TEXT = " ".join(_SEED_AGENTS_MD.split())


class SeedAgentsMdScopeTest(unittest.TestCase):
    def test_identifies_as_langchain_ecosystem_assistant(self):
        self.assertIn("an assistant for the LangChain ecosystem", _SEED_TEXT)
        self.assertNotIn("general-purpose", _SEED_TEXT)
        self.assertNotIn("many other topics", _SEED_TEXT)

    def test_declines_and_redirects_unrelated_requests(self):
        self.assertIn("## Scope", _SEED_TEXT)
        self.assertIn("politely decline", _SEED_TEXT)
        self.assertIn("LangChain-related angle", _SEED_TEXT)
        self.assertNotIn("Never say you don't know", _SEED_TEXT)

    def test_limits_bound_tools_to_langchain_topics(self):
        self.assertNotIn("Do not use any tools", _SEED_TEXT)
        for tool in TOOLS:
            self.assertIn(f"`{tool.name}`", _SEED_TEXT)
        self.assertIn("do not call them for unrelated topics", _SEED_TEXT)


if __name__ == "__main__":
    unittest.main()
