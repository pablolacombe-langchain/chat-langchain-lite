"""Tests for online evaluator setup in scripts/setup.py."""

import unittest
from unittest import mock

import requests

from scripts import setup


def _response(status_code: int, body=None) -> mock.Mock:
    resp = mock.Mock(status_code=status_code, text="")
    resp.json.return_value = body
    return resp


class CreateOnlineEvaluatorTest(unittest.TestCase):
    def test_every_rule_is_root_only(self):
        with mock.patch.object(setup.requests, "post", return_value=_response(200, {"id": "r"})) as post, \
                mock.patch("builtins.print"):
            for ev in setup.EVALUATORS:
                setup.create_online_evaluator("key", ev, "project", {"model": "m"})

        self.assertEqual(post.call_count, len(setup.EVALUATORS))
        for call in post.call_args_list:
            self.assertEqual(call.kwargs["json"]["filter"], "eq(is_root, true)")
            self.assertIsNone(call.kwargs["json"].get("trace_filter"))


class EvaluatorModelTest(unittest.TestCase):
    def test_model_routes_through_gateway_with_secret_reference(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}):
            kwargs = setup.evaluator_model_json()["kwargs"]

        self.assertEqual(kwargs["anthropic_api_url"], "https://gateway.smith.langchain.com/anthropic")
        self.assertEqual(kwargs["anthropic_api_key"], {"lc": 1, "type": "secret", "id": ["ANTHROPIC_API_KEY"]})

    def test_secret_present(self):
        with mock.patch.object(setup.requests, "get", return_value=_response(200, [{"key": "ANTHROPIC_API_KEY"}])), \
                mock.patch("builtins.print"):
            self.assertTrue(setup.check_evaluator_secret("key"))

    def test_secret_missing_is_reported(self):
        with mock.patch.object(setup.requests, "get", return_value=_response(200, [{"key": "OPENAI_API_KEY"}])), \
                mock.patch("builtins.print") as printed:
            self.assertFalse(setup.check_evaluator_secret("key"))
        output = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("ANTHROPIC_API_KEY", output)
        self.assertIn("Settings", output)

    def test_secret_lookup_failure_is_reported(self):
        with mock.patch.object(setup.requests, "get", return_value=_response(403)), \
                mock.patch("builtins.print") as printed:
            self.assertFalse(setup.check_evaluator_secret("key"))
        self.assertTrue(printed.called)


class CheckOnlineEvaluatorsTest(unittest.TestCase):
    def _check(self, response):
        with mock.patch.object(setup.requests, "get", **response) as get, mock.patch("builtins.print"):
            failures = setup.check_online_evaluators("key", ["rule-1"])
        self.assertIn("/runs/rules/rule-1/logs", get.call_args.args[0])
        return failures

    def test_success_is_not_a_failure(self):
        logs = [{"evaluators": None}, {"evaluators": {"outcome": "success", "payload": {}}}]
        self.assertEqual(self._check({"return_value": _response(200, logs)}), [])

    def test_no_evaluated_runs_is_not_a_failure(self):
        self.assertEqual(self._check({"return_value": _response(200, [{"evaluators": None}])}), [])

    def test_evaluator_error_is_reported(self):
        logs = [
            {"evaluators": None},
            {"evaluators": {"outcome": "error", "payload": {"errors": ["{'error': 'Anthropic authentication failed'}"]}}},
        ]
        failures = self._check({"return_value": _response(200, logs)})
        self.assertEqual(len(failures), 1)
        self.assertIn("Anthropic authentication failed", failures[0])

    def test_log_request_failure_is_reported(self):
        self.assertEqual(len(self._check({"return_value": _response(500)})), 1)
        self.assertEqual(len(self._check({"side_effect": requests.ConnectionError("boom")})), 1)


if __name__ == "__main__":
    unittest.main()
