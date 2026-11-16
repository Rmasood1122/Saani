"""The provider clients are tested against a local fake of each API: real HTTP, real JSON."""
import http.server
import json
import socketserver
import threading
import unittest

import tests.helpers  # noqa: F401  (sets NO_PROXY for localhost)
from saani import llm
from saani.llm import (AnthropicClient, Budget, BudgetExceeded, LLMError, LLMRouter, Msg, OpenAICompatClient,
                       ToolCall, extract_json)

TOOL = {"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}


class FakeAPI:
    def __init__(self, responder):
        self.requests = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                status, payload, headers = responder(len(outer.requests), body)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a): pass
        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()


class Base(unittest.TestCase):
    def setUp(self):
        self._sleep = llm._sleep
        llm._sleep = lambda s: None
        self.addCleanup(setattr, llm, "_sleep", self._sleep)

    def api(self, responder):
        f = FakeAPI(responder)
        self.addCleanup(f.close)
        return f


CONVO = [Msg("user", "weather in Karachi?"),
         Msg("assistant", "Let me check.", [ToolCall("tu_1", "get_weather", {"city": "Karachi"})]),
         Msg("tool", "31C sunny", tool_call_id="tu_1"),
         Msg("tool", "boom", tool_call_id="tu_2", is_error=True)]


class Anthropic(Base):
    def ok(self, n, body):
        return 200, {"content": [{"type": "thinking", "thinking": "hmm", "signature": "sig"},
                                 {"type": "text", "text": "Checking."},
                                 {"type": "tool_use", "id": "tu_9", "name": "get_weather", "input": {"city": "Lahore"}}],
                     "stop_reason": "tool_use", "usage": {"input_tokens": 12, "output_tokens": 7}}, {}

    def test_request_and_response_shapes(self):
        f = self.api(self.ok)
        r = AnthropicClient("key-123", f.url).complete(system="be brief", messages=CONVO, tools=[TOOL], model="claude-sonnet-5-5",
                                                       max_tokens=500, force_tool="get_weather")
        req = f.requests[0]
        self.assertEqual(req["path"], "/v1/messages")
        self.assertEqual((req["headers"]["x-api-key"], req["headers"]["anthropic-version"]), ("key-123", "2023-06-01"))
        b = req["body"]
        self.assertEqual((b["model"], b["max_tokens"]), ("claude-sonnet-5-5", 500))
        self.assertEqual(b["system"][0]["text"], "be brief")
        self.assertEqual(b["tools"][0]["input_schema"]["type"], "object")
        self.assertEqual(b["tool_choice"], {"type": "tool", "name": "get_weather"})
        self.assertEqual([m["role"] for m in b["messages"]], ["user", "assistant", "user"])       # tool results merged into one user turn
        self.assertEqual(b["messages"][1]["content"][1], {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Karachi"}})
        results = b["messages"][2]["content"]
        self.assertEqual([x["tool_use_id"] for x in results], ["tu_1", "tu_2"])
        self.assertTrue(results[1]["is_error"])
        self.assertEqual((r.text, r.stop_reason, r.input_tokens, r.output_tokens), ("Checking.", "tool_use", 12, 7))
        self.assertEqual((r.tool_calls[0].id, r.tool_calls[0].name, r.tool_calls[0].args), ("tu_9", "get_weather", {"city": "Lahore"}))

    def test_thinking_blocks_are_replayed_verbatim(self):
        f = self.api(self.ok)
        c = AnthropicClient("k", f.url)
        r = c.complete(system="", messages=[Msg("user", "hi")], tools=[TOOL], model="m")
        follow = [Msg("user", "hi"), Msg("assistant", r.text, r.tool_calls, raw=r.raw), Msg("tool", "20C", tool_call_id="tu_9")]
        c.complete(system="", messages=follow, tools=[TOOL], model="m")
        replayed = f.requests[1]["body"]["messages"][1]["content"]
        self.assertEqual(replayed[0], {"type": "thinking", "thinking": "hmm", "signature": "sig"})
        self.assertEqual(replayed[2]["type"], "tool_use")
        self.assertNotIn("system", f.requests[1]["body"])

    def test_retries_overload_then_succeeds(self):
        f = self.api(lambda n, b: (529, {"error": "overloaded"}, {}) if n < 3 else self.ok(n, b))
        r = AnthropicClient("k", f.url, max_retries=4).complete(system="", messages=[Msg("user", "x")], model="m")
        self.assertEqual((len(f.requests), r.text), (3, "Checking."))

    def test_gives_up_after_retries_and_does_not_retry_auth_errors(self):
        f = self.api(lambda n, b: (500, {"error": "x"}, {}))
        with self.assertRaises(LLMError):
            AnthropicClient("k", f.url, max_retries=2).complete(system="", messages=[Msg("user", "x")], model="m")
        self.assertEqual(len(f.requests), 3)
        f2 = self.api(lambda n, b: (401, {"error": "bad key"}, {}))
        with self.assertRaises(LLMError) as cm:
            AnthropicClient("k", f2.url).complete(system="", messages=[Msg("user", "x")], model="m")
        self.assertEqual(len(f2.requests), 1)
        self.assertIn("401", str(cm.exception))

    def test_forced_tool_rejected_falls_back_to_instruction(self):
        def responder(n, body):
            if "tool_choice" in body:
                return 400, {"error": {"message": "tool_choice not allowed with thinking"}}, {}
            return self.ok(n, body)
        f = self.api(responder)
        r = AnthropicClient("k", f.url).complete(system="plan", messages=[Msg("user", "x")], tools=[TOOL], model="m", force_tool="get_weather")
        self.assertEqual(r.tool_calls[0].name, "get_weather")
        self.assertNotIn("tool_choice", f.requests[1]["body"])
        self.assertIn("must respond by calling the `get_weather` tool", f.requests[1]["body"]["system"][-1]["text"])

    def test_unreachable_server(self):
        with self.assertRaises(LLMError) as cm:
            AnthropicClient("k", "http://127.0.0.1:1", max_retries=1).complete(system="", messages=[Msg("user", "x")], model="m")
        self.assertIn("network error", str(cm.exception))


class OpenAI(Base):
    def ok(self, n, body):
        return 200, {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Multan\"}"}}]}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4}}, {}

    def test_request_and_response_shapes(self):
        f = self.api(self.ok)
        r = OpenAICompatClient("sk-x", f.url).complete(system="be brief", messages=CONVO, tools=[TOOL], model="local-model",
                                                       max_tokens=300, force_tool="get_weather")
        req = f.requests[0]
        self.assertEqual(req["path"], "/chat/completions")
        self.assertEqual(req["headers"]["authorization"], "Bearer sk-x")
        b = req["body"]
        self.assertEqual([m["role"] for m in b["messages"]], ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(b["messages"][2]["tool_calls"][0]["function"], {"name": "get_weather", "arguments": "{\"city\": \"Karachi\"}"})
        self.assertEqual(b["messages"][3], {"role": "tool", "tool_call_id": "tu_1", "content": "31C sunny"})
        self.assertTrue(b["messages"][4]["content"].startswith("ERROR: "))
        self.assertEqual(b["tools"][0], {"type": "function", "function": {"name": "get_weather", "description": "weather", "parameters": TOOL["input_schema"]}})
        self.assertEqual(b["tool_choice"], {"type": "function", "function": {"name": "get_weather"}})
        self.assertEqual(b["max_tokens"], 300)
        self.assertEqual((r.tool_calls[0].id, r.tool_calls[0].args, r.input_tokens, r.output_tokens), ("call_1", {"city": "Multan"}, 9, 4))

    def test_no_key_for_local_servers_and_bad_shapes(self):
        f = self.api(lambda n, b: (200, {"choices": [{"message": {"content": "plain answer"}, "finish_reason": "stop"}]}, {}))
        r = OpenAICompatClient("", f.url).complete(system="", messages=[Msg("user", "x")], model="m")
        self.assertEqual(r.text, "plain answer")
        self.assertNotIn("authorization", f.requests[0]["headers"])
        f2 = self.api(lambda n, b: (200, {"unexpected": True}, {}))
        with self.assertRaises(LLMError):
            OpenAICompatClient("", f2.url).complete(system="", messages=[Msg("user", "x")], model="m")

    def test_malformed_tool_arguments_do_not_crash(self):
        f = self.api(lambda n, b: (200, {"choices": [{"message": {"tool_calls": [{"function": {"name": "t", "arguments": "{not json"}}]}}]}, {}))
        r = OpenAICompatClient("", f.url).complete(system="", messages=[Msg("user", "x")], model="m")
        self.assertEqual(r.tool_calls[0].args, {"_raw": "{not json"})

    def test_rate_limit_retry(self):
        f = self.api(lambda n, b: (429, {"error": "slow down"}, {"Retry-After": "0"}) if n == 1 else self.ok(n, b))
        OpenAICompatClient("", f.url).complete(system="", messages=[Msg("user", "x")], model="m")
        self.assertEqual(len(f.requests), 2)


class RouterAndHelpers(unittest.TestCase):
    def test_budget(self):
        b = Budget(2)
        b.charge(); b.charge()
        with self.assertRaises(BudgetExceeded):
            b.charge()

    def test_extract_json(self):
        self.assertEqual(extract_json('noise {"a": 1} trailing'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"b": [1, 2]}\n```'), {"b": [1, 2]})
        self.assertIsNone(extract_json("no json here"))
        self.assertIsNone(extract_json(""))

    def test_tiers(self):
        from saani.simllm import SimLLM
        sim = SimLLM(scripts={"*": ["ok"]})
        r = LLMRouter(sim, {"planner": "big", "worker": "mid"}, 100)
        r.complete(tier="planner", system="s", messages=[Msg("user", "x")])
        r.complete(tier="fast", system="s", messages=[Msg("user", "x")])      # unknown tier falls back to worker
        self.assertEqual([c["model"] for c in sim.calls], ["big", "mid"])


if __name__ == "__main__":
    unittest.main()
