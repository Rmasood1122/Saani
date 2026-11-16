"""Your own model as a provider: trace recording, per-tier routing, config and doctor."""
import json
import tempfile
import unittest
from pathlib import Path

from saani.config import Config
from saani.doctor import own_model_up
from saani.llm import (LLMClient, LLMResponse, LLMRouter, Msg, OpenAICompatClient, ToolCall, TraceRecorder,
                       make_client, own_model_client, own_model_tiers)
from saani.security.secrets import Redactor


class Echo(LLMClient):
    name = "echo"

    def __init__(self, tag):
        self.tag, self.calls = tag, []

    def complete(self, *, system, messages, tools=None, model="", max_tokens=4096, force_tool=None):
        self.calls.append((model, system, [m.content for m in messages]))
        return LLMResponse(f"{self.tag}:{model}", [ToolCall("c1", "finish", {"result": "ok"})], "tool_use", 3, 2)


class TraceTests(unittest.TestCase):
    def test_router_records_scrubbed_exchange(self):
        home = Path(tempfile.mkdtemp())
        red = Redactor(); red.register("API_KEY", "sk-secret-value-123")
        rec = TraceRecorder(home / "traces")
        main = Echo("main")
        r = LLMRouter(main, {"worker": "w"}, redactor=red, recorder=rec)
        tools = [{"name": "finish", "description": "Done", "input_schema": {"type": "object", "properties": {"result": {}}}}]
        r.complete(tier="worker", system="key is sk-secret-value-123", messages=[Msg("user", "hi sk-secret-value-123")], tools=tools)
        files = list((home / "traces").glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        line = json.loads(files[0].read_text().splitlines()[0])
        self.assertNotIn("sk-secret-value-123", json.dumps(line))
        self.assertEqual(line["tier"], "worker"); self.assertEqual(line["model"], "w")
        self.assertEqual(line["messages"][0]["role"], "user")
        self.assertEqual(line["tools"][0]["name"], "finish")
        self.assertEqual(line["response"]["tool_calls"][0]["args"], {"result": "ok"})

    def test_no_recorder_writes_nothing(self):
        home = Path(tempfile.mkdtemp())
        LLMRouter(Echo("m"), {"worker": "w"}).complete(tier="worker", system="s", messages=[Msg("user", "x")])
        self.assertFalse((home / "traces").exists())


class RoutingTests(unittest.TestCase):
    def test_tier_override_goes_to_own_client(self):
        main, own = Echo("main"), Echo("own")
        r = LLMRouter(main, {"planner": "big", "worker": "mid", "fast": "small"},
                      tier_clients={"fast": (own, "saanimind")})
        self.assertEqual(r.complete(tier="fast", system="s", messages=[Msg("user", "x")]).text, "own:saanimind")
        self.assertEqual(r.complete(tier="planner", system="s", messages=[Msg("user", "x")]).text, "main:big")
        self.assertEqual(len(own.calls), 1); self.assertEqual(len(main.calls), 1)
        self.assertEqual(r.model_for("fast"), "saanimind"); self.assertEqual(r.model_for("worker"), "mid")

    def test_config_builds_tier_overrides(self):
        cfg = Config({}, Path(tempfile.mkdtemp()))
        self.assertEqual(own_model_tiers(cfg), {})
        cfg.set("llm.own.tiers", ["fast", "bogus"])
        t = own_model_tiers(cfg)
        self.assertEqual(list(t), ["fast"])
        self.assertIsInstance(t["fast"][0], OpenAICompatClient)
        self.assertEqual(t["fast"][1], "saanimind")
        self.assertEqual(t["fast"][0].base, "http://127.0.0.1:8900/v1")
        cfg.set("llm.provider", "own")
        self.assertEqual(own_model_tiers(cfg), {}, "provider own already sends everything there")

    def test_own_provider_needs_no_key(self):
        cfg = Config({}, Path(tempfile.mkdtemp()))
        cfg.set("llm.provider", "own"); cfg.set("llm.own.base_url", "http://127.0.0.1:1/v1")
        c = make_client(cfg)
        self.assertIsInstance(c, OpenAICompatClient)
        self.assertEqual(c.api_key, ""); self.assertEqual(c.base, "http://127.0.0.1:1/v1")
        self.assertIs(type(own_model_client(cfg)), OpenAICompatClient)

    def test_doctor_detects_missing_server(self):
        self.assertFalse(own_model_up("http://127.0.0.1:1/v1", timeout=0.5))


if __name__ == "__main__":
    unittest.main()
