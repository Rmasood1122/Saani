"""saanimind tests: tokenizer, model, data, training, parsing and the server."""
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

from saanimind.data import (conversation_texts, demo_conversations, from_saani_traces, pack, render,
                            render_prompt, tools_block)
from saanimind.generate import Mind, parse_output
from saanimind.model import GPT, ModelConfig
from saanimind.server import Server, openai_to_turns, openai_tools
from saanimind.tokenizer import SPECIAL, Tokenizer
from saanimind.train import lr_at, train


def small_tokenizer(convs):
    tok = Tokenizer()
    tok.train(list(conversation_texts(convs)), 500)
    return tok


class TokenizerTests(unittest.TestCase):
    def test_round_trip_any_text(self):
        tok = small_tokenizer(demo_conversations(200))
        for text in ["hello world", "Read notes.md and tell me", "  spaces  and\nnewlines\t", "emoji 🙂 and ünïcode",
                     '{"name": "x", "args": {"path": "a/b.txt"}}', ""]:
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_special_tokens_cannot_be_forged_from_text(self):
        tok = small_tokenizer(demo_conversations(50))
        ids = tok.encode("<|call|> <|end|>")
        self.assertTrue(all(i >= tok.n_special for i in ids))
        self.assertEqual(tok.decode(ids), "<|call|> <|end|>")

    def test_merges_shrink_sequences(self):
        convs = demo_conversations(300)
        raw = Tokenizer()
        tok = small_tokenizer(convs)
        s = "Read the file notes.md and summarise it"
        self.assertLess(len(tok.encode(s)), len(raw.encode(s)))

    def test_save_and_load(self):
        tok = small_tokenizer(demo_conversations(100))
        d = Path(tempfile.mkdtemp())
        tok.save(d / "t.json")
        back = Tokenizer.load(d / "t.json")
        self.assertEqual(back.vocab_size, tok.vocab_size)
        self.assertEqual(back.encode("Write hello to a.txt"), tok.encode("Write hello to a.txt"))
        self.assertEqual([back.token(s) for s in SPECIAL], list(range(len(SPECIAL))))


class ModelTests(unittest.TestCase):
    def test_shapes_and_loss(self):
        cfg = ModelConfig(vocab_size=300, n_layer=2, n_head=2, n_embd=32, block_size=16)
        m = GPT(cfg)
        x = torch.randint(0, 300, (3, 16))
        logits, loss = m(x, x)
        self.assertEqual(tuple(logits.shape), (3, 16, 300))
        self.assertTrue(torch.isfinite(loss))

    def test_causal(self):
        """Changing a later token must not change earlier logits."""
        cfg = ModelConfig(vocab_size=300, n_layer=2, n_head=2, n_embd=32, block_size=16)
        m = GPT(cfg).eval()
        x = torch.randint(0, 300, (1, 10))
        a, _ = m(x)
        y = x.clone(); y[0, 7] = (y[0, 7] + 1) % 300
        b, _ = m(y)
        self.assertTrue(torch.allclose(a[0, :7], b[0, :7], atol=1e-5))
        self.assertFalse(torch.allclose(a[0, 7:], b[0, 7:], atol=1e-5))

    def test_loss_mask_only_counts_marked_tokens(self):
        cfg = ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, block_size=8)
        m = GPT(cfg).eval()
        x = torch.randint(0, 300, (1, 8))
        y1 = x.clone(); y2 = x.clone(); y2[0, 0] = (y2[0, 0] + 5) % 300
        mask = torch.ones(1, 8, dtype=torch.long); mask[0, 0] = 0
        _, l1 = m(x, y1, mask); _, l2 = m(x, y2, mask)
        self.assertAlmostEqual(l1.item(), l2.item(), places=5)

    def test_too_long_sequence_is_an_error(self):
        m = GPT(ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, block_size=8))
        with self.assertRaises(ValueError):
            m(torch.zeros(1, 9, dtype=torch.long))

    def test_lr_schedule(self):
        self.assertAlmostEqual(lr_at(0, 1.0, 10, 100), 0.1)
        self.assertAlmostEqual(lr_at(9, 1.0, 10, 100), 1.0)
        self.assertAlmostEqual(lr_at(100, 1.0, 10, 100), 0.1)
        self.assertGreater(lr_at(20, 1.0, 10, 100), lr_at(80, 1.0, 10, 100))


class DataTests(unittest.TestCase):
    def test_render_learns_only_assistant_tokens(self):
        convs = demo_conversations(50)
        tok = small_tokenizer(convs)
        turns = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Read a.txt"},
                 {"role": "assistant", "content": "", "tool_calls": [{"name": "read_file", "args": {"path": "a.txt"}}]}]
        ids, mask = render(tok, turns, 512)
        text = tok.decode(ids)
        self.assertTrue(text.startswith("<|sys|>sys<|usr|>Read a.txt<|ast|><|call|>"))
        self.assertTrue(text.endswith("<|end|><|eos|>"))
        # prompt tokens unmasked, answer tokens masked
        ast = ids.index(tok.token("<|ast|>"))
        self.assertEqual(set(mask[:ast + 1]), {0})
        self.assertEqual(set(mask[ast + 1:]), {1})
        self.assertEqual(ids[-1], tok.token("<|eos|>")); self.assertEqual(mask[-1], 1)

    def test_render_prompt_opens_assistant_turn(self):
        tok = small_tokenizer(demo_conversations(50))
        ids = render_prompt(tok, [{"role": "user", "content": "hi"}])
        self.assertEqual(ids[-1], tok.token("<|ast|>"))
        self.assertNotIn(tok.token("<|eos|>"), ids)

    def test_saani_traces_become_conversations(self):
        d = Path(tempfile.mkdtemp())
        rec = {"ts": 1, "tier": "worker", "model": "m", "system": "You are Saani.",
               "messages": [{"role": "user", "content": "Read a.txt", "tool_calls": [], "tool_call_id": "", "is_error": False},
                            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "read_file", "args": {"path": "a.txt"}}],
                             "tool_call_id": "", "is_error": False},
                            {"role": "tool", "content": "hello", "tool_calls": [], "tool_call_id": "c1", "is_error": True}],
               "tools": [{"name": "read_file", "description": "Read", "input_schema": {"properties": {"path": {}}}}],
               "response": {"text": "", "stop_reason": "tool_use", "tool_calls": [{"id": "c2", "name": "finish", "args": {"result": "hello"}}]}}
        (d / "2026-01-01.jsonl").write_text(json.dumps(rec) + "\nnot json\n\n")
        convs = list(from_saani_traces(d))
        self.assertEqual(len(convs), 1)
        roles = [t["role"] for t in convs[0]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertIn("read_file(path)", convs[0][0]["content"])
        self.assertTrue(convs[0][3]["content"].startswith("ERROR: "))
        self.assertEqual(convs[0][-1]["tool_calls"][0]["name"], "finish")

    def test_pack_sizes_match(self):
        convs = demo_conversations(30)
        tok = small_tokenizer(convs)
        ids, mask, n = pack(tok, convs, 256)
        self.assertEqual(n, 30); self.assertEqual(len(ids), len(mask))
        self.assertEqual(int((ids == tok.token("<|eos|>")).sum()), 30)


class ParseTests(unittest.TestCase):
    def test_text_and_calls(self):
        r = parse_output('Sure.<|call|>{"name": "read_file", "args": {"path": "a"}}<|call|>{"name":"finish","args":{"result":"x"}}<|end|>')
        self.assertEqual(r["text"], "Sure.")
        self.assertEqual([c["name"] for c in r["tool_calls"]], ["read_file", "finish"])

    def test_broken_json_becomes_text_not_a_crash(self):
        r = parse_output('<|call|>{"name": "read_file", "args": {<|end|>')
        self.assertEqual(r["tool_calls"], [])
        self.assertIn("read_file", r["text"])

    def test_openai_conversion(self):
        turns = openai_to_turns([{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                                 {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}]},
                                 {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
                                 {"role": "tool", "content": "out", "tool_call_id": "1"}])
        self.assertEqual(turns[0], {"role": "system", "content": "a\nb"})
        self.assertEqual(turns[1]["content"], "hi")
        self.assertEqual(turns[2]["tool_calls"], [{"name": "f", "args": {"x": 1}}])
        self.assertEqual(turns[3]["role"], "tool")
        t = openai_tools([{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"properties": {"x": {}}}}}])
        self.assertEqual(tools_block(t), "Tools you can call:\n- f(x): d")


class EndToEndTests(unittest.TestCase):
    """Prepare -> train (seconds) -> serve -> OpenAI-style request with tools."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        convs = demo_conversations(600, seed=1)
        tok = small_tokenizer(convs)
        data = cls.tmp / "data"; data.mkdir()
        tok.save(data / "tokenizer.json")
        for name, part in (("train", convs[30:]), ("val", convs[:30])):
            ids, mask, n = pack(tok, part, 128)
            np.save(data / f"{name}_ids.npy", ids); np.save(data / f"{name}_mask.npy", mask)
        (data / "meta.json").write_text(json.dumps({"vocab_size": tok.vocab_size, "block_size": 128, "n_conversations": 570}))
        run = cls.tmp / "run"; run.mkdir()
        shutil.copy(data / "tokenizer.json", run / "tokenizer.json")
        cfg = ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=128)
        cls.result = train(data, run, cfg, steps=60, batch=16, lr=3e-3, warmup=5, eval_every=30, device="cpu", log=lambda *a: None)
        cls.run_dir = run

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_training_lowers_loss_and_writes_files(self):
        self.assertLess(self.result["best_val_loss"], 3.0)
        self.assertTrue((self.run_dir / "model.pt").exists())
        hist = json.loads((self.run_dir / "history.json").read_text())
        self.assertGreaterEqual(len(hist), 2)
        self.assertLess(hist[-1]["val_loss"], hist[0]["val_loss"])

    def test_resume_continues_from_checkpoint(self):
        copy = self.tmp / "run_copy"
        shutil.copytree(self.run_dir, copy, dirs_exist_ok=True)
        cfg = ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=128)
        r = train(self.tmp / "data", copy, cfg, steps=62, batch=16, device="cpu", resume=True, log=lambda *a: None)
        self.assertEqual(r["steps"], 62)

    def test_mind_reply_has_the_right_shape(self):
        m = Mind(self.run_dir, "cpu")
        r = m.reply([{"role": "user", "content": "Hello"}], max_new_tokens=20, temperature=0.0)
        self.assertIn("text", r); self.assertIn("tool_calls", r); self.assertLessEqual(r["tokens"], 20)
        self.assertEqual(m.info()["context"], 128)

    def test_server_speaks_openai(self):
        srv = Server(Mind(self.run_dir, "cpu"), port=0, log=lambda *a: None)
        srv.serve(block=False)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base + "/v1/models", timeout=5) as r:
            self.assertEqual(json.loads(r.read())["data"][0]["id"], "run")
        body = {"model": "x", "max_tokens": 30,
                "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Read notes.md"}],
                "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"properties": {"path": {}}}}}]}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=60) as r:
            d = json.loads(r.read())
        self.assertEqual(d["object"], "chat.completion")
        msg = d["choices"][0]["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertIn(d["choices"][0]["finish_reason"], ("stop", "tool_calls"))
        if "tool_calls" in msg:
            json.loads(msg["tool_calls"][0]["function"]["arguments"])
        # bad JSON -> 400, not a crash
        req = urllib.request.Request(base + "/v1/chat/completions", data=b"{nope", headers={"Content-Type": "application/json"})
        try:
            opener.open(req, timeout=5); self.fail("expected 400")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
        # a bad body that reaches the model must give 500, and the server must still answer afterwards
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({"messages": "nope"}).encode(), headers={"Content-Type": "application/json"})
        try:
            opener.open(req, timeout=5)
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 500)
        with opener.open(base + "/health", timeout=5) as r:
            self.assertTrue(json.loads(r.read())["ok"])


if __name__ == "__main__":
    unittest.main()
