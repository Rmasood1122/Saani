"""The browser dashboard's new pieces: the picture, the settings API, and the desktop window command."""
import io
import json
import sys
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from unittest import mock

from saani.cli import main as cli_main
from saani.config import Config
from saani.daemon import Daemon, load_token
from tests.helpers import SaaniCase, free_port


class UiTests(SaaniCase):
    def start(self):
        cfg = Config({}, self.home)
        cfg.set("llm.provider", "demo")
        cfg.set("daemon.port", free_port())
        cfg.save()
        self.cfg = cfg
        self.d = Daemon(cfg, enable_voice=False)
        self.d.serve(block=False)
        self.addCleanup(self.d.shutdown)
        self.base = f"http://127.0.0.1:{cfg.get('daemon.port')}"
        self.token = load_token(self.home)

    def call(self, path, body=None, token=True, method=None):
        h = {"X-Requested-With": "saani"}
        if token:
            h["X-Saani-Token"] = self.token
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=h, method=method)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=5) as r:
                raw, ctype = r.read(), r.headers.get("Content-Type", "")
                return r.status, raw, ctype
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type", "")

    def test_background_picture_is_public_and_real(self):
        self.start()
        status, raw, ctype = self.call("/assets/blackhole.png", token=False)
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "image/png")
        self.assertTrue(raw.startswith(b"\x89PNG"))

    def test_dashboard_page_uses_the_picture(self):
        self.start()
        status, raw, _ = self.call("/", token=True)
        self.assertEqual(status, 200)
        self.assertIn(b"/assets/blackhole.png", raw)
        self.assertIn(b"btn-settings", raw)

    def test_settings_need_the_token(self):
        self.start()
        status, _, _ = self.call("/api/settings", token=False)
        self.assertEqual(status, 401)

    def test_settings_read_and_change_safe_fields(self):
        self.start()
        status, raw, _ = self.call("/api/settings")
        self.assertEqual(status, 200)
        s = json.loads(raw)
        self.assertEqual(s["mode"], "ask")
        status, raw, _ = self.call("/api/settings", {"name": "Ahmad", "mode": "strict",
                                                     "open_on_start": True, "open_on_attention": False})
        self.assertEqual(status, 200)
        self.assertEqual(self.cfg.get("security.mode"), "strict")
        self.assertEqual(self.cfg.get("name"), "Ahmad")
        self.assertTrue(self.cfg.get("daemon.open_dashboard_on_start"))
        self.assertFalse(self.cfg.get("daemon.open_dashboard_on_attention"))

    def test_settings_refuse_trusted_and_bad_values(self):
        self.start()
        for body in ({"mode": "trusted"}, {"mode": "anything"}, {"name": "   "},
                     {"open_on_start": "yes"}):
            status, raw, _ = self.call("/api/settings", body)
            self.assertEqual(status, 400, body)
            self.assertIn("error", json.loads(raw))
        self.assertEqual(self.cfg.get("security.mode"), "ask")

    def test_window_command_explains_itself_without_pywebview(self):
        self.start()
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"webview": None}), redirect_stdout(out):
            code = cli_main(["--home", str(self.home), "window"])
        self.assertEqual(code, 1)
        self.assertIn("saani[window]", out.getvalue())

    def test_window_command_needs_the_service_running(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(["--home", str(self.home), "window"])
        self.assertEqual(code, 1)
        self.assertIn("saani daemon", out.getvalue())


if __name__ == "__main__":
    unittest.main()
