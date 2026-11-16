"""Secret vault and redaction.

Agents never see secret values. They refer to secrets by name; values are
injected at execution time (after approval) and scrubbed from every tool
output, log line and model-bound message.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"), "[REDACTED API KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "[REDACTED API KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED AWS KEY]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "[REDACTED SLACK TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED GOOGLE KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "[REDACTED JWT]"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/\-]{20,}=*"), r"\1 [REDACTED]"),
    # KEY=value lines in env files / exports, where the UPPER_CASE key name says it is a secret
    (re.compile(r"(?m)^((?:[^\n:=]{1,200}:(?:\d+:)?)?\s*(?:export\s+|set\s+|SET\s+)?[A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIALS?)[A-Z0-9_]*\s*=\s*)(\S.*)$"), r"\1[REDACTED]"),
    # lower-case ini/properties style, only when the value is a bare literal (not code)
    (re.compile(r"(?im)^(\s*[a-z0-9_.\-]*(?:password|passwd|secret|token|api_?key)[a-z0-9_.\-]*\s*=\s*)([^\s()\[\]{}'\"]{6,})\s*$"), r"\1[REDACTED]"),
    # hard-coded string literals assigned to secret-looking names
    (re.compile(r"(?i)((?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)[\"']?\s*[:=]\s*)([\"'])[^\"'\n]{8,}\2"), r"\1\2[REDACTED]\2"),
    # a whole line of base64 (the body of a key file read line by line, e.g. through a search)
    (re.compile(r"(?m)(?:(?<=[\s:])|^)(?=[A-Za-z0-9+/]*[a-z])(?=[A-Za-z0-9+/]*[A-Z])(?=[A-Za-z0-9+/]*[0-9+/])[A-Za-z0-9+/]{60,}={0,2}[ \t]*$"), "[REDACTED ENCODED DATA]"),
    # credentials embedded in URLs
    (re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:)[^@\s/]{3,}(@)"), r"\1[REDACTED]\2"),
]


class Redactor:
    def __init__(self):
        self._known: dict[str, str] = {}
        self._lock = threading.Lock()

    def register(self, name: str, value: str | None) -> None:
        if value and len(value) >= 4:
            with self._lock:
                self._known[value] = name

    def redact(self, text: str) -> str:
        if not text:
            return text
        with self._lock:
            known = sorted(self._known.items(), key=lambda kv: -len(kv[0]))
        for value, name in known:
            if value in text:
                text = text.replace(value, f"[SECRET:{name}]")
        for rx, repl in _PATTERNS:
            text = rx.sub(repl, text)
        return text


class Vault:
    """Named secrets. Uses the OS keychain through ``keyring`` when installed,
    otherwise a file readable only by the current user (obfuscated, NOT encrypted)."""

    SERVICE = "saani"

    def __init__(self, home: Path, redactor: Redactor | None = None, use_keyring: bool | None = None):
        self.home = Path(home)
        self.redactor = redactor
        self._lock = threading.RLock()
        self._index = self.home / "vault_index.json"
        self._file = self.home / "vault.json"
        self._kr = None
        if use_keyring is not False:
            try:
                import keyring  # type: ignore
                from keyring.backends import fail  # type: ignore
                if not isinstance(keyring.get_keyring(), fail.Keyring):
                    self._kr = keyring
            except Exception:
                self._kr = None
        for n in self.names():
            self.get(n)  # registers values with the redactor

    @property
    def backend(self) -> str:
        return "os-keychain" if self._kr else "file"

    def names(self) -> list[str]:
        try:
            return sorted(json.loads(self._index.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return []

    def _write_private(self, path: Path, text: str) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)

    def _file_data(self) -> dict:
        try:
            return json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def set(self, name: str, value: str) -> None:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$", name):
            raise ValueError("secret names must look like ENV_VAR_NAMES")
        with self._lock:
            if self._kr:
                self._kr.set_password(self.SERVICE, name, value)
            else:
                data = self._file_data()
                data[name] = base64.b64encode(value.encode()).decode()
                self._write_private(self._file, json.dumps(data))
            self._write_private(self._index, json.dumps(sorted(set(self.names()) | {name})))
        if self.redactor:
            self.redactor.register(name, value)

    def get(self, name: str) -> str | None:
        with self._lock:
            if self._kr:
                try:
                    value = self._kr.get_password(self.SERVICE, name)
                except Exception:
                    value = None
            else:
                raw = self._file_data().get(name)
                value = base64.b64decode(raw).decode() if raw else None
        if value and self.redactor:
            self.redactor.register(name, value)
        return value

    def delete(self, name: str) -> None:
        with self._lock:
            if self._kr:
                try:
                    self._kr.delete_password(self.SERVICE, name)
                except Exception:
                    pass
            else:
                data = self._file_data()
                data.pop(name, None)
                self._write_private(self._file, json.dumps(data))
            self._write_private(self._index, json.dumps([n for n in self.names() if n != name]))

    def resolve(self, name: str) -> str | None:
        """Vault first, then the process environment (also registered for redaction)."""
        value = self.get(name)
        if value is None and name in os.environ:
            value = os.environ[name]
            if self.redactor:
                self.redactor.register(name, value)
        return value
