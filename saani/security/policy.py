"""Risk policy: decides how dangerous a tool call is *before* it runs.

Four levels:

* SAFE     - read-only / harmless; runs without asking.
* CAUTION  - changes something ordinary; asks unless the user saved a rule.
* DANGER   - destructive, privileged, credential-touching or outward-sending;
             always asks, and can never be "always allowed".
* BLOCKED  - catastrophic or self-tampering; never runs.

Design rule: only a strict allowlist is ever SAFE. Anything the classifier does
not positively recognise falls to CAUTION (ask), so obfuscation can downgrade a
command to "ask the user" but never to "run silently" in the default mode.
This is defence in depth, not a sandbox - see docs/SECURITY.md.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from urllib.parse import urlparse


class Risk(IntEnum):
    SAFE = 0
    CAUTION = 1
    DANGER = 2
    BLOCKED = 3


@dataclass
class Verdict:
    risk: Risk
    reason: str
    rule_key: str = ""       # identity used for "allow for this task" / "always allow"
    always_ok: bool = False  # may the user save a permanent rule for this?


def _worst(*verdicts: Verdict) -> Verdict:
    return max(verdicts, key=lambda v: v.risk)


def exact_key(tool: str, args: dict) -> str:
    blob = json.dumps({"t": tool, "a": args}, sort_keys=True, default=str)
    return "exact:" + hashlib.sha256(blob.encode()).hexdigest()[:24]


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SECRET_DIRS = {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".password-store", "keychains"}
SECRET_NAMES = {
    ".env", ".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials", "credentials",
    "credentials.json", "credentials.db", "secrets.json", "secrets.yaml", "secrets.yml",
    "secrets.toml", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "login data", "cookies",
    "cookies.sqlite", "key4.db", "logins.json", "shadow", "sudoers", "master.key",
    "token.json", "wallet.dat", ".htpasswd", ".pgpass", ".my.cnf", ".dockercfg", "kubeconfig",
    "terraform.tfstate", "terraform.tfvars", ".bash_history", ".zsh_history", ".python_history",
    ".psql_history", ".mysql_history", "consolehost_history.txt",
}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx", ".keystore", ".jks", ".ovpn", ".ppk"}
ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template", ".env.dist"}
SECRET_SUBSTR = (
    "/google/chrome/", "/google-chrome/", "/chromium/", "/mozilla/firefox/", "/microsoft/edge/",
    "/bravesoftware/", "/.config/gcloud/", "/.docker/config.json", "/library/keychains/",
    "/1password/", "/bitwarden/",
)
SYSTEM_PREFIXES = (
    "/etc/", "/usr/", "/bin/", "/sbin/", "/boot/", "/lib/", "/lib64/", "/opt/", "/system/",
    "/library/", "/private/etc/", "c:/windows/", "c:/program files/",
    "c:/program files (x86)/", "c:/programdata/",
)
PERSIST_NAMES = {
    ".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile", ".zshenv", ".bash_login",
    ".bash_logout", "config.fish", "profile.ps1", "microsoft.powershell_profile.ps1",
    "crontab", "authorized_keys", ".gitconfig",
}
PERSIST_SUBSTR = (
    "/.config/autostart/", "/launchagents/", "/launchdaemons/", "/start menu/programs/startup/",
    "/systemd/user/", "/systemd/system/", "/.git/hooks/", "/.git/config", "/cron.d/", "/etc/cron",
)
EXEC_SUFFIXES = {".exe", ".bat", ".cmd", ".com", ".msi", ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh",
                 ".scr", ".pif", ".lnk", ".reg", ".hta", ".cpl", ".jar", ".sh", ".bash", ".zsh", ".command",
                 ".app", ".pkg", ".dmg", ".scpt", ".workflow", ".desktop", ".appimage", ".run", ".deb", ".rpm",
                 ".py", ".pyw", ".rb", ".pl", ".php", ".url", ".inf", ".gadget", ".msc", ".action"}
SELF_CRITICAL = {"config.json", "audit.log", "daemon.token", "vault.json", "vault_index.json",
                 "saani.db", "saani.db-wal", "saani.db-shm"}
SELF_UNREADABLE = {"daemon.token", "vault.json"}


def norm_path(p: str | Path) -> str:
    """Absolute, symlink-resolved, forward-slashed, lower-cased path for matching."""
    s = os.path.expanduser(str(p))
    try:
        s = os.path.realpath(s)
    except (OSError, ValueError):
        s = os.path.abspath(s)
    return s.replace("\\", "/").lower()


def _inside(path: str, root: str) -> bool:
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


def is_secret_path(norm: str) -> bool:
    parts = norm.split("/")
    name = parts[-1]
    if any(p in SECRET_DIRS for p in parts):
        return True
    if name in ENV_TEMPLATES:
        return False
    if name in SECRET_NAMES or name.startswith(".env."):
        return True
    if any(name.endswith(s) for s in SECRET_SUFFIXES):
        return True
    return any(s in norm + ("/" if not norm.endswith("/") else "") for s in SECRET_SUBSTR)


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------


def split_unquoted(cmd: str, mode: str = "posix") -> list[str]:
    """Split a command line on ; | || & && and newlines that are outside quotes.

    mode 'posix'  : single and double quotes, backslash escapes the next character
    mode 'literal': same quotes, backslash is an ordinary character (Windows paths)
    mode 'cmd'    : cmd.exe rules - only double quotes quote, ^ escapes
    Callers classify every reading and keep the worst, so a command cannot hide a
    second command behind whichever quoting rule the classifier did not expect.
    """
    quotes = '"' if mode == "cmd" else "'\""
    esc = "^" if mode == "cmd" else ("\\" if mode == "posix" else "")
    parts, cur, q, i, n = [], [], "", 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if q:
            cur.append(ch)
            if ch == "\\" and q == '"' and mode == "posix" and i + 1 < n:
                cur.append(cmd[i + 1]); i += 1
            elif ch == q:
                q = ""
        elif ch in quotes:
            q = ch; cur.append(ch)
        elif esc and ch == esc and i + 1 < n:
            cur.append(ch); cur.append(cmd[i + 1]); i += 1
        elif ch in ";|&\n":
            parts.append("".join(cur)); cur = []
            if ch in "|&" and i + 1 < n and cmd[i + 1] == ch:
                i += 1
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return [x.strip() for x in parts if x.strip()]

_BENIGN_REDIR = re.compile(r"\d?>&\d|&?\d?>>?\s*(/dev/null|nul)\b", re.I)

SAFE_CMDS = {
    "ls", "dir", "pwd", "cd", "echo", "cat", "type", "head", "tail", "wc", "grep", "egrep",
    "fgrep", "rg", "findstr", "find", "which", "where", "whereis", "whoami", "hostname", "date",
    "uptime", "uname", "ver", "df", "du", "free", "ps", "tasklist", "tree", "stat", "file",
    "basename", "dirname", "realpath", "sort", "uniq", "cut", "tr", "diff", "cmp", "md5sum",
    "sha1sum", "sha256sum", "nproc", "lscpu", "sw_vers", "systeminfo", "id", "groups", "more",
    "true", "test", "sleep", "column", "nl", "fold", "rev", "jq", "printf",
    # read-only PowerShell cmdlets
    "get-childitem", "get-content", "get-location", "get-process", "get-date", "select-string",
    "test-path", "get-item", "measure-object", "select-object", "where-object", "sort-object",
    "format-table", "format-list", "out-string", "get-command", "get-host",
}
SAFE_SUB = {
    "git": {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "describe",
            "shortlog", "branch", "tag", "remote", "--version"},
    "pip": {"list", "show", "freeze", "--version", "check"},
    "pip3": {"list", "show", "freeze", "--version", "check"},
    "npm": {"ls", "list", "view", "--version", "outdated"},
    "docker": {"ps", "images", "version", "info"},
    "python": {"--version", "-v"}, "python3": {"--version", "-v"},
    "node": {"--version", "-v"}, "go": {"version"}, "cargo": {"--version"},
    "java": {"-version", "--version"}, "dotnet": {"--version", "--info"},
}
_GIT_MUTATING = {
    "branch": {"-d", "-m", "-c", "-f", "-u", "--delete", "--move", "--copy", "--force", "--set-upstream-to",
               "--unset-upstream", "--edit-description"},
    "tag": {"-a", "-s", "-d", "-f", "-m", "-u", "--delete", "--force", "--annotate", "--sign"},
    "remote": {"add", "remove", "rm", "rename", "set-url", "prune", "set-head", "set-branches", "update"},
}

DELETE_CMDS = {"rm", "rmdir", "del", "erase", "rd", "remove-item", "ri", "unlink", "shred", "srm"}
DANGER_CMDS = {
    **{c: "deletes files" for c in DELETE_CMDS},
    **{c: "runs with elevated privileges" for c in ("sudo", "su", "doas", "runas", "pkexec", "gsudo")},
    **{c: "changes permissions or ownership" for c in ("chmod", "chown", "chgrp", "icacls", "takeown", "attrib", "cacls")},
    **{c: "kills processes" for c in ("kill", "pkill", "killall", "taskkill", "stop-process")},
    **{c: "shuts down, restarts or logs off the machine" for c in ("shutdown", "reboot", "halt", "poweroff", "logoff", "restart-computer", "stop-computer")},
    **{c: "changes system services, scheduled jobs or startup items" for c in ("systemctl", "service", "launchctl", "sc", "schtasks", "crontab", "at", "new-service", "set-service", "register-scheduledtask")},
    **{c: "edits the registry or system settings" for c in ("reg", "regedit", "setx", "set-itemproperty", "new-itemproperty", "set-executionpolicy", "defaults", "netsh", "iptables", "ufw", "bcdedit")},
    **{c: "manages user accounts or passwords" for c in ("net", "useradd", "userdel", "usermod", "passwd", "chpasswd", "adduser", "deluser", "dscl")},
    **{c: "opens a remote connection" for c in ("ssh", "scp", "sftp", "ftp", "nc", "ncat", "netcat", "telnet", "rsync", "socat")},
    **{c: "low-level disk operation" for c in ("dd", "mount", "umount", "fdisk", "parted", "format-volume", "clear-disk", "diskutil")},
    **{c: "runs another shell or evaluates a string as code" for c in ("eval", "exec", "source", ".", "xargs", "bash", "sh", "zsh", "fish", "dash", "ksh", "csh", "cmd", "powershell", "pwsh", "invoke-expression", "iex", "osascript", "wscript", "cscript", "mshta", "rundll32", "regsvr32")},
    **{c: "reads or changes stored credentials" for c in ("security", "secret-tool", "keyring", "cmdkey", "gpg", "ssh-add", "ssh-keygen", "pass", "op", "bw", "vaultcmd")},
    **{c: "sends e-mail" for c in ("mail", "sendmail", "mutt", "mailx", "send-mailmessage")},
    "truncate": "truncates files", "mkfifo": "creates a pipe", "ln": "creates links",
    "wmic": "runs system management commands",
    "msiexec": "installs software system-wide", "start-process": "starts a process",
}
INTERPRETERS = {"python", "python3", "py", "node", "perl", "ruby", "php", "deno", "bun", "lua",
                "rscript", "pythonw"}
PKG_SYS = {"apt", "apt-get", "yum", "dnf", "pacman", "zypper", "brew", "choco", "winget", "snap",
           "scoop", "dpkg", "rpm", "port", "apk", "flatpak"}
PKG_READ = {"list", "search", "show", "info", "--version", "-v", "outdated", "leaves", "deps", "policy"}
CLOUD_CMDS = {"aws", "gcloud", "az", "kubectl", "terraform", "helm", "pulumi", "doctl", "oc", "eksctl"}
CLOUD_READ = {"get", "describe", "list", "ls", "show", "plan", "version", "status", "logs", "diff",
              "validate", "fmt", "whoami", "top", "explain"}
CLOUD_WRITE = {"delete", "destroy", "apply", "create", "rm", "remove", "put", "update", "patch",
               "terminate", "stop", "start", "scale", "rollout", "exec", "cp", "sync", "mb", "rb",
               "deploy", "set", "edit", "replace", "drain", "cordon", "taint", "install", "upgrade",
               "uninstall", "rollback", "run", "import", "mv"}
DEPLOY_CMDS = {"vercel", "netlify", "fly", "flyctl", "heroku", "firebase", "render", "railway",
               "wrangler", "serverless", "sls", "cdk", "supabase"}
WRAPPERS = {"nohup", "time", "nice", "ionice", "timeout", "command", "builtin", "env", "stdbuf",
            "start", "call", "watch", "caffeinate", "setsid", "busybox", "toybox"}
SHELL_WORDS = {"{", "}", "(", ")", "!", "then", "do", "else", "elif", "if", "while", "until", "fi", "done", "esac"}
RECURSIVE_READERS = {"grep", "egrep", "fgrep", "rg", "findstr", "select-string"}
BULK_COPIERS = {"tar", "zip", "7z", "7za", "rar", "cp", "copy", "xcopy", "robocopy", "rsync", "copy-item",
                "compress-archive", "ditto", "cpio", "gzip", "bzip2", "xz"}
GIT_CONFIG_READS = {"--get", "--get-all", "--get-regexp", "--list", "-l", "--show-origin", "--show-scope",
                    "--name-only", "--get-urlmatch", "--get-color", "--get-colorbool"}
GIT_CONFIG_EDITS = {"--unset", "--unset-all", "--remove-section", "--rename-section", "--edit", "-e", "--add",
                    "--replace-all"}
GIT_CONFIG_BENIGN = {"user.name", "user.email", "init.defaultbranch", "pull.rebase", "pull.ff", "push.default",
                     "push.autosetupremote", "core.autocrlf", "core.ignorecase", "core.filemode", "core.eol",
                     "color.ui", "merge.ff", "rebase.autostash", "fetch.prune", "core.longpaths"}
MULTI = {"git", "npm", "pnpm", "yarn", "pip", "pip3", "docker", "cargo", "go", "gh", "dotnet",
         "poetry", "uv", "conda", "make", "npx", "bun", "deno", "composer", "mvn", "gradle",
         "bundle", "rake", "flutter", "dart", "swift", "kubectl", "vercel"}

_BLOCKED_RX = [
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\bmkfs(\.\w+)?\b|\bwipefs\b", "formats a filesystem"),
    (r"\bdd\b.*\bof=/dev/(sd|hd|nvme|disk|mmcblk|rdisk)", "overwrites a raw disk"),
    (r">\s*/dev/(sd|hd|nvme|disk|rdisk)", "overwrites a raw disk"),
    (r"\bformat\s+[a-z]:", "formats a drive"),
    (r"\bdiskpart\b|\bcipher\s+/w|\bvssadmin\s+delete|\bformat-volume\b|\bclear-disk\b", "destroys disk data"),
    (r"\bdiskutil\s+(erase|reformat|partitiondisk|zerodisk|secureerase)", "erases a disk"),
    (r"\bchmod\s+(-\w+\s+)*-r\w*\s+(-\w+\s+)*[0-7]{3,4}\s+/\s*$", "recursive permission change on /"),
    (r"\b(shred|srm)\b.*\s/dev/(sd|hd|nvme|disk|mmcblk|rdisk)", "destroys a raw disk"),
    (r"\bfind\s+(/|~/?|\$home/?|\$\{home\}/?|/home/?|/users/?)\s+(.*\s)?-(delete|exec\s+rm)\b",
     "deletes everything under a root or home directory"),
    (r"\b(mv|move|move-item)\s+(-\S+\s+)*(/|~|\$home|\$\{home\}|/home|/users|[a-z]:)/?\s+\S",
     "moves a root or home directory away"),
]
_DANGER_RX = [
    (r"\|\s*(sudo\s+)?(ba|z|fi|da|k|c|tc)?sh\b", "pipes data into a shell"),
    (r"\|\s*(python[0-9.]*|node|perl|ruby|php|powershell|pwsh|iex|invoke-expression)\b", "pipes data into an interpreter"),
    (r"\b(curl|wget|iwr|irm|invoke-webrequest|invoke-restmethod|http|https)\b.*(\s-x\s*(post|put|delete|patch)|--request\b|\s-d\b|--data\b|--data-\w+|\s-f\b|--form\b|\s-t\b|--upload-file|--post-data|--post-file|--body-data|-method\s+(post|put|delete|patch)|-infile\b|-body\b|\s@\S+)", "sends data to a remote server"),
    (r"\bbase64\b.*(\s-d\b|--decode)", "decodes hidden content"),
    (r"\s-e(nc(odedcommand)?)?\s+[a-z0-9+/=]{24,}", "encoded command"),
    (r"\b(drop|truncate)\s+(table|database|schema)\b|\bdelete\s+from\b", "destructive SQL"),
    (r"--privileged\b", "privileged container"),
    (r"/dev/tcp/", "raw network socket"),
    (r"--no-verify\b", "skips safety hooks"),
    (r"\bgit\b.*\breset\b.*--hard", "discards local changes"),
    (r"\bgit\b.*\b(push|clean|filter-branch|filter-repo|restore)\b", "publishes or discards git changes"),
    (r"\bgit\s+checkout\s+(--\s+)?\.(\s|$)", "discards local changes"),
    (r"\bgit\s+stash\s+(drop|clear)", "discards stashed work"),
    (r"\bgit\b.*--force", "forced git operation"),
    (r"\bgit\s+branch\b.*\s(-d|--delete)\b", "deletes a git branch"),
    (r"\b(mv|move|move-item)\b.*\s(/dev/null|nul)\s*$", "moves files into the void"),
    (r"\bgh\s+repo\s+(delete|create|rename|archive|edit|sync)|\bgh\s+pr\s+(merge|close)|\bgh\s+(auth|secret|ssh-key|gpg-key)\b|\bgh\s+release\s+(create|delete|upload|edit)|\bgh\s+api\b.*(\s-x|--method)", "changes things on GitHub"),
    (r"\bdocker\s+(compose\s+)?(down|rm|rmi|kill|stop|push|login|system\s+prune|volume\s+(rm|prune)|image\s+prune|container\s+prune)\b", "removes, stops or publishes containers"),
    (r"\b(npm|pnpm|yarn|cargo|gem|poetry|uv|flutter|dart)\s+(publish|unpublish|login|adduser|token|push|owner)\b|\btwine\s+upload\b", "publishes a package or changes registry credentials"),
    (r"\bfind\b.*\s-(delete|exec|execdir|ok|okdir|fprint\w*|fls)\b", "find with an action that changes files or runs commands"),
    (r"<<", "feeds an inline script to a program"),
    (r"\bgit\s+(checkout|switch)\b.*\s(-f|--force|--discard-changes)\b", "discards local changes"),
    (r"dropdatabase|\.drop\s*\(|deletemany\s*\(\s*\{\s*\}|\bflush(all|db)\b|\.remove\s*\(\s*\{\s*\}", "destroys database data"),
]
_SECRET_RX = re.compile(
    r"(\.ssh\b|\.gnupg|\.aws\b|\.azure\b|\.kube\b|id_(rsa|dsa|ecdsa|ed25519)|"
    r"(?<![a-z0-9_])\.env\b(?!\.(example|sample|template|dist))|\.netrc|\.npmrc|\.pypirc|"
    r"\.git-credentials|/etc/(shadow|sudoers)|login data|cookies\.sqlite|key4\.db|logins\.json|"
    r"keychain|\.kdbx|\.pem\b|\.p12\b|\.pfx\b|\.ppk\b|password-store|credentials\b|"
    r"secrets?\.(json|ya?ml|toml))"
)
_PY_DANGER = [
    (r"\bsubprocess\b|\bos\.(system|popen|exec\w*|spawn\w*|startfile)\b|\bpty\b", "starts other programs"),
    (r"\bos\.(remove|unlink|rmdir|removedirs|kill|chmod|chown|truncate)\b|\bshutil\.(rmtree|move)\b|\.unlink\s*\(|\.rmdir\s*\(", "deletes or moves files, or kills processes"),
    (r"\bsocket\b|\bsmtplib\b|\bparamiko\b|\bftplib\b|\btelnetlib\b", "opens network connections"),
    (r"\b(requests|httpx|session)\.(post|put|delete|patch)\b|urlopen\([^)]*data\s*=|method\s*=\s*['\"](post|put|delete|patch)", "sends data to a remote server"),
    (r"\bctypes\b|\bwinreg\b|\bkeyring\b|\bpyautogui\b|\bpynput\b", "low-level system access"),
    (r"\beval\s*\(|\bexec\s*\(|__import__|\bimportlib\b|\bcompile\s*\(|\bgetattr\s*\(|\bglobals\s*\(|base64|\bcodecs\b|\bmarshal\b|\bpickle\b", "dynamic or obfuscated code"),
]


_PY_WRITES = re.compile(
    r"\bopen\s*\(.{0,300}?,\s*(mode\s*=\s*)?['\"][rbt]*[wax+]|\.write_(text|bytes)\s*\(|\.touch\s*\(|\.mkdir\s*\(|"
    r"\bos\.(makedirs|mkdir|rename|replace|link|symlink)\b|\bshutil\.(copy\w*|make_archive|unpack_archive)\b|"
    r"\.(rename|replace|symlink_to|hardlink_to)\s*\(\s*(path|pathlib|['\"/~])|\bzipfile\b|\btarfile\b|\bsqlite3\b|\.to_(csv|json|excel|parquet)\s*\(")
_PY_DYNAMIC_PATH = re.compile(
    r"expanduser|expandvars|\.home\s*\(|\benviron\b|\bgetenv\b|\bgettempdir\b|\bos\.sep\b|\bos\.path\.join\b|"
    r"\bsys\.argv\b|\binput\s*\(|\bchr\s*\(|\.join\s*\(|\bformat\s*\(|%\s*\(|\bf['\"]|['\"]\s*\+|\+\s*['\"]")
_PY_PATH_LITERAL = re.compile(r"""['"]((?:~[\\/]|/|[A-Za-z]:[\\/])[^'"\n]{1,300})['"]""")


_HIDDEN_HOME_GLOB = re.compile(r"^(~|\$home|\$\{home\}|%userprofile%|\$env:userprofile)[/\\]+\.[^/\\]*[*?\[]", re.I)


def _strip_shell_syntax(toks: list[str]) -> list[str]:
    """Drop grouping and control words so `( rm x )`, `{ rm x; }`, `then rm x`, `! rm x` are seen as `rm x`."""
    toks = list(toks)
    while toks:
        t = toks[0]
        if t in SHELL_WORDS:
            toks.pop(0)
            continue
        cut = t.lstrip("({!")
        if cut == t:
            break
        if cut:
            toks[0] = cut
        else:
            toks.pop(0)
    while toks and toks[-1] in SHELL_WORDS:
        toks.pop()
    if toks and len(toks) > 1:
        toks[-1] = toks[-1].rstrip(")}") or toks[-1]
    return toks


def _base(tok: str) -> str:
    b = tok.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return re.sub(r"\.(exe|cmd|bat|com|ps1)$", "", b)


class Policy:
    def __init__(self, home: Path, workspace: Path, trusted_dirs: list[str] | None = None,
                 daemon_port: int = 8765):
        self.home = norm_path(home)
        self.workspace = norm_path(workspace)
        self.trusted = [norm_path(d) for d in (trusted_dirs or [])]
        self.daemon_port = int(daemon_port)
        self.user_home = norm_path("~")
        self.windows = os.name == "nt"

    # ------------------------------------------------------------------ paths
    def classify_path(self, path: str | Path, op: str, workdir: Path | None = None,
                      skip_system: bool = False) -> Verdict:
        """op is one of read | write | delete."""
        p = Path(os.path.expanduser(str(path)))
        if not p.is_absolute() and workdir is not None:
            p = Path(workdir) / p
        n = norm_path(p)
        name = n.rsplit("/", 1)[-1]
        roots = [self.workspace] + self.trusted + ([norm_path(workdir)] if workdir else [])

        if _inside(n, self.home) and not _inside(n, self.workspace):
            if name in SELF_UNREADABLE:
                return Verdict(Risk.BLOCKED, "Saani's own credentials are off-limits to agents")
            if op != "read":
                if name in SELF_CRITICAL or n == self.home:
                    return Verdict(Risk.BLOCKED, "agents may not modify Saani's config, database or audit log")
                return Verdict(Risk.DANGER, "modifies Saani's own files (skills, agents, tools or persona)")
        if is_secret_path(n):
            return Verdict(Risk.DANGER, f"{'reads' if op == 'read' else 'modifies'} a credentials or secrets file")
        if op == "read":
            return Verdict(Risk.SAFE, "read-only")
        if not skip_system and any(n.startswith(pre) or n + "/" == pre for pre in SYSTEM_PREFIXES):
            return Verdict(Risk.DANGER, "modifies a system location")
        if name in PERSIST_NAMES or any(s in n + "/" for s in PERSIST_SUBSTR):
            return Verdict(Risk.DANGER, "modifies a startup, login or hook file (runs code automatically later)")
        if op == "delete":
            if n in (self.user_home, "/") or n in roots:
                return Verdict(Risk.BLOCKED, "refuses to delete a root, home or workspace directory")
            if _inside(n, self.workspace):
                return Verdict(Risk.SAFE, "deletes inside Saani's own workspace (recoverable from trash)")
            if any(_inside(n, r) for r in roots):
                return Verdict(Risk.CAUTION, "deletes a file in a trusted project folder (recoverable from trash)",
                               rule_key="delete:" + n.rsplit("/", 1)[0])
            return Verdict(Risk.DANGER, "deletes a file outside trusted folders")
        if any(_inside(n, r) for r in roots):
            return Verdict(Risk.SAFE, "writes inside the workspace or a folder you trusted")
        parent = n.rsplit("/", 1)[0] or "/"
        return Verdict(Risk.CAUTION, "writes a file outside the workspace", rule_key="write:" + parent, always_ok=True)

    def _roots(self, workdir: Path | None) -> list[str]:
        return [self.workspace] + self.trusted + ([norm_path(workdir)] if workdir else [])

    def _existing_outside(self, path: str, workdir: Path | None) -> bool:
        """True if path is an existing file that lies outside the workspace and trusted folders."""
        if not path or any(c in path for c in "$*?`"):
            return False
        p = Path(os.path.expanduser(path))
        if not p.is_absolute() and workdir is not None:
            p = Path(workdir) / p
        try:
            if not p.is_file():
                return False
        except (OSError, ValueError):
            return False
        n = norm_path(p)
        return not any(_inside(n, r) for r in self._roots(workdir))

    # ------------------------------------------------------------------ shell
    def _catastrophic_delete(self, cmd: str, low: list[str]) -> bool:
        if cmd not in DELETE_CMDS:
            return False
        recursive = any(re.match(r"^-[a-z]*r", t) or t in ("--recursive", "/s", "-recurse") for t in low[1:])
        if not recursive:
            return False
        roots = {"/", "/*", "~", "~/", "~/*", "$home", "${home}", "$home/", "$home/*", "/home",
                 "/home/", "/home/*", "/users", "/users/", "/users/*", "%userprofile%",
                 "%homepath%", "$env:userprofile", "/etc", "/usr", "/var", "/bin", "/boot",
                 "/lib", "/system", "/library", "c:windows", "c:users", "c:/windows", "c:/users",
                 self.user_home, self.user_home + "/", self.user_home + "/*", self.home, self.home + "/"}
        for t in low[1:]:
            if t in roots or re.match(r"^[a-z]:[\\/]?\*?$", t):
                return True
        return False

    def _segment(self, seg: str, cwd: Path | None = None) -> Verdict:
        # Parse twice: POSIX-style (so r\\m collapses to rm) and with backslashes as path
        # separators (so Windows paths like c:\\ survive). The stricter reading wins.
        verdicts = []
        for variant in dict.fromkeys((seg, seg.replace("\\", "/"))):
            try:
                toks = shlex.split(variant, posix=True)
            except ValueError:
                continue
            toks = _strip_shell_syntax(toks)
            v = self._segment_tokens(list(toks))
            if v.risk == Risk.CAUTION:   # a changing command aimed at a protected place is dangerous
                v = _worst(v, self._sensitive_target(toks, cwd), self._bulk_copy(toks, cwd))
            elif v.risk == Risk.SAFE:    # a read-only command is only harmless if what it reads is
                v = _worst(v, self._read_targets(toks, cwd))
            verdicts.append(v)
        if not verdicts:
            return Verdict(Risk.DANGER, "shell syntax could not be parsed (unbalanced quotes?)")
        v = _worst(*verdicts)
        if v.risk < Risk.DANGER and "{" in seg:
            first = (seg.split() or [""])[0].lower()
            if re.match(r"^[a-z]+-[a-z]+$", first) or first in ("%", "?", "foreach", "where", "select", "group"):
                # a PowerShell script block is code: classify what is inside it
                inner = re.search(r"\{(.*)\}", seg, re.S)
                v = _worst(v, Verdict(Risk.CAUTION, "runs a PowerShell script block"),
                           self.classify_shell(inner.group(1), cwd) if inner else v)
        return v

    def _expand(self, tok: str, cwd: Path | None) -> list[str]:
        """The paths a command-line word may stand for: itself with ~ and variables expanded,
        plus whatever its wildcards match on disk right now."""
        if not tok or "://" in tok or len(tok) > 400:
            return []
        ex = os.path.expandvars(os.path.expanduser(tok.replace("%userprofile%", "~").replace("%USERPROFILE%", "~")))
        p = Path(ex)
        try:
            if not p.is_absolute() and cwd is not None:
                p = Path(cwd) / p
            out = [str(p)]
            if any(c in ex for c in "*?["):
                out += glob.glob(str(p))[:300]
            return out
        except (OSError, ValueError):
            return []

    def _holds_credentials(self, path: str) -> bool:
        """Is this folder the home folder (or above it), or does it directly contain a credential store?"""
        n = norm_path(path)
        if _inside(self.user_home, n) or _inside(self.home, n):
            return True
        try:
            return any(os.path.isdir(os.path.join(path, d)) for d in SECRET_DIRS)
        except OSError:
            return False

    def _read_targets(self, toks: list[str], cwd: Path | None) -> Verdict:
        if not toks:
            return Verdict(Risk.SAFE, "")
        cmd = _base(toks[0])
        low = [t.lower() for t in toks]
        recursive = cmd == "rg" or (cmd in ("grep", "egrep", "fgrep") and any(
            re.match(r"^-[a-z]*r", t) or t in ("--recursive", "--dereference-recursive") for t in low[1:])) \
            or (cmd == "findstr" and "/s" in low[1:]) \
            or (cmd == "select-string" and any(t.startswith("-rec") for t in low[1:]))
        words = [t for t in toks[1:] if not t.startswith("-") and not (re.match(r"^/[a-zA-Z?]{1,2}$", t) and cmd in ("findstr", "dir", "type", "where", "tasklist", "more"))]
        targets = words[1:] if cmd in RECURSIVE_READERS else words
        if recursive and not targets and cwd is not None:
            targets = [str(cwd)]
        worst = Verdict(Risk.SAFE, "")
        for t in targets:
            if _HIDDEN_HOME_GLOB.match(t):   # e.g. ~/.s*h/id_* : judged even if nothing matches right now
                worst = _worst(worst, Verdict(Risk.DANGER, f"wildcard over hidden folders in your home folder, where credentials live ({t})"))
            for c in self._expand(t, cwd):
                v = self.classify_path(c, "read", None)
                if v.risk >= Risk.DANGER:
                    worst = _worst(worst, Verdict(v.risk, f"reads a credentials or secrets location ({t})"))
                if recursive and os.path.isdir(c) and self._holds_credentials(c):
                    worst = _worst(worst, Verdict(Risk.DANGER, f"searches through a folder that holds credential stores ({t})"))
        return worst

    def _bulk_copy(self, toks: list[str], cwd: Path | None) -> Verdict:
        if not toks or _base(toks[0]) not in BULK_COPIERS:
            return Verdict(Risk.SAFE, "")
        for t in toks[1:]:
            if t.startswith("-"):
                continue
            for c in self._expand(t, cwd):
                if os.path.isdir(c) and self._holds_credentials(c):
                    return Verdict(Risk.DANGER, f"copies or archives a folder that holds credential stores ({t})")
        return Verdict(Risk.SAFE, "")

    def _git(self, toks: list[str]) -> tuple[Verdict | None, list[str]]:
        """git can be made to run arbitrary programs through its configuration; treat that as execution."""
        i = 1
        while i < len(toks) and toks[i].startswith("-"):
            o = toks[i]
            if o == "-c" or (o.startswith("-c") and len(o) > 2) or o.startswith(("--config-env", "--exec-path")):
                return Verdict(Risk.DANGER, "overrides git configuration for this command (it can run programs)"), toks
            i += 2 if o in ("-C", "--git-dir", "--work-tree", "--namespace", "--super-prefix") else 1
        norm = [toks[0]] + toks[i:]
        sub = norm[1].lower() if len(norm) > 1 else ""
        rest, lrest = norm[2:], [t.lower() for t in norm[2:]]
        if any(t in ("--ext-diff", "--textconv", "--exec") or t.startswith(("--exec=", "--upload-pack", "--receive-pack", "--open-files-in-pager"))
               for t in lrest) or (sub == "grep" and any(t.startswith("-O") for t in rest)):
            return Verdict(Risk.DANGER, "makes git run an external program"), norm
        if any(t.startswith("--output") for t in lrest):
            return Verdict(Risk.CAUTION, "makes git write its output to a file"), norm
        if sub == "config":
            pos = [t for t in rest if not t.startswith("-")]
            edits = any(t in GIT_CONFIG_EDITS for t in lrest)
            if any(t in GIT_CONFIG_READS for t in lrest) or (len(pos) == 1 and not edits):
                return Verdict(Risk.SAFE, "reads git settings"), norm
            if pos and pos[0].lower() in GIT_CONFIG_BENIGN and not edits:
                return Verdict(Risk.CAUTION, "changes a git setting", rule_key="shell:git config", always_ok=True), norm
            return Verdict(Risk.DANGER, "changes a git setting that can run programs or redirect connections"), norm
        return None, norm

    def _sensitive_target(self, toks: list[str], cwd: Path | None) -> Verdict:
        for t in toks[1:]:
            if t.startswith("-"):
                if "=" not in t:
                    continue
                t = t.split("=", 1)[1]
            if not t or "://" in t or len(t) > 400:
                continue
            if _HIDDEN_HOME_GLOB.match(t):
                return Verdict(Risk.DANGER, f"wildcard over hidden folders in your home folder ({t})")
            for c in [t] + self._expand(t, cwd)[1:]:
                v = self.classify_path(c, "write", cwd, skip_system=True)
                if v.risk >= Risk.DANGER:
                    return Verdict(v.risk, f"command targets a protected location ({t})")
        return Verdict(Risk.SAFE, "")

    def _segment_tokens(self, toks: list[str]) -> Verdict:
        had_env = False
        while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
            toks.pop(0)
            had_env = True
        if not toks:
            return Verdict(Risk.CAUTION if had_env else Risk.SAFE, "sets a variable" if had_env else "empty")
        lowered = [t.lower() for t in toks]
        for i in range(len(toks)):   # also catches "sudo rm -rf /", "xargs rm -rf ~", "nohup rm -rf /"
            if self._catastrophic_delete(_base(toks[i]), lowered[i:]):
                return Verdict(Risk.BLOCKED, "recursive delete of a root, home or system directory")
        raw0 = toks[0]
        if re.search(r"[$*?\[\]{}`%]", raw0):
            return Verdict(Risk.DANGER, "command name is computed or obfuscated")
        cmd = _base(raw0)
        bare = "/" not in raw0 and "\\" not in raw0
        low = [t.lower() for t in toks]
        sub = low[1] if len(low) > 1 else ""

        if cmd in WRAPPERS:
            inner = [_base(t) for t in toks[1:]]
            if any(t in DANGER_CMDS or t in INTERPRETERS for t in inner):
                return Verdict(Risk.DANGER, f"'{cmd}' wraps a sensitive command")
            return Verdict(Risk.CAUTION, f"'{cmd}' wraps another command")
        if cmd in DANGER_CMDS:
            return Verdict(Risk.DANGER, DANGER_CMDS[cmd])
        if cmd == "git":
            gv, toks = self._git(toks)
            if gv is not None:
                return gv
            low = [t.lower() for t in toks]
            sub = low[1] if len(low) > 1 else ""
        if cmd in INTERPRETERS:
            if sub in SAFE_SUB.get(cmd, ()) and len(toks) == 2 and bare:
                return Verdict(Risk.SAFE, "version check")
            if len(toks) == 1 or any(t in ("-c", "-e", "--eval", "-", "-p", "--print", "-r") for t in low[1:]):
                return Verdict(Risk.DANGER, "runs inline code through an interpreter")
            if "-m" in low[1:3] and len(low) > low.index("-m") + 1:
                mod = low[low.index("-m") + 1]
                if mod in ("pip", "ensurepip", "venv", "http.server"):
                    return Verdict(Risk.CAUTION, f"runs python -m {mod}", rule_key=f"shell:{cmd} -m {mod}")
                return Verdict(Risk.CAUTION, f"runs module {mod}", rule_key=f"shell:{cmd} -m {mod}", always_ok=True)
            return Verdict(Risk.CAUTION, "runs a script", rule_key=f"shell:{cmd}")
        if cmd in PKG_SYS:
            if sub in PKG_READ:
                return Verdict(Risk.CAUTION, "queries the system package manager", rule_key=f"shell:{cmd} {sub}", always_ok=True)
            return Verdict(Risk.DANGER, "installs or removes system software")
        if cmd in CLOUD_CMDS:
            words = set(low[1:])
            if words & CLOUD_READ and not words & CLOUD_WRITE:
                return Verdict(Risk.CAUTION, "reads cloud resources", rule_key=f"shell:{cmd}")
            return Verdict(Risk.DANGER, "changes cloud infrastructure")
        if cmd in DEPLOY_CMDS:
            words = set(low[1:])
            if not words or words & {"deploy", "publish", "destroy", "remove", "rm", "up", "--prod",
                                     "secrets", "secret", "env", "login", "db", "promote", "rollback", "scale"} \
                    or any(re.search(r"(reset|destroy|delete|remove|kill|wipe|purge|drop)", t) for t in low[1:]):
                return Verdict(Risk.DANGER, "deploys or changes hosted services")
            return Verdict(Risk.CAUTION, "queries a hosting CLI", rule_key=f"shell:{cmd}")

        if bare and not had_env:
            if cmd in SAFE_CMDS:
                bad = None
                if cmd == "rg" and any(t == "--pre" or t.startswith(("--pre=", "--pre-glob", "--hostname-bin")) for t in low[1:]):
                    return Verdict(Risk.DANGER, "makes the search run a program on every file")
                if cmd == "sort" and any(t.startswith("--compress-program") for t in low[1:]):
                    return Verdict(Risk.DANGER, "makes sort run an external program")
                if any(re.match(r"^/dev/(zero|u?random|full)$", t) for t in low[1:]):
                    bad = "reads from an endless device"
                elif any(t.startswith("--output") for t in low[1:]):
                    bad = "writes a file"
                elif cmd == "sort" and any(t == "-o" or t.startswith("--output") for t in low[1:]): bad = "writes a file"
                elif cmd == "uniq" and len([t for t in low[1:] if not t.startswith("-")]) > 1: bad = "writes a file"
                elif cmd == "date" and any(t in ("-s", "--set") for t in low[1:]): bad = "sets the clock"
                elif cmd == "hostname" and any(not t.startswith("-") for t in low[1:]): bad = "sets the hostname"
                elif cmd == "tree" and "-o" in low[1:]: bad = "writes a file"
                if bad is None:
                    return Verdict(Risk.SAFE, "read-only command")
                return Verdict(Risk.CAUTION, bad)
            if cmd in SAFE_SUB and sub in SAFE_SUB[cmd]:
                if cmd == "git" and sub in _GIT_MUTATING:
                    rest = low[2:]
                    positional = [t for t in rest if not t.startswith("-")]
                    if any(t in _GIT_MUTATING[sub] for t in rest) or (sub != "remote" and positional) or \
                       (sub == "remote" and positional and positional[0] not in ("show", "get-url")):
                        return Verdict(Risk.CAUTION, f"changes git {sub} state", rule_key=f"shell:git {sub}", always_ok=True)
                return Verdict(Risk.SAFE, "read-only command")
        key = f"shell:{cmd}"
        if cmd in MULTI and sub and re.match(r"^[a-z][a-z0-9:_.-]*$", sub):
            key = f"shell:{cmd} {sub}"
        named = cmd not in MULTI or key != f"shell:{cmd}"   # never a blanket "always allow git/npm/docker"
        return Verdict(Risk.CAUTION, "runs a command that may change things", rule_key=key,
                       always_ok=bare and not had_env and named)

    def classify_shell(self, command: str, cwd: Path | None = None) -> Verdict:
        cmd = (command or "").strip()
        if not cmd:
            return Verdict(Risk.SAFE, "empty command")
        low = cmd.lower().replace("\\", "/")
        for rx, why in _BLOCKED_RX:
            if re.search(rx, low):
                return Verdict(Risk.BLOCKED, why)

        found: list[Verdict] = []
        # self-protection: any reference to Saani's home outside the workspace
        scrub = low.replace(self.workspace, " ")
        scrub = re.sub(r"(~|\$home|\$\{home\}|%userprofile%)?/?\.saani/workspace", " ", scrub)
        if self.home in scrub or re.search(r"\.saani\b", scrub) or re.search(r"\.saa\S*[*?\[]|\.s[*?]\S*ni", scrub):
            if any(n in scrub for n in SELF_CRITICAL) or re.search(r"(daemon|vault|audit|config)\S*[*?]", scrub):
                return Verdict(Risk.BLOCKED, "touches Saani's own config, credentials or audit log")
            found.append(Verdict(Risk.DANGER, "touches Saani's own files"))
        if f"127.0.0.1:{self.daemon_port}" in low or f"localhost:{self.daemon_port}" in low or f"[::1]:{self.daemon_port}" in low:
            return Verdict(Risk.BLOCKED, "agents may not talk to Saani's own control port")
        for rx, why in _DANGER_RX:
            if re.search(rx, low):
                found.append(Verdict(Risk.DANGER, why))
        if _SECRET_RX.search(low):
            found.append(Verdict(Risk.DANGER, "references a credentials or secrets location"))

        complex_syntax = False
        # command substitution: classify what is inside
        work = cmd
        for _ in range(4):
            inner = re.findall(r"\$\(([^()]*)\)|`([^`]*)`", work)
            if not inner:
                break
            complex_syntax = True
            for a, b in inner:
                found.append(_worst(self.classify_shell(a or b, cwd), Verdict(Risk.CAUTION, "command substitution")))
            work = re.sub(r"\$\([^()]*\)|`[^`]*`", " $SUBST ", work)
        if "$(" in work or "`" in work or "<(" in work or ">(" in work:
            found.append(Verdict(Risk.DANGER, "nested or process substitution"))
            complex_syntax = True

        stripped = _BENIGN_REDIR.sub(" ", work)
        for op, target in re.findall(r"(>>?)\s*([^\s;|&<>]+)", stripped):
            complex_syntax = True
            target = target.strip("'\"")
            found.append(_worst(self.classify_path(target, "write", cwd), Verdict(Risk.CAUTION, "writes a file through redirection")))
            if op == ">" and self._existing_outside(target, cwd):
                found.append(Verdict(Risk.DANGER, "overwrites an existing file outside the workspace (a shell overwrite cannot be undone)"))
        if re.search(r">\s*$", stripped):
            complex_syntax = True
            found.append(Verdict(Risk.CAUTION, "redirection"))
        stripped = re.sub(r">>?\s*[^\s;|&<>]+", " ", stripped)

        modes = ["posix", "literal"] + (["cmd"] if self.windows else [])
        segments = list(dict.fromkeys(seg for mode in modes for seg in split_unquoted(stripped, mode)))
        seg_verdicts, run_cwd = [], cwd
        for seg in segments:
            seg_verdicts.append(self._segment(seg, run_cwd))
            m = re.match(r"^\s*(?:cd|pushd|chdir|set-location|sl)\s+(?:/d\s+)?(\S.*?)\s*$", seg, re.I)
            if m:
                moved = self._expand(m.group(1).strip("'\""), run_cwd)
                if moved:
                    run_cwd = Path(moved[-1] if len(moved) > 1 else moved[0])
        found.extend(seg_verdicts)
        if not found:
            return Verdict(Risk.SAFE, "empty command")
        top = _worst(*found)
        if top.risk == Risk.CAUTION:
            cautions = [v for v in seg_verdicts if v.risk == Risk.CAUTION]
            if not complex_syntax and cautions and all(v.rule_key for v in cautions):
                # every changing segment is individually recognisable: a saved rule must
                # cover each of them ("cd x && pytest" needs only the pytest rule)
                keys = sorted({v.rule_key for v in cautions})
                return Verdict(Risk.CAUTION, cautions[0].reason,
                               rule_key=keys[0] if len(keys) == 1 else "all:" + "|".join(keys),
                               always_ok=all(v.always_ok for v in cautions))
            return Verdict(Risk.CAUTION, top.reason, rule_key=exact_key("run_shell", {"c": cmd}))
        if top.risk == Risk.SAFE and complex_syntax:
            return Verdict(Risk.CAUTION, "uses shell substitution or redirection", rule_key=exact_key("run_shell", {"c": cmd}))
        if top.risk == Risk.SAFE:
            return Verdict(Risk.SAFE, "read-only command")
        return Verdict(top.risk, top.reason)

    # ----------------------------------------------------------------- python
    def classify_python(self, code: str, workdir: Path | None = None) -> Verdict:
        code = code or ""
        low = code.lower()
        if any(n in low for n in SELF_UNREADABLE) or ("audit.log" in low and ".saani" in low):
            return Verdict(Risk.BLOCKED, "code references Saani's own credentials or audit log")
        for rx, why in _PY_DANGER:
            if re.search(rx, low):
                return Verdict(Risk.DANGER, f"code {why}")
        if _SECRET_RX.search(low) or ".saani" in low:
            return Verdict(Risk.DANGER, "code references a credentials or secrets location")
        writes = bool(_PY_WRITES.search(low))
        dynamic = bool(_PY_DYNAMIC_PATH.search(low))
        for lit in _PY_PATH_LITERAL.findall(code):
            v = self.classify_path(lit, "write" if writes else "read", workdir)
            if v.risk >= Risk.DANGER:
                return Verdict(v.risk, f"code refers to a protected location ({lit})")
            if writes and self._existing_outside(lit, workdir):
                return Verdict(Risk.DANGER, f"code may overwrite an existing file outside the workspace ({lit})")
        if writes and dynamic:
            # the target of the write is computed at run time, so it cannot be checked here: the
            # owner sees this exact code every time instead of approving "run_python" wholesale
            return Verdict(Risk.CAUTION, "runs Python code that writes files at a computed location",
                           rule_key=exact_key("run_python", {"c": code}))
        return Verdict(Risk.CAUTION, "runs Python code", rule_key="tool:run_python", always_ok=True)

    # -------------------------------------------------------------------- web
    def _url(self, url: str) -> Verdict | None:
        try:
            u = urlparse(url if "://" in url else "http://" + url)
            host, port = (u.hostname or "").lower(), u.port
        except ValueError:
            return Verdict(Risk.BLOCKED, "malformed URL")
        if u.scheme not in ("http", "https"):
            return Verdict(Risk.DANGER, f"non-web URL scheme '{u.scheme}'")
        if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0") and (port or 80) == self.daemon_port:
            return Verdict(Risk.BLOCKED, "agents may not talk to Saani's own control port")
        if host in ("169.254.169.254", "metadata.google.internal"):
            return Verdict(Risk.BLOCKED, "cloud metadata endpoint")
        return None

    # ------------------------------------------------------------------ tools
    def classify(self, tool: str, args: dict, default: Risk = Risk.CAUTION,
                 workdir: Path | None = None) -> Verdict:
        a = args or {}
        gen_key = f"tool:{tool}"
        if tool in ("read_file", "list_dir"):
            return self.classify_path(a.get("path", "."), "read", workdir)
        if tool == "search_files":
            return self.classify_path(a.get("root", "."), "read", workdir)
        if tool in ("write_file", "edit_file"):
            return self.classify_path(a.get("path", ""), "write", workdir)
        if tool == "delete_path":
            return self.classify_path(a.get("path", ""), "delete", workdir)
        if tool == "move_path":
            return _worst(self.classify_path(a.get("src", ""), "delete", workdir),
                          self.classify_path(a.get("dst", ""), "write", workdir))
        if tool == "run_shell":
            cwd = Path(os.path.expanduser(a["cwd"])) if a.get("cwd") else workdir
            v = self.classify_shell(a.get("command", ""), cwd)
            if a.get("secrets") and v.risk < Risk.DANGER:
                v = Verdict(Risk.DANGER, "uses stored secrets: " + ", ".join(map(str, a["secrets"])))
            if a.get("background") and v.risk == Risk.SAFE:
                v = Verdict(Risk.CAUTION, "starts a background process", rule_key=exact_key(tool, a))
            if a.get("cwd") and v.risk < Risk.DANGER and is_secret_path(norm_path(cwd)):
                v = Verdict(Risk.DANGER, "runs inside a credentials directory")
            return v
        if tool == "run_python":
            return self.classify_python(a.get("code", ""), workdir)
        if tool == "create_tool":
            v = self.classify_python(a.get("code", ""), workdir)
            return Verdict(max(v.risk, Risk.CAUTION), "creates a reusable tool: " + v.reason, rule_key=exact_key(tool, a))
        if tool in ("web_fetch", "http_request", "open_target"):
            target = a.get("url") or a.get("target") or ""
            is_url = "://" in target or tool != "open_target"
            if is_url:
                bad = self._url(target)
                if bad:
                    return bad
            if tool == "web_fetch":
                return Verdict(Risk.SAFE, "reads a web page")
            if tool == "open_target":
                if is_url:
                    return Verdict(Risk.SAFE, "opens a web page in your browser")
                v = self.classify_path(target, "read", workdir)
                tp = Path(os.path.expanduser(target))
                if not tp.is_absolute() and workdir is not None:
                    tp = Path(workdir) / tp
                runnable = tp.suffix.lower() in EXEC_SUFFIXES
                try:
                    if not runnable and os.name != "nt" and tp.is_file() and os.access(tp, os.X_OK):
                        runnable = True
                except OSError:
                    runnable = True
                if runnable:   # "opening" a program or script runs it, outside every other check
                    return _worst(v, Verdict(Risk.DANGER, "opens a program, script or shortcut, which runs it"))
                return _worst(v, Verdict(Risk.CAUTION, "opens a file or folder with its default program", rule_key=gen_key, always_ok=True))
            method = str(a.get("method", "GET")).upper()
            blob = json.dumps(a, default=str)
            if "{{secret:" in blob:
                return Verdict(Risk.DANGER, "sends a stored secret to a web service")
            if method in ("GET", "HEAD"):
                return Verdict(Risk.SAFE, "reads from a web API")
            host = urlparse(target).hostname or "?"
            return Verdict(Risk.DANGER, f"sends data to {host} ({method})")
        if tool == "type_text" and "{{secret:" in str(a.get("text", "")):
            return Verdict(Risk.DANGER, "types a stored secret into the active window")
        if tool in ("screenshot", "mouse_click", "type_text", "press_keys"):
            # one consent per task covers the whole GUI session; never a permanent rule
            return Verdict(Risk.CAUTION, "sees or controls your screen, mouse and keyboard", rule_key="tool:desktop")
        if default == Risk.SAFE:
            return Verdict(Risk.SAFE, "harmless")
        return Verdict(default, f"uses the {tool} tool", rule_key=gen_key, always_ok=default == Risk.CAUTION)
