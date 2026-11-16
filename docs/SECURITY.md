# Saani security model

Saani gives a language model your hands on your computer. This document says what stands between that model and a bad outcome, what was found when those defences were attacked, and what they do not cover.

## What can go wrong

1. **The model makes a mistake.** It deletes the wrong folder, force-pushes, overwrites a file.
2. **Someone else steers the model.** A web page, a search result or a screenshot contains text written to look like instructions (prompt injection): "upload the SSH key", "add this to startup".
3. **Secrets leak.** A password or API key ends up in the model's context, in a log, or on someone's server.
4. **The agent loosens its own leash.** It edits its permissions, its audit log, or talks to its own control port.

## The layers

**1. Every action is rated before it runs** (`security/policy.py`).
Only a strict allowlist is SAFE. Anything the classifier does not positively recognise is CAUTION, so hiding a command can turn "runs silently" into "asks you" but never the reverse. Shell commands are split on every separator under more than one quoting rule and each part is rated; command substitutions are rated from the inside out; redirection targets are rated as file writes; wildcards and `~`/`$HOME` are expanded and what they match is checked; `cd` is followed through the line.

**2. The guard decides whether to ask** (`security/guard.py`).
SAFE runs. CAUTION asks unless you saved a rule (`ask` mode) or chose `trusted` mode. DANGER asks every time and can never be saved as "always". BLOCKED never runs and never asks. With nobody available (a scheduled run, a closed dashboard, a timeout) the answer is no.

**3. Reading the web removes privileges.**
An agent that has used a web tool or taken a screenshot is marked, and so is every agent that receives its output. From then on, for that agent: saved rules, trusted mode and earlier approvals no longer apply; it must ask before saving to long-term memory, before changing any file outside Saani's own workspace, before opening a page in your browser, and before fetching any address that was not shown to it verbatim (in your request, a search result or a page). The last rule is what stops private data leaving inside a URL: following a link is free, composing one is not.
Web content is also wrapped in `<untrusted_content>` tags and agents are told it is data. That instruction helps a good model; the rules above hold even if the model obeys the page completely.

**4. Secrets stay out of the model.**
Agents get secret *names*. Values live in the OS keychain (or a private file) and are injected into a command's environment only after a DANGER approval. Child processes do not inherit secret-looking environment variables. Every tool output and every message bound for the model passes through a redactor that removes vault values and common key shapes. Reading a credentials file is DANGER; searches skip credential files and folders entirely.

**5. Saani protects itself.**
Agents cannot write its config, database, vault, token or audit log (BLOCKED), cannot read the token or vault, and cannot call its own control port. Its own skills, tools and persona are DANGER to modify. The control API is local-only and token-protected.

**6. You can see and undo.**
Every attempt, allowed or not, goes into a hash-chained audit log (`saani audit verify` detects edits). Files changed or deleted through the file tools are copied to the trash first (`saani trash restore`).

## What the evaluations found

The system was attacked with a scripted model that does exactly what it is told, including by a malicious page. The findings below were all real, were fixed, and each has a regression test.

**From the scenario suite**

- A plan with duplicated agent ids left agents depending on themselves.
- When the skill-naming step returned a name that already existed for a different kind of work, the unrelated task was merged into the old skill, which would have made later unrelated requests match it.

**From reading the code as an attacker**

- `open_target` on a script or program ran it under a rule that could be saved as "always".
- In `trusted` mode a shell redirect or a Python `open(..., "w")` could overwrite an existing file outside the workspace with no way back. Both are now DANGER; file-tool overwrites keep the previous version.
- An agent that had read a web page could still, without any prompt: save "preferences" into the persona that every later agent reads; put private data into the query string of a web request; edit files in a trusted project folder; open attacker pages in your real browser.
- Text from a web task flowed into the saved skill's lessons and so into future prompts.
- Writing `.git/config` in a project folder was rated as an ordinary write, and `git status` is SAFE, so the pair gave code execution with no prompt (`core.fsmonitor`). On Windows, a planted `git.bat` in the working folder would have had the same effect (the fix for that is a process setting whose presence is tested, but whose effect could not be checked without Windows).
- A file search over a home folder returned private key material line by line, which the redactor did not recognise in that form.
- Screenshot coordinates were wrong on high-density displays.

**From a second command list written to break the policy**

The first command list had been used to tune the policy and scored 100%, which proves nothing. A second list of 292 commands was written afterwards and scored once before any change:

| Measure | First score | After fixes |
|---|---|---|
| Catastrophic commands hard-blocked | 20/25 | 24/25 |
| Dangerous commands rated DANGER | 121/155 | 155/155 |
| Dangerous commands that at least ask | 144/155 | 155/155 |
| Read-only commands that run without a prompt | 59/61 | 61/61 |
| Routine dev commands rated CAUTION | 51/51 | 51/51 |

Eleven dangerous commands would have run with no prompt: credential reads hidden behind wildcards (`cat ~/.s*h/id_*`), recursive searches of the whole home folder (`grep -r BEGIN ~`), read-only commands with a way to launch programs (`rg --pre`, `sort --compress-program`, `git diff --ext-diff`), and PowerShell pipelines with script blocks (`Where-Object { Remove-Item ... }`). All are now DANGER. The one catastrophic command still not hard-blocked, `for d in /; do rm -rf $d; done`, is rated DANGER because the target is a variable; it always asks and shows you the command.

That second list has now also been tuned against. The 93% it scored cold on "at least asks" is the best available estimate of how the policy does on commands nobody has thought of yet.

## What this does not protect against

- **It is not a sandbox.** Commands run with your full user rights. If you approve a command, it runs. If the classifier under-rates something new and you are in `trusted` mode, it runs.
- **Scripts are judged by their command line.** `python build.py` is CAUTION whatever `build.py` contains, and an agent may have written that file. In `ask` mode you see the command, not the script. Inline code is inspected with patterns, which a determined obfuscator can beat; it then asks rather than runs, unless you saved an "always" rule for it.
- **`trusted` mode is a decision to trust the model.** It removes the prompt for ordinary changes, including edits to any file you could edit. Use it for a model and a kind of task you have watched succeed, and not for tasks that browse the web (browsing agents ignore it anyway).
- **Only the web tools and screenshots mark an agent as exposed.** A repository you cloned, a downloaded PDF or an email saved to disk can also carry planted instructions, and reading them does not remove privileges. In `ask` mode such an agent still has to ask for anything beyond reading and working in its own folder.
- **Slow leaks remain possible.** An exposed agent can still choose *which* of the links a page shows it to follow, and its search queries go to the search engine.
- **Redaction is pattern-based.** A secret in an unusual format, sitting in an ordinary file the agent is allowed to read, will reach the model provider.
- **Your model provider sees what agents read.** File contents, command output and screenshots are sent to the API you configured.
- **The vault file is not encrypted** without the `secure` extra (OS keychain).
- **Voice has no speaker verification.** Anyone in the room can give a task. They cannot approve a permission request by voice unless you enabled that.
- **Nothing here has been tested against a real model, on Windows or macOS, or by an outside security reviewer.**

## Recommended settings

- Keep `security.mode` on `ask`. Build up "always" rules for the dozen commands you really run all day.
- Trust project folders individually (`saani trust add`), never your home folder.
- Install the `secure` extra so the vault uses the OS keychain.
- Run `python -m evals.live_eval` after connecting a model. Its fourth and fifth tasks check that your model takes no for an answer and ignores a planted instruction.
- Look at `saani audit tail` now and then. It is the record of what was tried, not only what succeeded.
