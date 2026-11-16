# Saani

**Saani** (ثانی, "the second one") is a personal multi-agent AI system that runs on your own computer and works the way you would: you say or type what you want, it decides how many agents the job needs, creates them, runs them in parallel, checks the result, and keeps what it learned so the same kind of work is faster next time. It starts when you log in, and it asks you before anything sensitive.

It has no required dependencies beyond Python 3.10+. Voice, desktop control and the OS keychain are optional extras.

## Install

Run these from the `saani` folder. They call the environment's Python directly, so no activation step is needed.

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\python -m pip install ".[all]"     # or "." for the core only; extras: voice, desktop, secure, window
.venv\Scripts\saani init        # model + API key, 7 questions about how you work, start-at-login
.venv\Scripts\saani doctor      # shows what is ready and what is missing
```

```bash
# macOS / Linux
python3 -m venv .venv
.venv/bin/python -m pip install ".[all]"
.venv/bin/saani init
.venv/bin/saani doctor
```

The voice extra needs Python 3.12 or 3.13 on Windows at the moment (some speech packages have no ready-made builds for newer Python yet). On a newer Python, install the core and the other extras, and leave voice off.

### Your own model

Saani can also run on a model you train yourself. The [saanimind](saanimind/README.md) project
builds one from scratch and serves it locally; Saani uses it through provider option 5 in
`saani init`, or alongside Claude for selected tiers (`llm.own.tiers`). Turn on
`llm.record_traces` to collect training data from your real tasks.

`saani init` stores your API key in the OS keychain (with the `secure` extra) or in a private file under `~/.saani`. It works with Anthropic (default: Opus plans, Sonnet works, Haiku for small jobs), OpenAI, or any OpenAI-compatible endpoint such as a local Ollama server, in which case nothing leaves your machine.

Platform notes for voice: Linux needs PortAudio and a speech engine (`sudo apt install libportaudio2 espeak-ng`); macOS will ask for microphone permission the first time; the first run downloads the Whisper speech model (about 150 MB for `base`).

Try it without an API key: `saani demo` replays a scripted model so you can watch planning, parallel agents, a permission prompt and a skill being saved.

## Using it

| You want to | Do this |
|---|---|
| Talk to it | Log in. The background service is already running. Say **"Saani, fix the failing tests in my shop project."** Saying just "Saani" makes it answer "Yes?" and wait for the request. "Saani, stop everything" cancels. |
| Type to it | `saani run "..."` for one task in the terminal, `saani chat` for a session, or the dashboard. |
| Look at it | `saani window` shows Saani in its own desktop window (needs the `window` extra). `saani open` opens the same dashboard in your browser for more detail. Settings and the info buttons are in both. |
| Work inside a project | `saani run --dir ~/projects/shop "..."`, or mark folders as yours once: `saani trust add ~/projects/shop`. Agents may edit files there without asking. |
| Hand off and walk away | `saani submit "..."` queues it with the background service. |
| Run something on a schedule | `saani schedule add ai-news "Research today's AI news and write my brief" --weekdays 08:30` |
| See what happened | `saani history`, `saani audit tail`, and the dashboard's live feed. |
| Undo a file change | `saani trash`, then `saani trash restore <number>`. |

When an agent needs your permission, the service says so aloud and opens the dashboard by itself if you do not have it open.

## How a task runs

1. **Plan.** A planner model reads your request, your persona, relevant memory and any saved skill that looks similar, and returns a team: one agent for a simple request, several when the parts need different expertise or can run at once (never more than `limits.max_agents`, default 8). A broken plan is repaired (unknown tools, cycles, duplicates) or replaced by a single generalist.
2. **Run.** Agents with no dependency on each other run in parallel. Each has its own role, instructions and only the tools the planner gave it. Results flow to the agents that depend on them.
3. **Check.** For complex tasks a reviewer inspects the actual files; if it finds problems a fixer repairs them before you see the answer.
4. **Report.** You get one answer that says what was done, where the files are, and what failed or was refused.
5. **Keep.** The team becomes a reusable **skill**, every agent becomes a **template**, any tool an agent built is saved, and the outcome goes into **memory**.

## What is reused

- **Skills** (`saani skills`): the team that worked plus lessons learned, as plain JSON under `~/.saani/skills`. A similar request starts from the skill. Set `skills.auto_reuse` to `direct` to skip planning entirely for near-identical requests.
- **Agent templates** (`saani agents`): the planner can pull any earlier agent into a new team.
- **Custom tools** (`saani tools`): agents can write a tool once (`create_tool`); it runs in its own process and is available to every later task.
- **Memory** (`saani memory search ...`): past results, facts and your stated preferences are searched for every new plan.

## Making it think like you

`~/.saani/persona.md` is read by every agent before it acts: who you are, your priorities, how you decide, your stack and standards, how you write, what it must never do. `saani init` builds it from seven questions; edit it any time. Teach it as you go:

```bash
saani feedback "Always write a failing test before fixing a bug"
saani feedback "Client emails: short, no exclamation marks, sign off with - Ahmad"
```

The persona is the single biggest lever on quality. An empty one gives you a generic assistant.

## Permissions

Every action an agent takes is rated before it runs.

| Level | Examples | What happens |
|---|---|---|
| SAFE | reading files, `ls`, `git status`, web search, writing inside its own workspace or a folder you trusted | runs |
| CAUTION | `pytest`, `npm install`, writing a file elsewhere, running a script | asks once; you can answer *once*, *rest of this task* or *always* |
| DANGER | deleting, `sudo`, anything touching passwords or keys, `git push`, sending data out, installing system software, startup files, typing a stored password | asks every time; can never be "always" |
| BLOCKED | wiping a disk or your home folder, changing Saani's own config, vault or audit log | never runs |

Three modes (`saani config set security.mode ...`): `strict` asks for everything that is not read-only; `ask` (default) is the table above; `trusted` runs CAUTION without asking and still asks for DANGER.

Things worth knowing:

- **Passwords and keys never reach the model.** Agents refer to a secret by name; you type the value into the vault (`saani secret set NAME`, or when an agent asks), and it is injected only into the command that needs it, after you approve. Values are scrubbed from every output, log and message.
- **After an agent reads a web page, it loses its privileges.** Saved rules and trusted mode stop applying to it, it cannot save to memory, edit your project files, open pages in your browser or fetch addresses it made up without asking you. This is what stops a malicious page from steering it.
- **Voice can refuse a request but not approve one**, unless you set `security.voice_approval` (and then only for CAUTION).
- **File changes can be undone.** Anything an agent deletes or overwrites with its file tools is kept in `~/.saani/trash`.
- **Everything is logged** in a tamper-evident audit log: `saani audit tail`, `saani audit verify`.
- `saani check <command>` shows how any shell command would be rated. `saani rules` lists your "always" rules.

Read [docs/SECURITY.md](docs/SECURITY.md) before switching to `trusted` mode. It lists what this design does not protect against.

## Commands

`init` · `doctor` · `demo` · `run` · `chat` · `daemon` · `open` · `window` · `status` · `stop` · `submit` · `listen` (test the microphone) · `autostart install|remove|status` · `persona show|init|add|path` · `feedback` · `skills` · `agents` · `tools` · `memory search|add` · `schedule add|list|remove` · `secret set|list|delete` · `trust add|remove|list` · `rules` · `check` · `trash` · `audit tail|verify` · `history` · `config get|set`

Add `--home <folder>` (or set `SAANI_HOME`) to use a different data folder.

## What has and has not been tested

Run them yourself: `python -m unittest discover -s tests -t .` (or `pytest`) and `python -m evals.run_evals`.

- **181 unit tests** pass. The original 168 were run on Python 3.10, 3.11 and 3.13; the 13 dashboard and own-model tests were added later and run on one Python version. They cover the policy, guard, vault, redaction, audit log, tools, both model providers (against a local stand-in server), planner, orchestrator, voice logic, scheduler, autostart file generation for all three systems, the background service and the CLI.
- **27 end-to-end scenarios** pass with a scripted model, including one that obeys every malicious instruction it is given. See [evals/REPORT.md](evals/REPORT.md).
- The installed command was run for real on Linux: `saani demo`, the background service, a task submitted and approved through its API.

Not tested, because it needs your machine or your key:

- **A real model.** No API key was available in the build environment, so no real model has ever driven this system. `python -m evals.live_eval` runs six real tasks (a bug fix, a data report, a refusal, a prompt-injection page, a persona check) against your model. Run it first; it costs a few cents.
- **A real microphone and speakers.** The wake-word and speech pipeline is tested with synthetic audio and a fake recogniser. Check yours with `saani listen`.
- **Windows and macOS.** Start-at-login, PowerShell handling and the speech commands for those systems are tested with fakes on Linux only.
- **Real mouse and keyboard control**, and the **desktop window** (`saani window`) on Windows. The dashboard was rendered and clicked through in headless Chrome on Linux (no script errors, settings and info dialogs work); it has not been seen on Windows.
- **Web search results.** The DuckDuckGo result parser is tested against a hand-written sample, not the live site.

## Limits

- This is not a sandbox. Agents run commands as you. The rating system is a careful filter in front of a human, and the human is the real control.
- Whatever agents read is sent to the model provider you chose. Use a local model for work that must not leave the machine.
- "Thinks like you" means it follows what you wrote in the persona and taught it through feedback. It does not learn from watching you.
- It will not apply for jobs, send messages or publish anything on its own: those actions always stop for your approval. Plan for it to prepare drafts and applications for you to send.
- Without the `secure` extra the vault file is only obfuscated, not encrypted.
