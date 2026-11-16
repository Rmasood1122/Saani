# Saani architecture

## The shape of it

```
 voice ─┐                                        ┌─ Skill library   (teams that worked + lessons)
 CLI  ──┼─► Orchestrator ─► Planner ─► Plan ─►   ├─ Agent templates (every agent ever created)
 dash ──┤        │                     (team)    ├─ Custom tools    (built by agents, kept)
 cron ──┘        │                               └─ Memory (SQLite) (results, facts, preferences)
                 ▼
          Agents run in parallel ──► every tool call ──► Guard ──► Policy (SAFE/CAUTION/DANGER/BLOCKED)
                 │                                        │  ├──► you (terminal, dashboard, voice)
                 ▼                                        │  └──► Audit log (hash-chained)
          Reviewer ─► Fixer ─► final answer               ▼
                                                     the tool runs (files, shell, web, desktop)
```

One process. Standard library only for the core. Everything Saani owns is under one folder (`~/.saani`, or `SAANI_HOME`).

## Modules

| Path | Responsibility |
|---|---|
| `saani/core/planner.py` | Asks the planner model for a team through a forced `submit_plan` tool call; validates and repairs the result; falls back to one generalist. |
| `saani/core/orchestrator.py` | The task lifecycle: plan, execute the dependency graph in a thread pool, synthesize, verify and fix, learn. |
| `saani/core/agent.py` | One agent: system prompt (role + persona + rules), the think/act loop, step limit, taint tracking. |
| `saani/security/policy.py` | Rates an action before it runs. Pure functions over the tool name and arguments. |
| `saani/security/guard.py` | Applies the rating: mode, saved rules, per-task grants, the web-content rules, asking you, auditing. |
| `saani/security/secrets.py` | The vault (OS keychain or private file) and the redactor. |
| `saani/security/audit.py` | Append-only log where each entry carries the hash of the one before. |
| `saani/tools/` | `files`, `shell`, `web`, `system` (open, desktop control), `human` (ask, secrets, memory), `forge` (agents build tools). |
| `saani/llm.py` | One message format, two providers (Anthropic Messages API, OpenAI-compatible chat completions), retries, per-task budget. |
| `saani/simllm.py` | A scripted stand-in model for tests, evals and `saani demo`. |
| `saani/skills.py`, `memory.py`, `persona.py` | What is kept between tasks. |
| `saani/voice.py` | Frames to utterances (energy-based detection), speech-to-text, wake word, text-to-speech. |
| `saani/daemon.py`, `dashboard.html` | The background service: task queue, scheduler, voice loop, local HTTP API and page. |
| `saani/autostart.py` | Start at login: registry Run key (Windows), LaunchAgent (macOS), XDG autostart (Linux). |
| `saani/interaction.py` | How you are asked: terminal, dashboard queue, scripted (tests), deny-all (unattended). |

## Life of a task

1. **Entry.** `Orchestrator.run_task(text, workdir=...)`. A task gets an id, a working folder (`~/.saani/workspace/<id>` unless you gave one), a cancel flag and a budget of model calls (`limits.max_llm_calls_per_task`).
2. **Reuse check.** The skill library is searched by similarity to the request. With `skills.auto_reuse = direct` and a strong match the saved team is used as-is with no planning call. Otherwise the best matches are handed to the planner as a starting point.
3. **Planning.** The planner sees the task, persona, relevant memories, matching skills, matching agent templates and the tool catalogue, and must call `submit_plan`: agents (name, role, instructions, tools, dependencies, model tier), how to combine results, success criteria, and a complexity label. Validation drops unknown tools, removes invalid and self-referencing dependencies, breaks cycles, de-duplicates ids and names, and trims to the agent limit. If nothing usable comes back twice, a single generalist with every tool takes the task.
4. **Execution.** Agents whose dependencies are finished are submitted to a thread pool (`limits.max_parallel`). An agent receives the owner's request, the team list, lessons from the skill, relevant memory notes and the full reports of the agents it depends on. A crashed agent fails alone; the rest continue and the final status is `partial`.
5. **The agent loop.** Call the model with the agent's tools. For each tool call: check the agent was given that tool, validate arguments, ask the guard, run, redact the output, add it to the conversation. Stop on a reply with no tool call, on the step limit (one last call with tools switched off asks for a wrap-up), on cancellation or when the budget runs out. Old tool outputs are shortened when the conversation grows large.
6. **Synthesis.** One agent's output is returned as-is. Several are combined by a model call that is told to use only what the agents reported and to say plainly what failed, was refused or is unverified.
7. **Verification.** For plans labelled `complex`, a reviewer with read-only tools checks the result against the success criteria and must answer `VERDICT: PASS` or `VERDICT: FAIL`. On FAIL a fixer with the team's tools repairs the listed problems.
8. **Learning.** Every agent that finished is saved as a template. A small model call decides whether the task is a repeatable kind of work and, if so, names it and lists lessons; the team is stored as a skill. A name collision with a skill for different work creates a second skill instead of merging. The outcome is stored as an episode in memory.

## Models

Three tiers map to three model names in `llm.models`: `planner` (planning, hardest reasoning), `worker` (agents, synthesis, review) and `fast` (skill naming, agents the planner marks as simple). Defaults are `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`. For an OpenAI-compatible endpoint set `llm.provider = openai`, `llm.base_url` and the three names; the model must support tool calling.

## The background service

`saani daemon` binds an HTTP server to 127.0.0.1 only. Every API call needs a random token kept in `~/.saani/daemon.token`; `saani open` passes it to the page in the URL, where it is exchanged for an HttpOnly, SameSite=Strict cookie. Requests with a non-local `Host` header are refused (DNS rebinding), and POSTs need a custom header (cross-site forms). Tasks run up to `daemon.max_concurrent_tasks` at a time. Permission requests and questions wait in a queue that the dashboard shows; if no dashboard has polled in the last few seconds the service opens one. The scheduler fires due jobs once and skips runs missed while the computer was off.

## Voice

Microphone frames (16 kHz, 30 ms) go through an energy-based detector that learns the room's noise floor for a second and then cuts utterances at about 0.8 s of silence. Each utterance is transcribed locally by faster-whisper (or by an OpenAI-compatible transcription endpoint if you choose). The wake word must open the sentence, with spelling variants configurable in `voice.wake_words` because recognisers hear "Saani" as "sunny" or "sonny". Replies are spoken with the operating system's voice; the microphone is ignored while Saani is speaking.

## Extending

- **A tool:** add a `Tool(name, description, schema, fn, risk, taints)` to a module in `saani/tools/` and, if it needs special rating, a branch in `Policy.classify`. Anything not recognised there is rated by the tool's default risk.
- **A provider:** subclass `LLMClient.complete` in `saani/llm.py` and add it to `make_client`.
- **A channel to ask you** (a phone notification, a chat app): subclass `Interaction`.
