"""Tools for talking to the owner and to long-term memory."""
from __future__ import annotations

from ..security.policy import Risk
from .base import I, S, Tool, ToolContext, ToolResult, obj


def _ask_user(a: dict, ctx: ToolContext):
    ans = ctx.services.interaction.ask(a["question"], task_id=ctx.task_id, agent=ctx.agent)
    if not ans:
        return "No answer (the user is not available). Proceed with the most reasonable assumption and state it in your result."
    return f"The user answered: {ans}"


def _request_secret(a: dict, ctx: ToolContext):
    name = a["name"]
    vault = ctx.services.vault
    if vault.resolve(name) is not None:
        return f"Secret {name} is already stored. Refer to it by name; you will never see its value."
    value = ctx.services.interaction.ask(
        f"Secret needed: {name}. Reason: {a.get('reason', '(none given)')}. It is stored in your vault and never shown to the AI.",
        task_id=ctx.task_id, agent=ctx.agent, secret=True)
    if not value:
        return ToolResult(f"The user did not provide {name}. Continue without it and report what could not be done.", True)
    try:
        vault.set(name, value)
    except ValueError as e:
        return ToolResult(str(e), True)
    return f"Secret {name} stored. Refer to it by name (run_shell 'secrets', or {{{{secret:{name}}}}}); you will never see its value."


def _notify(a: dict, ctx: ToolContext):
    ctx.services.interaction.notify(a["message"])
    ctx.services.events.emit("notify", task_id=ctx.task_id, agent=ctx.agent, text=a["message"])
    return "Delivered."


def _remember(a: dict, ctx: ToolContext):
    kind = a.get("kind") or "fact"
    if kind not in ("fact", "preference"):
        kind = "fact"
    ctx.services.store.add_memory(kind, a["text"], {"task": ctx.task_id, "agent": ctx.agent})
    if kind == "preference":
        ctx.services.persona.add_preference(a["text"])
    return "Saved to long-term memory."


def _recall(a: dict, ctx: ToolContext):
    rows = ctx.services.store.search(a["query"], limit=max(1, min(int(a.get("limit") or 5), 15)))
    if not rows:
        return "Nothing relevant in memory."
    return "\n".join(f"- [{r['kind']}] {r['text'][:600]}" for r in rows)


TOOLS = [
    Tool("ask_user", "Ask the owner a question when you genuinely cannot proceed without their input.",
         obj({"question": S}, ["question"]), _ask_user, Risk.SAFE),
    Tool("request_secret", "Ask the owner to enter a password/API key into the vault. You only get its NAME back.",
         obj({"name": S, "reason": S}, ["name", "reason"]), _request_secret, Risk.SAFE),
    Tool("notify", "Send the owner a short progress note or heads-up.", obj({"message": S}, ["message"]), _notify, Risk.SAFE),
    Tool("remember", "Save a durable fact or an owner preference for future tasks (kind: fact | preference).",
         obj({"text": S, "kind": S}, ["text"]), _remember, Risk.SAFE),
    Tool("recall", "Search long-term memory (past tasks, facts, preferences).",
         obj({"query": S, "limit": I}, ["query"]), _recall, Risk.SAFE),
]
