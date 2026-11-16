"""Wires everything together into one ``Services`` object."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config
from .events import EventBus
from .interaction import DenyInteraction, Interaction
from .llm import LLMClient, LLMRouter, TraceRecorder, make_client, own_model_tiers
from .memory import Store
from .persona import Persona
from .security.audit import AuditLog
from .security.guard import Guard
from .security.policy import Policy
from .security.secrets import Redactor, Vault
from .skills import AgentRegistry, SkillLibrary
from .tools import files, forge, human, shell, system, web
from .tools.base import ToolRegistry


@dataclass
class Services:
    config: Config
    store: Store
    vault: Vault
    redactor: Redactor
    audit: AuditLog
    policy: Policy
    guard: Guard
    interaction: Interaction
    events: EventBus
    tools: ToolRegistry
    skills: SkillLibrary
    agents: AgentRegistry
    persona: Persona
    llm: Any

    def set_interaction(self, interaction: Interaction) -> None:
        self.interaction = interaction
        self.guard.interaction = interaction


def build(config: Config | None = None, interaction: Interaction | None = None,
          client: LLMClient | None = None, use_keyring: bool | None = None) -> Services:
    config = config or Config.load()
    config.ensure_dirs()
    redactor = Redactor()
    vault = Vault(config.home, redactor, use_keyring=use_keyring)
    interaction = interaction or DenyInteraction()
    audit = AuditLog(config.audit_path)
    policy = Policy(config.home, config.workspace, config.get("security.trusted_dirs", []),
                    config.get("daemon.port", 8765))
    guard = Guard(config, policy, interaction, audit, redactor)
    tools = ToolRegistry()
    for module in (files, shell, web, system, human, forge):
        for t in module.TOOLS:
            tools.register(t)
    if system.desktop_available():
        for t in system.DESKTOP_TOOLS:
            tools.register(t)
    forge.load_custom_tools(tools, config.tools_dir)
    client = client or make_client(config, vault)
    recorder = TraceRecorder(config.traces_dir) if config.get("llm.record_traces", False) else None
    llm = LLMRouter(client, config.get("llm.models", {}), config.get("llm.max_tokens", 16000), redactor,
                    tier_clients=own_model_tiers(config), recorder=recorder)
    persona = Persona(config.persona_path, config.get("user_name", ""))
    persona.ensure()
    return Services(config=config, store=Store(config.db_path), vault=vault, redactor=redactor,
                    audit=audit, policy=policy, guard=guard, interaction=interaction, events=EventBus(),
                    tools=tools, skills=SkillLibrary(config.skills_dir), agents=AgentRegistry(config.agents_dir),
                    persona=persona, llm=llm)
