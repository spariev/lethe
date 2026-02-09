"""Integration layer — connects actors to the existing Agent/LLM system.

The butler (principal) is a pure coordinator — she NEVER calls tools herself.
All work is delegated to subagents who have the actual tools.
Butler only has actor tools: spawn, kill, send, discover, ping, wait, terminate.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional

from lethe.actor import Actor, ActorConfig, ActorRegistry, ActorState
from lethe.actor.tools import create_actor_tools
from lethe.actor.runner import ActorRunner
from lethe.memory.llm import AsyncLLMClient, LLMConfig

logger = logging.getLogger(__name__)

# Actor tools that the butler uses — everything else goes to subagents
ACTOR_TOOL_NAMES = {
    'send_message', 'wait_for_response', 'discover_actors',
    'terminate', 'spawn_subagent', 'kill_actor', 'ping_actor',
}


class ActorSystem:
    """Manages the actor system, wiring it into the existing Agent.
    
    The butler (principal) only gets actor tools — no file, web, or CLI tools.
    Those are collected and passed to subagents when they're spawned.
    """

    def __init__(self, agent):
        self.agent = agent
        self.registry = ActorRegistry()
        self.principal: Optional[Actor] = None
        self._background_tasks: Dict[str, asyncio.Task] = {}
        
        # Tools from the agent that subagents can use (not the butler)
        self._available_tools: Dict[str, tuple] = {}

    async def setup(self):
        """Set up the actor system.
        
        1. Collect agent's tools for subagent use
        2. Strip non-actor tools from the agent's LLM (butler doesn't use them)
        3. Create principal actor
        4. Register actor tools with the agent
        """
        # Collect all agent tools BEFORE stripping them
        self._collect_available_tools()
        
        # Strip non-actor tools from butler's LLM
        # Butler delegates all work — she only needs actor tools
        if hasattr(self.agent, 'llm') and hasattr(self.agent.llm, '_tools'):
            non_actor = [name for name in self.agent.llm._tools if name not in ACTOR_TOOL_NAMES]
            for name in non_actor:
                del self.agent.llm._tools[name]
            if non_actor:
                logger.info(f"Stripped {len(non_actor)} tools from butler (delegated to subagents)")
        
        # Create principal actor
        self.principal = self.registry.spawn(
            ActorConfig(
                name="butler",
                group="main",
                goals="Serve the user. You are a coordinator — delegate ALL work to subagents.",
            ),
            is_principal=True,
        )
        
        # Set up LLM factory
        self.registry.set_llm_factory(self._create_llm_for_actor)
        
        # Register actor tools with the butler's LLM
        actor_tools = create_actor_tools(self.principal, self.registry)
        for func, _ in actor_tools:
            self.agent.add_tool(func)
        
        # Hook spawn to auto-start actors in background
        original_spawn = self.registry.spawn
        def spawn_and_start(*args, **kwargs):
            actor = original_spawn(*args, **kwargs)
            if not actor.is_principal:
                self._start_actor(actor)
            return actor
        self.registry.spawn = spawn_and_start
        
        tool_count = len(self.agent.llm._tools)
        available_count = len(self._available_tools)
        logger.info(
            f"Actor system initialized. Principal: {self.principal.id}, "
            f"butler tools: {tool_count}, subagent tools available: {available_count}"
        )

    def _collect_available_tools(self):
        """Collect tools from the agent that subagents can request."""
        if hasattr(self.agent, 'llm') and hasattr(self.agent.llm, '_tools'):
            for name, (func, schema) in self.agent.llm._tools.items():
                if name not in ACTOR_TOOL_NAMES:
                    self._available_tools[name] = (func, schema)

    def get_available_tool_names(self) -> List[str]:
        """List tool names available for subagents."""
        return sorted(self._available_tools.keys())

    async def _create_llm_for_actor(self, actor: Actor) -> AsyncLLMClient:
        """Create an LLM client for a subagent actor."""
        config = LLMConfig()
        if actor.config.model:
            config.model = actor.config.model
        else:
            config.model = config.model_aux
        
        config.context_limit = min(config.context_limit, 64000)
        config.max_output_tokens = min(config.max_output_tokens, 4096)
        
        client = AsyncLLMClient(
            config=config,
            system_prompt=actor.build_system_prompt(),
        )
        
        return client

    def _start_actor(self, actor: Actor):
        """Start a non-principal actor running in the background."""
        async def _run():
            try:
                runner = ActorRunner(
                    actor=actor,
                    registry=self.registry,
                    llm_factory=self._create_llm_for_actor,
                    available_tools=self._available_tools,
                )
                result = await runner.run()
                logger.info(f"Actor {actor.config.name} (id={actor.id}) finished: {result[:80]}...")
            except Exception as e:
                logger.error(f"Actor {actor.config.name} (id={actor.id}) error: {e}", exc_info=True)
                if actor.state != ActorState.TERMINATED:
                    actor.terminate(f"Error: {e}")
            finally:
                self._background_tasks.pop(actor.id, None)
        
        task = asyncio.create_task(_run(), name=f"actor-{actor.id}-{actor.config.name}")
        self._background_tasks[actor.id] = task
        actor._task = task
        logger.info(f"Started background actor: {actor.config.name} (id={actor.id})")

    async def shutdown(self):
        """Shut down all actors gracefully."""
        logger.info(f"Shutting down actor system ({self.registry.active_count} active actors)")
        
        for actor in list(self.registry._actors.values()):
            if not actor.is_principal and actor.state != ActorState.TERMINATED:
                actor.terminate("System shutdown")
        
        tasks = list(self._background_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        if self.principal and self.principal.state != ActorState.TERMINATED:
            self.principal.terminate("System shutdown")
        
        self.registry.cleanup_terminated()
        logger.info("Actor system shut down")

    @property
    def status(self) -> dict:
        return {
            "active_actors": self.registry.active_count,
            "background_tasks": len(self._background_tasks),
            "actors": [
                {
                    "id": a.id,
                    "name": a.name,
                    "group": a.group,
                    "state": a.state.value,
                    "goals": a.goals[:80],
                }
                for a in self.registry.all_actors
            ],
        }
