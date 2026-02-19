from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from agentape.assertions import TraceAssertions
from agentape.cassette import Cassette
from agentape.mock_tools import MockTool, MockToolRegistry
from agentape.proxy import Proxy, _normalize_base_url

logger = logging.getLogger(__name__)

DEFAULT_CASSETTE_DIR = Path("cassettes")


class AgentTestSession:
    """One test session.

    The session:
    1. Starts a local OpenAI-compatible proxy.
    2. Configures record or replay mode based on env/cassette presence.
    3. Provides the proxy base_url for your agent to point at.
    4. After the test, exposes assertions on the recorded trace.

    Args:
        name:         Cassette name — use the test function name.
        cassette_dir: Where cassettes are stored. Defaults to ./cassettes/
        mode:         'auto' (default), 'record', or 'replay'.
                      'auto' records if cassette missing, replays if present.
        real_base_url: Upstream API base (default api.openai.com).
        real_api_key:  Upstream API key (reads OPENAI_API_KEY if not set).
        port:         Proxy port (0 = random).
    """

    def __init__(
        self,
        name: str,
        cassette_dir: Path | str | None = None,
        mode: str = "auto",
        real_base_url: str = "",
        real_api_key: str = "",
        port: int = 0,
        debug: bool = False,
        pricing: dict[str, tuple[float, float]] | None = None,  # NEW
    ) -> None:
        self._pricing = pricing or {}
        self.name = name
        self._cassette_path = Path(cassette_dir) / f"{name}.json"
        self._real_base_url = _normalize_base_url(
            os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
            or real_base_url.rstrip("/")
            or "https://api.openai.com"
        )
        self._real_api_key = (
            real_api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._tool_registry = MockToolRegistry()
        self._proxy: Proxy | None = None
        self._cassette: Cassette | None = None
        self._port = port
        self._debug = debug

        # Determine mode
        force_record = (
            mode == "record" or os.environ.get("AGENTTEST_RECORD", "").strip() == "1"
        )
        if mode == "replay":
            self._mode = "replay"
        elif force_record or not self._cassette_path.exists():
            self._mode = "record"
        else:
            self._mode = "replay"

    async def __aenter__(self) -> AgentTestSession:
        self._cassette = (
            Cassette.load(self._cassette_path)
            if self._mode == "replay"
            else Cassette(self._cassette_path)
        )
        self._proxy = Proxy(
            cassette=self._cassette,
            mode=self._mode,
            real_base_url=self._real_base_url,
            real_api_key=self._real_api_key,
            port=self._port,
            tool_registry=self._tool_registry,
            debug=self._debug,
            pricing=self._pricing,  # NEW
        )
        await self._proxy.start()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc: BaseException | None, tb: object
    ) -> None:
        if self._proxy:
            await self._proxy.stop()
        if self._cassette and self._mode == "record" and exc_type is None:
            self._cassette.save()
            logger.info(
                "Cassette saved: %s (%d entries)",
                self._cassette_path,
                len(self._cassette),
            )

    @property
    def base_url(self) -> str:
        """Point your LLM client here instead of api.openai.com."""
        assert self._proxy, "Session not started — use `async with`"
        return self._proxy.base_url

    @property
    def api_key(self) -> str:
        """Use this as the API key when pointing at the proxy."""
        return "agenttest-proxy-key"

    @property
    def mode(self) -> str:
        return self._mode

    def mock_tool(
        self,
        name: str,
        *,
        returns: Any = None,
        raises: Exception | None = None,
    ) -> MockTool:
        """Register a mock for a tool function.

        The mock records every call. Use the returned MockTool object
        to make assertions, or use the session-level assertion methods.

        Args:
            name:    The tool function name (must match what the LLM sends).
            returns: The value the tool should return when called.
            raises:  An exception to raise instead of returning.
        """
        return self._tool_registry.register(
            name,
            return_value=returns,
            side_effect=raises,
        )

    # ------------------------------------------------------------------ #
    # Assertions — delegate to TraceAssertions + MockToolRegistry         #
    # ------------------------------------------------------------------ #

    @property
    def _trace(self) -> TraceAssertions:
        assert self._proxy, "Session not started."
        return TraceAssertions(self._proxy.recorded_calls)

    # LLM calls
    def assert_llm_call_count(self, n: int) -> None:
        self._trace.assert_llm_call_count(n)

    def assert_llm_calls_at_most(self, n: int) -> None:
        self._trace.assert_llm_calls_at_most(n)

    def assert_llm_calls_at_least(self, n: int) -> None:
        self._trace.assert_llm_calls_at_least(n)

    # Tool calls
    def assert_tool_called(self, name: str) -> None:
        self._trace.assert_tool_called(name)

    def assert_tool_not_called(self, name: str) -> None:
        self._trace.assert_tool_not_called(name)

    def assert_tool_called_before(self, first: str, second: str) -> None:
        self._trace.assert_tool_called_before(first, second)

    def assert_tool_called_with(self, name: str, **kwargs: Any) -> None:
        self._trace.assert_tool_called_with(name, **kwargs)

    def assert_tool_call_count(self, name: str, n: int) -> None:
        self._trace.assert_tool_call_count(name, n)

    def assert_tool_not_called_after(self, tool: str, after: str) -> None:
        self._trace.assert_tool_never_called_after(tool, after)

    # Cost / tokens
    def assert_cost_under(self, max_usd: float) -> None:
        self._trace.assert_cost_under(max_usd)

    def assert_tokens_under(self, max_tokens: int) -> None:
        self._trace.assert_tokens_under(max_tokens)

    # Content
    def assert_final_response_contains(self, substring: str) -> None:
        self._trace.assert_final_response_contains(substring)

    def assert_final_response_not_contains(self, substring: str) -> None:
        self._trace.assert_final_response_not_contains(substring)

    def assert_finished_cleanly(self) -> None:
        self._trace.assert_finished_cleanly()

    # raw access for custom assertions
    @property
    def calls(self) -> list[dict[str, Any]]:
        assert self._proxy, "Session not started."
        return self._proxy.recorded_calls

    @property
    def tools(self) -> MockToolRegistry:
        return self._tool_registry

    # Debug
    def print_summary(self) -> None:
        self._trace.print_summary()
