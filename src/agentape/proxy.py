"""OpenAI-compatible HTTP proxy — record and replay LLM calls.

Root cause of the original bug:
  The openai SDK uses base_url as-is and appends /chat/completions (no /v1).
  So when base_url=http://proxy:port, calls go to /chat/completions, not /v1/chat/completions.
  We must listen at /chat/completions.

URL normalization:
  OPENAI_BASE_URL may or may not include /v1 (OpenRouter includes it, api.openai.com does not).
  We normalize by stripping trailing /v1 and always appending /v1/chat/completions when forwarding.
  This works for all providers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web

from agentape.cassette import Cassette, CassetteEntry, _hash_request

if TYPE_CHECKING:
    from agentape.mock_tools import MockToolRegistry

logger = logging.getLogger(__name__)


class CassetteMissError(Exception):
    def __init__(self, msg_hash: str, messages: list[dict]) -> None:
        self.msg_hash = msg_hash
        last = messages[-1] if messages else {}
        super().__init__(
            f"\n\nCassette miss — hash={msg_hash}\n"
            f"Last message role: {last.get('role', '?')}\n"
            f"Content: {str(last.get('content', ''))[:100]}\n\n"
            f"Re-record this test:\n"
            f"    AGENTTEST_RECORD=1 uv run pytest <test_file> --record\n"
        )


def _json_error(status: int, message: str) -> web.Response:
    """Return a well-formed JSON error the openai SDK can parse cleanly."""
    body = {
        "error": {
            "message": message,
            "type": "proxy_error",
            "param": None,
            "code": str(status),
        }
    }
    return web.Response(
        text=json.dumps(body),
        content_type="application/json",
        status=status,
    )


def _normalize_base_url(url: str) -> str:
    """Strip trailing /v1 so we can always append /v1/chat/completions ourselves.

    https://api.openai.com          → https://api.openai.com
    https://openrouter.ai/api/v1    → https://openrouter.ai/api
    http://localhost:11434/v1       → http://localhost:11434
    """
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


class Proxy:
    """OpenAI-compatible HTTP proxy.

    Args:
        cassette:      Cassette to record into / replay from.
        mode:          'record' or 'replay'.
        real_base_url: Upstream API base URL. /v1 suffix is stripped and re-added
                       automatically so both https://api.openai.com and
                       https://openrouter.ai/api/v1 work identically.
        real_api_key:  Upstream API key (record mode only).
        port:          0 = pick a random free port.
        tool_registry: Optional mock tool registry.
        debug:         Log request/response bodies.
    """

    def __init__(
        self,
        cassette: Cassette,
        mode: str,
        real_base_url: str = "https://api.openai.com",
        real_api_key: str = "",
        port: int = 0,
        tool_registry: MockToolRegistry | None = None,
        debug: bool = False,
        pricing: dict[str, tuple[float, float]] | None = None,  # NEW
    ) -> None:
        self._cassette = cassette
        self._mode = mode
        self._pricing = pricing or {}
        # Normalize: strip /v1 suffix so we always control the full path
        self._real_base_url = _normalize_base_url(real_base_url)
        self._real_api_key = real_api_key
        self._port = port
        self._tool_registry = tool_registry
        self._debug = debug
        self._call_index = 0
        self.recorded_calls: list[dict[str, Any]] = []

        self._app = web.Application()

        # The openai SDK appends /chat/completions to base_url (no /v1 prefix).
        # Register both paths for maximum compatibility.
        self._app.router.add_post("/chat/completions", self._handle_chat)
        self._app.router.add_post("/v1/chat/completions", self._handle_chat)
        self._app.router.add_route("*", "/{path_info:.*}", self._handle_passthrough)

        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    @property
    def base_url(self) -> str:
        """Base URL to pass to your LLM client instead of api.openai.com."""
        return f"http://127.0.0.1:{self._port}"

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await self._site.start()
        sockets = self._site._server.sockets  # type: ignore[attr-defined]
        self._port = sockets[0].getsockname()[1]
        logger.info(
            "Proxy listening on %s (mode=%s upstream=%s/v1)",
            self.base_url,
            self._mode,
            self._real_base_url,
        )

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    # ------------------------------------------------------------------ #
    # Chat completions handler                                             #
    # ------------------------------------------------------------------ #

    async def _handle_chat(self, request: web.Request) -> web.Response:
        try:
            raw = await request.read()
            body = json.loads(raw)

            if self._debug:
                logger.debug(
                    "PROXY ← %s %s body=%s",
                    request.method,
                    request.path,
                    raw[:400].decode(errors="replace"),
                )

            messages: list[dict] = body.get("messages", [])
            tools: list[dict] | None = body.get("tools")

            if self._mode == "replay":
                return await self._replay(body, messages, tools)
            else:
                return await self._record(body, messages, tools)

        except CassetteMissError:
            raise
        except Exception as exc:
            logger.exception("Proxy _handle_chat error")
            return _json_error(500, f"agenttest proxy error: {exc}")

    # ------------------------------------------------------------------ #
    # Passthrough — everything that isn't chat/completions               #
    # ------------------------------------------------------------------ #

    async def _handle_passthrough(self, request: web.Request) -> web.Response:
        path = request.match_info.get("path_info", "")

        if self._debug:
            logger.debug("PROXY passthrough: %s %s", request.method, path)

        # In replay mode return a stub for model listing etc.
        if self._mode == "replay":
            if "model" in path:
                return web.Response(
                    text='{"object":"list","data":[{"id":"gpt-4o-mini","object":"model"}]}',
                    content_type="application/json",
                )
            return web.Response(
                text='{"object":"stub"}',
                content_type="application/json",
            )

        # Record mode — forward as-is
        if not self._real_api_key:
            return _json_error(401, "No API key. Set OPENAI_API_KEY.")

        url = f"{self._real_base_url}/v1/{path.lstrip('/')}"
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    request.method,
                    url,
                    headers={
                        "Authorization": f"Bearer {self._real_api_key}",
                        "Content-Type": "application/json",
                    },
                    data=await request.read(),
                ) as resp:
                    body_bytes = await resp.read()
                    return web.Response(
                        body=body_bytes,
                        content_type="application/json",
                        status=resp.status,
                    )
        except Exception as exc:
            return _json_error(502, f"Upstream passthrough error: {exc}")

    # ------------------------------------------------------------------ #
    # Replay                                                               #
    # ------------------------------------------------------------------ #

    async def _replay(
        self,
        body: dict,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> web.Response:
        entry = self._cassette.match(messages, tools)
        if entry is None:
            from agentape.cassette import _hash_request

            raise CassetteMissError(_hash_request(messages, tools), messages)

        self._record_call(body, entry.response, entry)
        resp_text = json.dumps(entry.response)

        if self._debug:
            logger.debug("PROXY → (replay) %s", resp_text[:300])

        return web.Response(
            text=resp_text,
            content_type="application/json",
            status=200,
        )

    # ------------------------------------------------------------------ #
    # Record                                                               #
    # ------------------------------------------------------------------ #

    async def _record(
        self,
        body: dict,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> web.Response:
        if not self._real_api_key:
            return _json_error(401, "No API key. Set OPENAI_API_KEY.")

        # Always forward as non-streaming — simplifies capture significantly
        forward_body = {**body, "stream": False}

        # /v1 suffix is controlled here — _real_base_url has it stripped
        url = f"{self._real_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._real_api_key}",
            "Content-Type": "application/json",
        }

        if self._debug:
            logger.debug("PROXY → (record) %s model=%s", url, body.get("model"))

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, json=forward_body, headers=headers
                ) as upstream:
                    resp_text = await upstream.text(encoding="utf-8")

                    if self._debug:
                        logger.debug(
                            "UPSTREAM ← status=%d body=%s",
                            upstream.status,
                            resp_text[:400],
                        )

                    if upstream.status != 200:
                        # Forward error response so SDK sees a real error message
                        logger.warning(
                            "Upstream returned %d: %s", upstream.status, resp_text[:200]
                        )
                        return web.Response(
                            text=resp_text,
                            content_type="application/json",
                            status=upstream.status,
                        )

                    if not resp_text.strip():
                        logger.error("Upstream returned empty body (status 200)")
                        return _json_error(502, "Upstream returned empty response body")

                    try:
                        resp_body = json.loads(resp_text)
                    except json.JSONDecodeError as exc:
                        logger.error("Upstream returned non-JSON: %s", resp_text[:300])
                        return _json_error(502, f"Upstream returned non-JSON: {exc}")

        except aiohttp.ClientConnectorError as exc:
            msg = f"Cannot connect to upstream {self._real_base_url}: {exc}"
            logger.error(msg)
            return _json_error(502, msg)
        except asyncio.TimeoutError:
            return _json_error(504, f"Upstream timeout after 120s ({url})")
        except Exception as exc:
            logger.exception("Upstream request failed")
            return _json_error(502, str(exc))

        # Save to cassette and record for assertions
        h = _hash_request(messages, tools)
        usage = resp_body.get("usage", {})
        entry = CassetteEntry(
            request_hash=h,
            request=body,
            response=resp_body,
            is_streaming=False,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=resp_body.get("model", body.get("model", "")),
            call_index=self._call_index,
        )
        self._cassette.add(entry)
        self._record_call(body, resp_body, entry)

        resp_text_out = json.dumps(resp_body)
        if self._debug:
            logger.debug("PROXY → (recorded) %s", resp_text_out[:300])

        return web.Response(
            text=resp_text_out,
            content_type="application/json",
            status=200,
        )

    # ------------------------------------------------------------------ #
    # Call recording (powers all assertions)                              #
    # ------------------------------------------------------------------ #

    def _record_call(self, request: dict, response: dict, entry: CassetteEntry) -> None:
        messages = request.get("messages", [])
        choices = response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}

        tool_calls: list[dict] = []
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": args,
                }
            )

        call_record = {
            "index": self._call_index,
            "messages": messages,
            "message_count": len(messages),
            "response_content": message.get("content"),
            "tool_calls": tool_calls,
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "prompt_tokens": entry.prompt_tokens,
            "completion_tokens": entry.completion_tokens,
            "model": entry.model,
            "cost_usd": _estimate_cost(
                entry.model,
                entry.prompt_tokens,
                entry.completion_tokens,
                self._pricing,
            ),
        }
        self.recorded_calls.append(call_record)
        self._call_index += 1

        if self._debug:
            logger.debug(
                "RECORDED call #%d: finish=%s tools=%s",
                self._call_index - 1,
                call_record["finish_reason"],
                [t["name"] for t in tool_calls],
            )

        if self._tool_registry and tool_calls:
            for tc in tool_calls:
                if self._tool_registry.has(tc["name"]):
                    self._tool_registry.record_call(tc["name"], tc["arguments"])


def _estimate_cost(
    model: str,
    prompt: int,
    completion: int,
    pricing: dict[str, tuple[float, float]],
) -> float:
    """Estimate cost using user-provided pricing.

    Args:
        model: Model name from API response
        prompt: Prompt token count
        completion: Completion token count
        pricing: Dict mapping model name to (input_price_per_token, output_price_per_token)

    Returns:
        Estimated cost in USD, or 0.0 if model not in pricing dict
    """
    if not pricing:
        return 0.0

    model_lower = (model or "").lower()

    # Exact match first
    if model_lower in pricing:
        input_price, output_price = pricing[model_lower]
        return prompt * input_price + completion * output_price

    # Partial match (for model names like "gpt-4o-mini-2024-07-18")
    for key, (input_price, output_price) in pricing.items():
        if key in model_lower:
            return prompt * input_price + completion * output_price

    return 0.0
