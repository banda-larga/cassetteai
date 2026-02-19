# agentape

**Deterministic testing for LLM agents**

Record LLM interactions once, replay them indefinitely. Zero API costs in CI. Framework-agnostic. Works with any OpenAI-compatible API.

```
record    ●──●──●──●──●   (real API, saved to cassette)
replay    ●──●──●──●──●   (cassette, <50ms, $0.00)
```

## Table of Contents

- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage Patterns](#usage-patterns)
- [API Reference](#api-reference)
- [Assertions](#assertions)
- [Framework Integration](#framework-integration)
- [Provider Configuration](#provider-configuration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Problem Statement

Testing LLM agents is expensive and nondeterministic:

- **Cost**: Running integration tests against real APIs costs money and burns through rate limits
- **Speed**: Each test takes seconds or minutes waiting for API responses
- **Flakiness**: Nondeterministic outputs cause test failures that are difficult to reproduce
- **CI/CD**: Running tests in CI requires managing API keys and incurs ongoing costs

**agentape solves this** by recording LLM interactions once and replaying them deterministically on subsequent runs. Tests run in milliseconds with zero API cost.

---

## Installation

```bash
# Core library
pip install agentape

# With LangGraph support
pip install "agentape[langgraph]"

# With LlamaIndex support
pip install "agentape[llamaindex]"

# All integrations
pip install "agentape[all]"
```

Alternatively, using `uv`:

```bash
uv add agentape
uv add "agentape[langgraph]"
```

---

## Quick Start

### 1. Write a test

```python
# tests/test_agent.py
import pytest
from agentape import AgentTestSession

@pytest.mark.asyncio
async def test_weather_query():
    async with AgentTestSession("weather_query") as session:
        result = await your_agent.run(
            "What's the weather in London?",
            base_url=session.base_url,  # Point agent at local proxy
            api_key=session.api_key,
        )
        
        session.assert_tool_called("get_weather")
        session.assert_tool_called_with("get_weather", city="London")
        session.assert_cost_under(0.05)
        session.assert_finished_cleanly()
```

### 2. Record (once)

```bash
OPENAI_API_KEY=sk-... pytest tests/test_agent.py
```

This creates `tests/cassettes/weather_query.json` containing the recorded interaction.

### 3. Replay (forever)

```bash
pytest tests/test_agent.py
```

Subsequent runs replay from the cassette. No API key needed, no network calls, no cost.

---

## Core Concepts

### Cassettes

A cassette is a JSON file containing recorded LLM request/response pairs:

```json
{
  "version": 1,
  "entries": [
    {
      "request_hash": "a3f8b2c1d4e5f6a7",
      "request": {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}]
      },
      "response": {
        "choices": [{"message": {"content": "Hi there!"}}]
      },
      "prompt_tokens": 8,
      "completion_tokens": 4,
      "model": "gpt-4o-mini"
    }
  ]
}
```

**Cassettes should be committed to version control.** They serve as:

- Regression tests for agent behavior
- Documentation of expected interactions
- Diff targets for code review

### Modes

agentape operates in three modes:

| Mode | Behavior | When to Use |
|------|----------|-------------|
| `auto` | Replay if cassette exists, otherwise record | Default for local development |
| `record` | Always hit real API and save to cassette | Force re-recording after changes |
| `replay` | Fail if cassette missing | CI/CD to prevent accidental API calls |

Mode selection (in order of precedence):

1. `mode` parameter in `AgentTestSession()`
2. `AGENTTEST_RECORD=1` environment variable (forces record)
3. Cassette existence (auto mode)

### Proxy Architecture

agentape runs a local HTTP proxy that intercepts OpenAI-compatible API calls:

```
Your Agent → http://127.0.0.1:<port>/chat/completions → Proxy
                                                          ↓
                              Record Mode: Forward to real API, save response
                              Replay Mode: Return saved response from cassette
```

The proxy is OpenAI-compatible and works with any client library that accepts a custom `base_url`.

---

## Usage Patterns

### Direct AgentTestSession Usage

For maximum control, instantiate `AgentTestSession` directly:

```python
from pathlib import Path
from agentape import AgentTestSession

CASSETTE_DIR = Path(__file__).parent / "cassettes"

@pytest.mark.asyncio
async def test_my_agent():
    async with AgentTestSession(
        name="my_agent_test",
        cassette_dir=CASSETTE_DIR,
        mode="auto",
    ) as session:
        result = await run_agent(
            base_url=session.base_url,
            api_key=session.api_key,
        )
        
        session.assert_tool_called("search")
```

**Note**: When using direct instantiation, pytest CLI flags (`--record`, `--replay`) are not available. Use the `mode` parameter or `AGENTTEST_RECORD` environment variable instead.

### Pytest Fixture Usage

The `agent_session` fixture provides zero-config integration:

```python
@pytest.mark.asyncio
async def test_my_agent(agent_session):
    async with agent_session:
        result = await run_agent(
            base_url=agent_session.base_url,
            api_key=agent_session.api_key,
        )
        agent_session.assert_tool_called("search")
```

Benefits:

- Automatic cassette naming (based on test function name)
- Cassette directory auto-discovered (sibling to test file)
- Supports `--record` and `--replay` CLI flags
- Prints trace summary on test failure

**CLI flags** (fixture only):

```bash
pytest tests/ --record    # Force re-record all tests
pytest tests/ --replay    # Fail if any cassette missing (CI mode)
pytest tests/             # Auto mode (default)
```

### Workflow Example

```bash
# 1. Initial development - record cassettes
OPENAI_API_KEY=sk-... pytest tests/

# 2. Commit cassettes
git add tests/cassettes/
git commit -m "Add agent behavior tests"

# 3. Ongoing development - free replays
pytest tests/

# 4. After changing prompts/tools - re-record specific test
AGENTTEST_RECORD=1 pytest tests/test_agent.py::test_weather_query

# 5. CI pipeline - strict replay mode
pytest tests/ --replay
```

---

## API Reference

### AgentTestSession

```python
class AgentTestSession:
    def __init__(
        self,
        name: str,
        cassette_dir: Path | str | None = None,
        mode: str = "auto",
        real_base_url: str = "",
        real_api_key: str = "",
        port: int = 0,
        debug: bool = False,
    )
```

**Parameters:**

- `name`: Cassette filename (without `.json` extension)
- `cassette_dir`: Directory for cassette storage (default: `./cassettes/`)
- `mode`: `"auto"`, `"record"`, or `"replay"`
- `real_base_url`: Upstream API URL (default: `https://api.openai.com`, or `OPENAI_BASE_URL` env var)
- `real_api_key`: Upstream API key (default: `OPENAI_API_KEY` env var)
- `port`: Proxy port (default: `0` = random free port)
- `debug`: Enable request/response logging

**Properties:**

- `base_url`: Proxy URL to pass to your agent (e.g., `"http://127.0.0.1:61234"`)
- `api_key`: Static proxy key (always `"agenttest-proxy-key"`)
- `mode`: Current operating mode
- `calls`: List of recorded LLM calls for custom assertions

**Context Manager:**

```python
async with AgentTestSession("test_name") as session:
    # Session is active, proxy is running
    await run_agent(base_url=session.base_url, api_key=session.api_key)
    # Assertions
# Cassette saved (if recording), proxy stopped
```

---

## Assertions

All assertions operate on the recorded trace of LLM interactions.

### LLM Call Assertions

```python
session.assert_llm_call_count(n: int)
# Exact number of LLM API calls

session.assert_llm_calls_at_most(n: int)
# No more than n calls (efficiency check)

session.assert_llm_calls_at_least(n: int)
# At least n calls
```

### Tool Call Assertions

```python
session.assert_tool_called(name: str)
# Tool was invoked at least once

session.assert_tool_not_called(name: str)
# Tool was never invoked

session.assert_tool_call_count(name: str, n: int)
# Tool called exactly n times

session.assert_tool_called_with(name: str, **kwargs)
# Tool called with specific arguments
# Example: session.assert_tool_called_with("search", query="weather")

session.assert_tool_called_before(first: str, second: str)
# Ordering constraint (subsequence, not strict adjacency)

session.assert_tool_not_called_after(tool: str, after: str)
# Tool never appears after another tool in call sequence
```

### Cost and Token Assertions

```python
session.assert_cost_under(max_usd: float)
# Total estimated cost in USD

session.assert_tokens_under(max_tokens: int)
# Total prompt + completion tokens
```

**Cost Estimation:**

Uses hardcoded pricing for common models (GPT-4o, GPT-4o-mini, Claude). For accurate cost tracking, verify pricing matches your provider.

### Response Content Assertions

```python
session.assert_final_response_contains(substring: str)
# Final LLM response contains substring (case-insensitive)

session.assert_final_response_not_contains(substring: str)
# Final response does not contain substring

session.assert_finished_cleanly()
# Last message has finish_reason="stop" (not "length" or "tool_calls")
```

### Debug Utilities

```python
session.print_summary()
# Print human-readable summary:
# - LLM call count
# - Tools called (in order)
# - Total tokens/cost
# - Final response preview
```

---

## Framework Integration

### LangGraph

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

async def test_langgraph_agent(agent_session):
    async with agent_session:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=agent_session.base_url,  # Only change needed
            api_key=agent_session.api_key,
        )
        
        graph = create_react_agent(llm, tools=[search_tool, calculator_tool])
        result = await graph.ainvoke({"messages": [("human", "What's 15 * 7?")]})
        
        agent_session.assert_tool_called("calculator")
        agent_session.assert_cost_under(0.02)
```

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

async def test_llamaindex_agent(agent_session):
    async with agent_session:
        llm = OpenAI(
            model="gpt-4o-mini",
            api_base=agent_session.base_url,
            api_key=agent_session.api_key,
        )
        
        agent = ReActAgent.from_tools([search_tool], llm=llm)
        response = await agent.achat("Search for recent AI news")
        
        agent_session.assert_tool_called("search")
```

### CrewAI, AutoGen, Raw OpenAI SDK

Any framework that accepts a custom `base_url` is compatible:

```python
from openai import AsyncOpenAI

async def test_raw_openai(agent_session):
    async with agent_session:
        client = AsyncOpenAI(
            base_url=agent_session.base_url,
            api_key=agent_session.api_key,
        )
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert "hello" in response.choices[0].message.content.lower()
```

---

## Provider Configuration

### Azure OpenAI

```bash
OPENAI_BASE_URL=https://my-resource.openai.azure.com/openai \
OPENAI_API_KEY=my-azure-key \
pytest tests/ --record
```

**Note**: Azure base URLs should include `/openai` but not `/v1`. The proxy handles path normalization.

### Anthropic via LiteLLM

Run a LiteLLM proxy locally:

```bash
# Terminal 1: Start LiteLLM proxy
litellm --model anthropic/claude-3-5-sonnet-20241022

# Terminal 2: Record tests
OPENAI_BASE_URL=http://localhost:4000 \
OPENAI_API_KEY=my-litellm-key \
pytest tests/ --record
```

### Local Ollama

```bash
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_API_KEY=ollama \
pytest tests/ --record
```

### Custom Providers (OpenRouter, Together, etc.)

Any OpenAI-compatible endpoint works:

```bash
# OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_API_KEY=sk-or-v1-... \
pytest tests/ --record

# Together AI
OPENAI_BASE_URL=https://api.together.xyz/v1 \
OPENAI_API_KEY=... \
pytest tests/ --record
```

**URL Normalization:**

The proxy strips trailing `/v1` from `OPENAI_BASE_URL` and adds it back when forwarding requests. This handles providers that include `/v1` in their base URL (OpenRouter) and those that don't (api.openai.com).

---

## Advanced Usage

### Custom Assertions

Access the raw trace for complex assertions:

```python
async with AgentTestSession("test") as session:
    await run_agent(base_url=session.base_url, api_key=session.api_key)
    
    # Raw access
    calls = session.calls
    
    # Custom assertion: no consecutive retries
    tool_names = [tc["name"] for call in calls for tc in call["tool_calls"]]
    for i in range(len(tool_names) - 1):
        assert tool_names[i] != tool_names[i+1], "Consecutive retry detected"
```

### Mock Tools

**Important**: agentape records what the LLM _requests_, not what your tools return. Tool execution happens in your application code, not in the proxy.

For testing tool execution separately, use standard mocking:

```python
from unittest.mock import patch

async def test_tool_error_handling(agent_session):
    async with agent_session:
        with patch("my_agent.search_tool", side_effect=Exception("API down")):
            result = await run_agent(
                base_url=agent_session.base_url,
                api_key=agent_session.api_key,
            )
            # Assert agent handles the error gracefully
```

### Parameterized Tests

```python
@pytest.mark.parametrize("city,expected_temp", [
    ("london", "13"),
    ("paris", "17"),
    ("tokyo", "26"),
])
async def test_weather_cities(agent_session, city, expected_temp):
    async with agent_session:
        result = await run_agent(
            f"What's the weather in {city}?",
            base_url=agent_session.base_url,
            api_key=agent_session.api_key,
        )
        
        assert expected_temp in result
        agent_session.assert_tool_called_with("get_weather", city=city)
```

Each parameter combination gets its own cassette: `test_weather_cities_london.json`, `test_weather_cities_paris.json`, etc.

### Debugging Failed Tests

Enable debug logging:

```python
async with AgentTestSession("test", debug=True) as session:
    # Logs all HTTP requests/responses
    await run_agent(base_url=session.base_url, api_key=session.api_key)
```

Or use pytest's built-in logging:

```bash
pytest tests/ -v --log-cli-level=DEBUG
```

---

## Troubleshooting

### Cassette Miss Error

```
CassetteMissError: Cassette miss — hash=a3f8b2c1
Last message role: user
Content: What's the weather in Paris?

Re-record this test:
    AGENTTEST_RECORD=1 pytest <test_file> --record
```

**Cause**: The LLM received messages that don't match any cassette entry.

**Solutions:**

1. Re-record the test (prompts or tools changed)
2. Check for nondeterministic inputs (timestamps, random IDs)
3. Verify you're in the correct mode (`--replay` in CI fails on missing cassettes)

### URL Normalization Issues

If you see connection errors like:

```
Cannot connect to upstream https://api.openai.com/v1/v1
```

**Cause**: Double `/v1` in the URL path.

**Solution**: Set `OPENAI_BASE_URL` without the `/v1` suffix:

```bash
# Correct
OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Also correct (proxy strips it)
OPENAI_BASE_URL=https://openrouter.ai/api
```

### Port Conflicts

If you see:

```
OSError: [Errno 48] Address already in use
```

**Cause**: Another process is using the proxy port.

**Solution**: Let the proxy pick a random port (default) or specify a different port:

```python
async with AgentTestSession("test", port=9999) as session:
    # ...
```

### Case Sensitivity in Assertions

```
AssertionError: Tool 'get_weather' was called, but never with {'city': 'tokyo'}.
Actual calls: [{'city': 'Tokyo'}]
```

**Cause**: LLM capitalized the argument.

**Solution**: Match the actual capitalization or use case-insensitive comparison:

```python
# Option 1: Match actual
session.assert_tool_called_with("get_weather", city="Tokyo")

# Option 2: Case-insensitive (coming in v0.2.0)
session.assert_tool_called_with("get_weather", city="tokyo", case_sensitive=False)
```

### CI/CD Integration

**Recommended CI configuration:**

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[langgraph]"
      - run: pytest tests/ --replay -v
        # --replay ensures tests fail if cassettes are missing
        # No OPENAI_API_KEY needed - cassettes contain everything
```

---

## Development

### Project Structure

```
agentape/
├── src/agentape/
│   ├── __init__.py
│   ├── session.py       # AgentTestSession API
│   ├── proxy.py         # HTTP proxy implementation
│   ├── cassette.py      # Cassette format and matching
│   ├── assertions.py    # Behavioral assertions
│   ├── mock_tools.py    # Tool mocking (if needed)
│   └── plugin.py        # Pytest integration
├── tests/
│   └── cassettes/
└── examples/
    ├── test_langgraph_agent.py
    └── cassettes/
```

### Running Tests Locally

```bash
# Install development dependencies
pip install -e ".[dev,all]"

# Run unit tests
pytest tests/

# Run examples (requires API key for initial recording)
OPENAI_API_KEY=sk-... pytest examples/ --record

# Subsequent runs (free)
pytest examples/
```

### Contributing

**Before submitting a PR:**

1. Add tests for new features
2. Update README if API changes
3. Run linters: `ruff check . && mypy src/`
4. Ensure all tests pass: `pytest tests/ examples/`

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Real API key (record mode only) | - |
| `OPENAI_BASE_URL` | Upstream API URL | `https://api.openai.com` |
| `AGENTTEST_RECORD` | Force record mode (`1` = record) | - |
| `AGENTTEST_CASSETTE_DIR` | Cassette directory override | `./cassettes` |

---

## FAQ

**Q: Does this work with streaming responses?**

A: Not yet. Currently, all responses are recorded as non-streaming. Streaming support is planned for v0.2.0.

**Q: Can I edit cassettes manually?**

A: Yes. Cassettes are plain JSON. You can edit responses to test error handling or modify tool call arguments.

**Q: What happens if I change my prompts?**

A: The cassette will miss (hash mismatch) and you'll need to re-record. This is intentional — it ensures tests fail when behavior changes.

**Q: Can I share cassettes between tests?**

A: No. Each test should have its own cassette. This ensures tests are independent and failures are isolated.

**Q: Does this work with function calling / tool use?**

A: Yes. Tool calls are recorded and can be asserted on. See `assert_tool_called()` and related assertions.

**Q: What about multimodal inputs (images, audio)?**

A: Not currently supported. Text-only for now.

---

## License

MIT License - see LICENSE file for details.

---

## Changelog

### v0.1.0 (2026-02-19)

- Initial release
- OpenAI-compatible proxy with record/replay
- Pytest integration with fixtures and CLI flags
- Behavioral assertions for tools, cost, tokens
- LangGraph and LlamaIndex examples
- Multi-provider support (Azure, Ollama, LiteLLM)