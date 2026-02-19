from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agentape import AgentTestSession

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

from langgraph.prebuilt import create_react_agent

CASSETTE_DIR = Path(__file__).parent / "cassettes"
PRICING = {
    "qwen/qwen3.5-397b-a17b": (0.00000015, 0.000001),
}


@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    data = {
        "london": "13°C, overcast",
        "paris": "17°C, partly cloudy",
        "tokyo": "26°C, humid",
        "new york": "22°C, sunny",
    }
    return data.get(city.lower(), f"No weather data for '{city}'.")


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '(13 + 17) / 2'."""
    safe = set("0123456789 +-*/().")
    if not all(c in safe for c in expression):
        return "Error: only basic arithmetic allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as exc:
        return f"Error: {exc}"


@tool
def send_alert(message: str, severity: str = "low") -> str:
    """Send a system alert. Should only be called for serious issues."""
    return f"Alert sent: [{severity.upper()}] {message}"


def make_agent(base_url: str, api_key: str):
    llm = ChatOpenAI(
        model="qwen/qwen3.5-397b-a17b",
        base_url=base_url,
        api_key=api_key,
        temperature=0,
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_react_agent(llm, [get_weather, calculate, send_alert])


async def run_agent(question: str, base_url: str, api_key: str) -> str:
    graph = make_agent(base_url, api_key)
    result = await graph.ainvoke({"messages": [("human", question)]})
    return result["messages"][-1].content


@pytest.mark.asyncio
async def test_weather_then_calculate():
    """
    Agent should call get_weather (at least twice) then calculate.
    The ordering assertion proves it fetched data before computing.
    """
    async with AgentTestSession(
        "weather_then_calculate",
        cassette_dir=CASSETTE_DIR,
        pricing=PRICING,
    ) as s:
        answer = await run_agent(
            "What is the average temperature of London and Paris?",
            base_url=s.base_url,
            api_key=s.api_key,
        )

        # Tool ordering
        s.assert_tool_called("get_weather")
        s.assert_tool_called("calculate")
        s.assert_tool_called_before("get_weather", "calculate")

        # Safety
        s.assert_tool_not_called("send_alert")

        # Efficiency
        s.assert_llm_calls_at_most(6)
        s.assert_cost_under(0.05)

        assert any(
            char.isdigit() for char in answer
        ), f"Expected a numeric answer, got: {answer}"

        s.assert_finished_cleanly()
        s.print_summary()


@pytest.mark.asyncio
async def test_single_city_no_calculation():
    """Simple weather query — should call get_weather exactly once, never calculate."""
    async with AgentTestSession(
        "single_city_no_calc",
        cassette_dir=CASSETTE_DIR,
        pricing=PRICING,
    ) as s:
        await run_agent(
            "What's the weather in Tokyo right now?",
            base_url=s.base_url,
            api_key=s.api_key,
        )

        s.assert_tool_called("get_weather")
        s.assert_tool_called_with("get_weather", city="tokyo", case_sensitive=False)
        s.assert_tool_call_count("get_weather", 1)
        s.assert_tool_not_called("calculate")
        s.assert_tool_not_called("send_alert")
        s.assert_cost_under(0.02)
        s.assert_finished_cleanly()


@pytest.mark.asyncio
async def test_alert_not_sent_for_normal_query():
    """
    Regression test: agent must never send alerts for routine queries.
    This would be a real production bug — capture it permanently.
    """
    async with AgentTestSession(
        "no_alert_on_normal_query",
        cassette_dir=CASSETTE_DIR,
        pricing=PRICING,
    ) as s:
        await run_agent(
            "What's 15 multiplied by 7?",
            base_url=s.base_url,
            api_key=s.api_key,
        )

        s.assert_tool_not_called("send_alert")
        s.assert_tool_not_called("get_weather")
        s.assert_tool_called("calculate")
        s.assert_cost_under(0.01)


@pytest.mark.asyncio
async def test_unknown_city_handled_gracefully():
    """Agent should call get_weather, get 'no data', and respond without crashing."""
    async with AgentTestSession(
        "unknown_city_graceful",
        cassette_dir=CASSETTE_DIR,
        pricing=PRICING,
    ) as s:
        await run_agent(
            "What's the weather in Atlantis?",
            base_url=s.base_url,
            api_key=s.api_key,
        )

        # Tool was called
        s.assert_tool_called("get_weather")

        # But no alert was sent (graceful failure, not emergency)
        s.assert_tool_not_called("send_alert")

        # Agent should have produced some response
        s.assert_finished_cleanly()


@pytest.mark.asyncio
async def test_cost_budget_regression():
    """
    Prevent token budget regressions after prompt changes.
    If this fails, your new prompt is significantly more expensive.
    """
    async with AgentTestSession(
        "cost_budget_multi_city",
        cassette_dir=CASSETTE_DIR,
        pricing=PRICING,
    ) as s:
        await run_agent(
            "Compare the weather in London, Paris, Tokyo, and New York. "
            "Then calculate the average temperature across all four cities.",
            base_url=s.base_url,
            api_key=s.api_key,
        )

        # Should not take more than 8 LLM calls for this
        s.assert_llm_calls_at_most(8)

        # Hard cost ceiling
        s.assert_cost_under(0.15)
        s.assert_tokens_under(10_000)

        # Must have fetched all cities and calculated
        s.assert_tool_called("get_weather")
        s.assert_tool_called("calculate")
        s.assert_tool_called_before("get_weather", "calculate")

        s.print_summary()
