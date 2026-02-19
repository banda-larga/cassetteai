from collections.abc import Generator
from pathlib import Path
from typing import AsyncGenerator

import pytest

from agenttest.session import AgentTestSession


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agenttest")
    group.addoption(
        "--record",
        action="store_true",
        default=False,
        help="Re-record all cassettes (hits real LLM APIs).",
    )
    group.addoption(
        "--replay",
        action="store_true",
        default=False,
        help="Force replay mode — fail if cassette missing.",
    )


@pytest.fixture
async def agent_session(
    request: pytest.FixtureRequest,
) -> AsyncGenerator[AgentTestSession, None]:
    """An AgentTestSession auto-named after the test function.

    Usage:

        async def test_my_agent(agent_session):
            async with agent_session:
                await my_agent.run(base_url=agent_session.base_url, ...)
                agent_session.assert_tool_called("search")
    """
    record = request.config.getoption("--record", default=False)
    force_replay = request.config.getoption("--replay", default=False)

    if record:
        mode = "record"
    elif force_replay:
        mode = "replay"
    else:
        mode = "auto"

    # Find cassette dir relative to the test file
    test_file = Path(request.fspath)
    cassette_dir = test_file.parent / "cassettes"
    name = request.node.name.replace("[", "_").replace("]", "_")

    session = AgentTestSession(
        name=name,
        cassette_dir=cassette_dir,
        mode=mode,
    )

    yield session

    # Print summary on test failure
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        print("\n--- AgentTest trace summary ---")
        session.print_summary()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo,  # noqa: ARG001
) -> Generator[None, None, None]:
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
