from typing import Any, Callable


class ToolCallRecord:
    """One invocation of a tool."""

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = arguments


class MockTool:
    """A registered mock for one tool function."""

    def __init__(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Callable | Exception | None = None,
    ) -> None:
        self.name = name
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls: list[ToolCallRecord] = []

    def record(self, arguments: dict[str, Any]) -> None:
        self.calls.append(ToolCallRecord(self.name, arguments))

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def called(self) -> bool:
        return len(self.calls) > 0

    def assert_called(self) -> None:
        assert self.called, f"Tool '{self.name}' was never called."

    def assert_called_once(self) -> None:
        assert (
            self.call_count == 1
        ), f"Tool '{self.name}' expected 1 call, got {self.call_count}."

    def assert_called_with(self, **kwargs: Any) -> None:
        assert self.called, f"Tool '{self.name}' was never called."
        last = self.calls[-1].arguments
        for k, v in kwargs.items():
            assert last.get(k) == v, (
                f"Tool '{self.name}' argument '{k}': "
                f"expected {v!r}, got {last.get(k)!r}"
            )

    def assert_not_called(self) -> None:
        assert (
            not self.called
        ), f"Tool '{self.name}' was called {self.call_count} time(s) but expected 0."


class MockToolRegistry:
    """Registry of mock tools for one test session."""

    def __init__(self) -> None:
        self._mocks: dict[str, MockTool] = {}
        # Global ordered list of all tool calls (any tool, any order)
        self._all_calls: list[ToolCallRecord] = []

    def register(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Callable | Exception | None = None,
    ) -> MockTool:
        mock = MockTool(name, return_value=return_value, side_effect=side_effect)
        self._mocks[name] = mock
        return mock

    def has(self, name: str) -> bool:
        return name in self._mocks

    def get(self, name: str) -> MockTool | None:
        return self._mocks.get(name)

    def record_call(self, name: str, arguments: dict[str, Any]) -> None:
        record = ToolCallRecord(name, arguments)
        self._mocks[name].calls.append(record)
        self._all_calls.append(record)

    @property
    def all_calls(self) -> list[ToolCallRecord]:
        return list(self._all_calls)

    def called_order(self) -> list[str]:
        """Return tool names in call order."""
        return [c.name for c in self._all_calls]

    def assert_call_order(self, *names: str) -> None:
        """Assert that tools were called in this order (subsequence, not strict)."""
        actual = self.called_order()
        it = iter(actual)
        for name in names:
            assert any(n == name for n in it), (
                f"Expected tool '{name}' in call order after previous tools.\n"
                f"Actual order: {actual}"
            )

    def assert_called_before(self, first: str, second: str) -> None:
        """Assert first tool was called before second tool."""
        order = self.called_order()
        try:
            i_first = order.index(first)
        except ValueError:
            raise AssertionError(f"Tool '{first}' was never called. Order: {order}")
        try:
            i_second = order.index(second)
        except ValueError:
            raise AssertionError(f"Tool '{second}' was never called. Order: {order}")
        assert i_first < i_second, (
            f"Expected '{first}' before '{second}', "
            f"but got order {i_first} vs {i_second}. Full order: {order}"
        )
