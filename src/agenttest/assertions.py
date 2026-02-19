from typing import Any


class TraceAssertions:
    """Assert on the recorded trace of LLM calls from one test session."""

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = calls

    def llm_calls(self) -> int:
        return len(self._calls)

    def assert_llm_call_count(self, expected: int) -> None:
        actual = self.llm_calls()
        assert actual == expected, f"Expected {expected} LLM calls, got {actual}."

    def assert_llm_calls_at_most(self, max_calls: int) -> None:
        actual = self.llm_calls()
        assert (
            actual <= max_calls
        ), f"Agent made {actual} LLM calls — exceeds max of {max_calls}."

    def assert_llm_calls_at_least(self, min_calls: int) -> None:
        actual = self.llm_calls()
        assert (
            actual >= min_calls
        ), f"Agent made {actual} LLM calls — below min of {min_calls}."

    def all_tool_calls(self) -> list[dict[str, Any]]:
        result = []
        for call in self._calls:
            result.extend(call.get("tool_calls", []))
        return result

    def tool_names_in_order(self) -> list[str]:
        return [tc["name"] for tc in self.all_tool_calls()]

    def assert_tool_called(self, tool_name: str) -> None:
        names = self.tool_names_in_order()
        assert tool_name in names, (
            f"Tool '{tool_name}' was never called.\n" f"Tools called: {names}"
        )

    def assert_tool_not_called(self, tool_name: str) -> None:
        names = self.tool_names_in_order()
        assert tool_name not in names, (
            f"Tool '{tool_name}' was called but should not have been.\n"
            f"Arguments: {[tc['arguments'] for tc in self.all_tool_calls() if tc['name'] == tool_name]}"
        )

    def assert_tool_called_before(self, first: str, second: str) -> None:
        order = self.tool_names_in_order()
        try:
            i_first = order.index(first)
        except ValueError:
            raise AssertionError(
                f"Tool '{first}' was never called.\nTools called: {order}"
            )
        try:
            i_second = order.index(second)
        except ValueError:
            raise AssertionError(
                f"Tool '{second}' was never called.\nTools called: {order}"
            )
        assert i_first < i_second, (
            f"Expected '{first}' before '{second}', "
            f"but first={i_first}, second={i_second}.\nFull order: {order}"
        )

    def assert_tool_call_count(self, tool_name: str, expected: int) -> None:
        actual = sum(1 for n in self.tool_names_in_order() if n == tool_name)
        assert (
            actual == expected
        ), f"Tool '{tool_name}' expected {expected} calls, got {actual}."

    def assert_tool_called_with(
        self, tool_name: str, case_sensitive: bool = True, **kwargs: Any
    ) -> None:
        """Assert that at least one call to tool_name had these arguments.

        Args:
            tool_name: Name of the tool
            case_sensitive: If False, string arguments are compared case-insensitively
            **kwargs: Expected argument values
        """
        calls = [tc for tc in self.all_tool_calls() if tc["name"] == tool_name]
        assert calls, f"Tool '{tool_name}' was never called."

        for call in calls:
            match = True
            for k, v in kwargs.items():
                actual = call["arguments"].get(k)
                if (
                    not case_sensitive
                    and isinstance(v, str)
                    and isinstance(actual, str)
                ):
                    # Case-insensitive comparison for strings
                    if actual.lower() != v.lower():
                        match = False
                        break
                else:
                    # Exact comparison for everything else
                    if actual != v:
                        match = False
                        break

            if match:
                return

        raise AssertionError(
            f"Tool '{tool_name}' was called, but never with {kwargs}.\n"
            f"Actual calls: {[c['arguments'] for c in calls]}"
        )

    def assert_tool_never_called_after(self, tool: str, after: str) -> None:
        """Assert that tool was never called after after_tool appeared."""
        order = self.tool_names_in_order()
        if after not in order:
            return
        after_idx = order.index(after)
        post = order[after_idx + 1 :]
        assert tool not in post, (
            f"Tool '{tool}' was called after '{after}' — this should not happen.\n"
            f"Full order: {order}"
        )

    def total_cost(self) -> float:
        return sum(c.get("cost_usd", 0.0) for c in self._calls)

    def total_tokens(self) -> int:
        return sum(
            c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)
            for c in self._calls
        )

    def assert_cost_under(self, max_usd: float) -> None:
        actual = self.total_cost()
        assert (
            actual <= max_usd
        ), f"Agent cost ${actual:.5f} exceeds limit of ${max_usd:.5f}."

    def assert_tokens_under(self, max_tokens: int) -> None:
        actual = self.total_tokens()
        assert (
            actual <= max_tokens
        ), f"Agent used {actual} tokens, exceeds limit of {max_tokens}."

    def final_response(self) -> str | None:
        if not self._calls:
            return None
        return self._calls[-1].get("response_content")

    def assert_final_response_contains(self, substring: str) -> None:
        resp = self.final_response()
        assert resp is not None, "Agent produced no final text response."
        assert substring.lower() in resp.lower(), (
            f"Expected '{substring}' in final response.\n" f"Got: {resp[:300]}"
        )

    def assert_final_response_not_contains(self, substring: str) -> None:
        resp = self.final_response() or ""
        assert substring.lower() not in resp.lower(), (
            f"Final response contains '{substring}' but should not.\n"
            f"Got: {resp[:300]}"
        )

    def assert_finished_cleanly(self) -> None:
        """Assert the agent finished with stop, not tool_calls or length."""
        if not self._calls:
            raise AssertionError("No LLM calls recorded.")
        last = self._calls[-1]
        reason = last.get("finish_reason")
        assert (
            reason == "stop"
        ), f"Agent did not finish cleanly — finish_reason={reason!r}"

    def summary(self) -> str:
        lines = [
            f"LLM calls      : {self.llm_calls()}",
            f"Tools called   : {self.tool_names_in_order()}",
            f"Total tokens   : {self.total_tokens():,}",
            f"Total cost     : ${self.total_cost():.5f}",
            f"Final response : {str(self.final_response())[:120]}",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print("\n" + self.summary())
