from service.builtin_tools import execute_builtin_tool, get_whether_tool_definition


def test_get_whether_tool_definition():
    tool = get_whether_tool_definition()
    assert tool.function.name == "get_whether"
    assert tool.function.parameters["required"] == ["city"]


def test_execute_builtin_tool_known_city():
    result = execute_builtin_tool("get_whether", {"city": "Hangzhou"})
    assert result == {
        "city": "Hangzhou",
        "weather": "rainy",
        "temperature_c": 20,
    }


def test_execute_builtin_tool_unknown_city():
    result = execute_builtin_tool("get_whether", {"city": "Suzhou"})
    assert result == {
        "city": "Suzhou",
        "weather": "unknown",
        "temperature_c": None,
    }
