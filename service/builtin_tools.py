from __future__ import annotations

from typing import Any

from .schemas import ToolDefinition


_CITY_WEATHER: dict[str, dict[str, Any]] = {
    "beijing": {"city": "Beijing", "weather": "sunny", "temperature_c": 22},
    "shanghai": {"city": "Shanghai", "weather": "cloudy", "temperature_c": 24},
    "hangzhou": {"city": "Hangzhou", "weather": "rainy", "temperature_c": 20},
    "shenzhen": {"city": "Shenzhen", "weather": "humid", "temperature_c": 28},
}


def get_whether_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        type="function",
        function={
            "name": "get_whether",
            "description": "Get predefined weather data by city name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, for example Beijing or Hangzhou.",
                    }
                },
                "required": ["city"],
            },
        },
    )


def execute_builtin_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "get_whether":
        raise ValueError(f"Unsupported builtin tool '{name}'")

    city = str(arguments.get("city", "")).strip()
    if not city:
        raise ValueError("city is required")

    city_key = city.lower()
    payload = _CITY_WEATHER.get(city_key)
    if payload is None:
        return {
            "city": city,
            "weather": "unknown",
            "temperature_c": None,
        }
    return payload
