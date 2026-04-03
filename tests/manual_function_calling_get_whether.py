from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual function-calling test client for the get_whether tool."
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="Base URL of the local FastAPI service.",
    )
    parser.add_argument(
        "--model",
        default="rkllm-local",
        help="Model name sent to /v1/chat/completions.",
    )
    parser.add_argument(
        "--city",
        default="Hangzhou",
        help="City argument to send to the get_whether tool.",
    )
    parser.add_argument(
        "--tool-choice",
        choices=["auto", "none", "force"],
        default="auto",
        help="Tool choice policy. 'force' maps to a forced get_whether call.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Print the expected OpenAI-style tool call shape before sending the request.",
    )
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": f"manual-fc-{uuid.uuid4().hex}",
        "model": args.model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": f"What is the weather in {args.city}? If needed, call the get_whether function.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
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
            }
        ],
    }
    if args.tool_choice == "force":
        payload["tool_choice"] = {"type": "function", "function": {"name": "get_whether"}}
    else:
        payload["tool_choice"] = args.tool_choice
    return payload


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    url = f"{args.api_base.rstrip('/')}/v1/chat/completions"

    if args.show_expected:
        expected = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_whether",
                        "arguments": {"city": args.city},
                    },
                }
            ]
        }
        print("expected model output shape:")
        print(json.dumps(expected, ensure_ascii=False, indent=2))
        print()

    print(f"POST {url}")
    print("request:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    response = httpx.post(url, json=payload, timeout=args.timeout)
    print(f"\nstatus: {response.status_code}")

    try:
        body = response.json()
    except Exception:
        print(response.text)
        return 1 if response.is_error else 0

    print("response:")
    print(json.dumps(body, ensure_ascii=False, indent=2))

    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        print("\nparsed tool call:")
        print(json.dumps(tool_calls[0], ensure_ascii=False, indent=2))
    else:
        print("\nno tool call returned")

    return 1 if response.is_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
