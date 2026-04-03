from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.builtin_tools import execute_builtin_tool, get_whether_tool_definition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual test for the builtin get_whether tool.")
    parser.add_argument("--city", default="Hangzhou", help="City name to query.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool = get_whether_tool_definition()
    result = execute_builtin_tool(tool.function.name, {"city": args.city})

    print("tool definition:")
    print(json.dumps(tool.model_dump(), ensure_ascii=False, indent=2))
    print("\nresult:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
