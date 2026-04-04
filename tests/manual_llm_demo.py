from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.prompting import build_prompt, load_tokenizer
from service.rkllm_bridge import EngineConfig, RkllmEngine, load_engine_config_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive RKLLM demo using the Python bridge library."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the .rkllm model file. Defaults to RKLLM_MODEL_PATH or the repo default.",
    )
    parser.add_argument(
        "--bridge-lib",
        default=None,
        help="Path to the bridge shared library. Defaults to RKLLM_BRIDGE_LIB or the repo default.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Default max_new_tokens used when initializing the engine.",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Default max_context_len used when initializing the engine.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sampling top_k.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Sampling top_p.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help="Sampling repeat penalty.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Sampling frequency penalty.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Sampling presence penalty.",
    )
    parser.add_argument(
        "--skip-special-token",
        action="store_true",
        help="Skip special tokens during generation.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EngineConfig:
    config = load_engine_config_from_env()
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.bridge_lib is not None:
        config.bridge_lib_path = args.bridge_lib
    if args.max_new_tokens is not None:
        config.default_max_new_tokens = args.max_new_tokens
    if args.max_context_len is not None:
        config.max_context_len = args.max_context_len
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.repeat_penalty is not None:
        config.repeat_penalty = args.repeat_penalty
    if args.frequency_penalty is not None:
        config.frequency_penalty = args.frequency_penalty
    if args.presence_penalty is not None:
        config.presence_penalty = args.presence_penalty
    if args.skip_special_token:
        config.skip_special_token = True
    return config


def main() -> int:
    args = parse_args()
    config = build_config(args)

    print("python llm demo")
    print(f"model_path: {config.model_path}")
    print(f"bridge_lib: {config.bridge_lib_path}")
    print("type 'exit' to quit")

    tokenizer = load_tokenizer(config.tokenizer_path)
    engine = RkllmEngine(config)
    try:
        while True:
            try:
                user_input = input("\nuser: ").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue
            if user_input.lower() == "exit":
                break

            prompt = build_prompt(
                [{"role": "user", "content": user_input}],
                tokenizer=tokenizer,
            )
            request_id = f"manual-{uuid.uuid4().hex}"

            print("robot: ", end="", flush=True)

            def on_token(token: str) -> None:
                print(token, end="", flush=True)

            engine.generate(prompt, on_token, request_id)
            print()
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
