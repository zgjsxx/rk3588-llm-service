from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_ALLOW_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.model",
    "*.tiktoken",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download tokenizer files needed by the RKLLM service.")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repo to download tokenizer files from.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"),
        help="Local directory to store tokenizer files.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit revision.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for gated/private repos.",
    )
    parser.add_argument(
        "--local-dir-use-symlinks",
        action="store_true",
        help="Allow symlinks in the output directory when supported by the platform.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        token=args.token,
        local_dir=str(output_dir),
        local_dir_use_symlinks=args.local_dir_use_symlinks,
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
    )

    print(f"Tokenizer files downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
