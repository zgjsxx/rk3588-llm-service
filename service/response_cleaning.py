from __future__ import annotations

import re


THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_CAPTURE_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|>]+?\|>")


def split_response_text(text: str) -> tuple[str, str]:
    match = THINK_CAPTURE_PATTERN.search(text)
    if match:
        think_text = match.group(1).strip()
        answer_text = THINK_CAPTURE_PATTERN.sub("", text).strip()
        answer_text = SPECIAL_TOKEN_PATTERN.sub("", answer_text).strip()
        return think_text, answer_text

    answer_text = text.replace("<think>", "").replace("</think>", "").strip()
    answer_text = SPECIAL_TOKEN_PATTERN.sub("", answer_text).strip()
    return "", answer_text


def clean_response_text(text: str) -> str:
    _, answer_text = split_response_text(text)
    return answer_text
