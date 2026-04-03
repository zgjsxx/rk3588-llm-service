from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path


RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3
RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4

RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("reserved", ctypes.c_uint8 * 112),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]


class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("payload",)
    _fields_ = [
        ("input_type", ctypes.c_int),
        ("payload", RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
    ]


CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Interactive RKLLM demo calling librkllmrt.so directly."
    )
    parser.add_argument(
        "--model-path",
        default=str(root / "models" / "DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"),
        help="Path to the .rkllm model file.",
    )
    parser.add_argument(
        "--runtime-lib",
        default=str(root / "third_party" / "rkllm_api" / "aarch64" / "librkllmrt.so"),
        help="Path to librkllmrt.so.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-context-len", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repeat-penalty", type=float, default=1.1)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--skip-special-token",
        action="store_true",
        help="Match llm_demo by skipping special tokens.",
    )
    parser.add_argument(
        "--no-skip-special-token",
        action="store_true",
        help="Print special tokens for debugging.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="<｜begin▁of▁sentence｜><｜User｜>",
        help="Prompt prefix.",
    )
    parser.add_argument(
        "--prompt-postfix",
        default="<｜Assistant｜>",
        help="Prompt postfix.",
    )
    return parser.parse_args()


def load_runtime(runtime_lib: str) -> ctypes.CDLL:
    runtime_path = Path(runtime_lib).resolve()
    if not runtime_path.exists():
        raise FileNotFoundError(f"runtime library not found: {runtime_path}")

    lib = ctypes.CDLL(str(runtime_path))
    lib.rkllm_createDefaultParam.restype = RKLLMParam
    lib.rkllm_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(RKLLMParam),
        CALLBACK,
    ]
    lib.rkllm_init.restype = ctypes.c_int
    lib.rkllm_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(RKLLMInput),
        ctypes.POINTER(RKLLMInferParam),
        ctypes.c_void_p,
    ]
    lib.rkllm_run.restype = ctypes.c_int
    lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
    lib.rkllm_destroy.restype = ctypes.c_int
    return lib


def main() -> int:
    args = parse_args()
    runtime = load_runtime(args.runtime_lib)

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"model file not found: {model_path}", file=sys.stderr)
        return 1

    handle = ctypes.c_void_p()

    @CALLBACK
    def callback(result_ptr: ctypes.POINTER(RKLLMResult), _: ctypes.c_void_p, state: int) -> None:
        if state == RKLLM_RUN_FINISH:
            print()
            return
        if state == RKLLM_RUN_ERROR:
            print("\\run error")
            return
        if state in (RKLLM_RUN_WAITING, RKLLM_RUN_GET_LAST_HIDDEN_LAYER):
            return
        if not result_ptr:
            return
        result = result_ptr.contents
        if result.text:
            chunk = ctypes.string_at(result.text)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    param = runtime.rkllm_createDefaultParam()
    param.model_path = str(model_path).encode("utf-8")
    param.top_k = args.top_k
    param.top_p = args.top_p
    param.temperature = args.temperature
    param.repeat_penalty = args.repeat_penalty
    param.frequency_penalty = args.frequency_penalty
    param.presence_penalty = args.presence_penalty
    param.max_new_tokens = args.max_new_tokens
    param.max_context_len = args.max_context_len
    param.skip_special_token = bool(args.skip_special_token or not args.no_skip_special_token)
    param.extend_param.base_domain_id = 0

    ret = runtime.rkllm_init(ctypes.byref(handle), ctypes.byref(param), callback)
    if ret != 0:
        print("rkllm init failed", file=sys.stderr)
        return ret

    print("python rkllm direct demo")
    print(f"model_path: {model_path}")
    print(f"runtime_lib: {Path(args.runtime_lib).resolve()}")
    print("type 'exit' to quit")

    infer_param = RKLLMInferParam()
    infer_param.mode = RKLLM_INFER_GENERATE
    infer_param.lora_params = None
    infer_param.prompt_cache_params = None

    try:
        while True:
            try:
                user_input = input("\nuser: ")
            except EOFError:
                print()
                break

            if not user_input.strip():
                continue
            if user_input.strip().lower() == "exit":
                break

            prompt = f"{args.prompt_prefix}{user_input}{args.prompt_postfix}"
            prompt_bytes = prompt.encode("utf-8")

            rk_input = RKLLMInput()
            rk_input.input_type = RKLLM_INPUT_PROMPT
            rk_input.prompt_input = prompt_bytes

            print("robot: ", end="", flush=True)
            ret = runtime.rkllm_run(handle, ctypes.byref(rk_input), ctypes.byref(infer_param), None)
            if ret != 0:
                print(f"\nrkllm_run failed: {ret}", file=sys.stderr)
                break
    finally:
        if handle:
            runtime.rkllm_destroy(handle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
