from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .trace import trace_event


class BridgeError(RuntimeError):
    pass


STREAM_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
EVENT_TOKEN = 0
EVENT_FINISH = 1
EVENT_ERROR = 2


class SamplingConfig(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("max_context_len", ctypes.c_int),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("skip_special_token", ctypes.c_int),
    ]


@dataclass
class EngineConfig:
    model_path: str
    bridge_lib_path: str
    default_max_new_tokens: int = 2048
    max_context_len: int = 2048
    top_k: int = 1
    top_p: float = 0.95
    temperature: float = 0.8
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    skip_special_token: bool = True
    model_name: str = "rkllm-local"
    prompt_prefix: str = "<｜begin▁of▁sentence｜><｜User｜>"
    prompt_postfix: str = "<｜Assistant｜>"


class RkllmBridgeLibrary:
    def __init__(self, lib_path: str) -> None:
        bridge_path = Path(lib_path).resolve()
        if not bridge_path.exists():
            raise BridgeError(f"Bridge library not found: {lib_path}")

        self._preload_runtime_dependency(bridge_path)
        self.lib = ctypes.CDLL(str(bridge_path))
        self.lib.rkllm_bridge_create.restype = ctypes.c_void_p
        self.lib.rkllm_bridge_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(SamplingConfig), ctypes.c_char_p, ctypes.c_int]
        self.lib.rkllm_bridge_init.restype = ctypes.c_int
        self.lib.rkllm_bridge_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, STREAM_CALLBACK, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.rkllm_bridge_generate.restype = ctypes.c_int
        self.lib.rkllm_bridge_cancel.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.rkllm_bridge_cancel.restype = ctypes.c_int
        self.lib.rkllm_bridge_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_bridge_destroy.restype = None

    @staticmethod
    def _preload_runtime_dependency(bridge_path: Path) -> None:
        candidate_paths = [
            bridge_path.parent / "librkllmrt.so",
            bridge_path.parent / "lib" / "librkllmrt.so",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                return

    def create(self) -> int:
        handle = self.lib.rkllm_bridge_create()
        if not handle:
            raise BridgeError("Failed to create RKLLM bridge handle")
        return handle


class RkllmEngine:
    def __init__(self, config: EngineConfig, library: Optional[RkllmBridgeLibrary] = None) -> None:
        self.config = config
        self._library = library or RkllmBridgeLibrary(config.bridge_lib_path)
        self._handle = self._library.create()
        self._callback_ref = None
        self._initialize(config)

    def _initialize(self, config: EngineConfig) -> None:
        error_buffer = ctypes.create_string_buffer(1024)
        sampling = SamplingConfig(
            max_new_tokens=config.default_max_new_tokens,
            max_context_len=config.max_context_len,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            repeat_penalty=config.repeat_penalty,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            skip_special_token=1 if config.skip_special_token else 0,
        )
        ret = self._library.lib.rkllm_bridge_init(
            self._handle,
            config.model_path.encode("utf-8"),
            ctypes.byref(sampling),
            error_buffer,
            len(error_buffer),
        )
        trace_event(
            "python_bridge.init",
            model_path=config.model_path,
            bridge_lib_path=config.bridge_lib_path,
            max_new_tokens=config.default_max_new_tokens,
            max_context_len=config.max_context_len,
            ret_code=ret,
            error_buffer=error_buffer.value.decode("utf-8", errors="ignore"),
        )
        if ret != 0:
            raise BridgeError(error_buffer.value.decode("utf-8") or "Failed to initialize RKLLM bridge")

    def generate(
        self,
        prompt: str,
        on_token: Callable[[str], None],
        request_id: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> str:
        del max_tokens, temperature, top_p, presence_penalty, frequency_penalty

        chunks: list[str] = []
        callback_errors: list[str] = []
        error_buffer = ctypes.create_string_buffer(1024)
        callback_started = False

        trace_event(
            "python_bridge.generate.start",
            request_id=request_id,
            prompt=prompt,
            prompt_length=len(prompt),
            model_path=self.config.model_path,
            bridge_lib_path=self.config.bridge_lib_path,
        )

        def _callback(text_ptr: bytes, event: int, error_code: int, _: int) -> None:
            nonlocal callback_started
            del error_code
            text = text_ptr.decode("utf-8") if text_ptr else ""
            try:
                if event == EVENT_TOKEN:
                    if not callback_started:
                        callback_started = True
                        trace_event(
                            "python_bridge.generate.first_token",
                            request_id=request_id,
                            token_length=len(text),
                        )
                    chunks.append(text)
                    on_token(text)
                elif event == EVENT_ERROR:
                    callback_errors.append(text or "RKLLM generation failed")
            except Exception as exc:  # pragma: no cover - defensive bridge boundary
                callback_errors.append(str(exc))

        callback = STREAM_CALLBACK(_callback)
        self._callback_ref = callback

        ret = self._library.lib.rkllm_bridge_generate(
            self._handle,
            request_id.encode("utf-8"),
            prompt.encode("utf-8"),
            callback,
            None,
            error_buffer,
            len(error_buffer),
        )
        trace_event(
            "python_bridge.generate.finish",
            request_id=request_id,
            ret_code=ret,
            prompt_length=len(prompt),
            output_length=len("".join(chunks)),
            output_text="".join(chunks),
            callback_errors=callback_errors,
            error_buffer=error_buffer.value.decode("utf-8", errors="ignore"),
        )
        if callback_errors:
            raise BridgeError(callback_errors[0])
        if ret != 0:
            raise BridgeError(error_buffer.value.decode("utf-8") or "RKLLM generation failed")
        return "".join(chunks)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.lib.rkllm_bridge_destroy(self._handle)
            self._handle = None

    def cancel(self) -> None:
        if not getattr(self, "_handle", None):
            return
        error_buffer = ctypes.create_string_buffer(1024)
        self._library.lib.rkllm_bridge_cancel(self._handle, error_buffer, len(error_buffer))


def load_engine_config_from_env() -> EngineConfig:
    root = Path(__file__).resolve().parents[1]
    default_bridge = root / "dist" / "native" / "linux_aarch64" / "librkllm_openai_bridge.so"
    default_model = root / "models" / "DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"

    return EngineConfig(
        model_path=os.getenv("RKLLM_MODEL_PATH", str(default_model)),
        bridge_lib_path=os.getenv("RKLLM_BRIDGE_LIB", str(default_bridge)),
        default_max_new_tokens=int(os.getenv("RKLLM_MAX_NEW_TOKENS", "2048")),
        max_context_len=int(os.getenv("RKLLM_MAX_CONTEXT_LEN", "2048")),
        top_k=int(os.getenv("RKLLM_TOP_K", "1")),
        top_p=float(os.getenv("RKLLM_TOP_P", "0.95")),
        temperature=float(os.getenv("RKLLM_TEMPERATURE", "0.8")),
        repeat_penalty=float(os.getenv("RKLLM_REPEAT_PENALTY", "1.1")),
        frequency_penalty=float(os.getenv("RKLLM_FREQUENCY_PENALTY", "0.0")),
        presence_penalty=float(os.getenv("RKLLM_PRESENCE_PENALTY", "0.0")),
        skip_special_token=os.getenv("RKLLM_SKIP_SPECIAL_TOKEN", "1") != "0",
        model_name=os.getenv("RKLLM_MODEL_NAME", "rkllm-local"),
        prompt_prefix=os.getenv("RKLLM_PROMPT_PREFIX", "<｜begin▁of▁sentence｜><｜User｜>"),
        prompt_postfix=os.getenv("RKLLM_PROMPT_POSTFIX", "<｜Assistant｜>"),
    )
