from __future__ import annotations

import ctypes


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


def dump_layout(struct_name: str, struct_type: type[ctypes.Structure], field_names: list[str]) -> None:
    print(f"sizeof({struct_name})={ctypes.sizeof(struct_type)}")
    for field_name in field_names:
        print(f"offsetof({struct_name}, {field_name})={getattr(struct_type, field_name).offset}")


def main() -> int:
    dump_layout("RKLLMExtendParam", RKLLMExtendParam, ["base_domain_id", "reserved"])
    dump_layout(
        "RKLLMParam",
        RKLLMParam,
        [
            "model_path",
            "max_context_len",
            "max_new_tokens",
            "top_k",
            "top_p",
            "temperature",
            "repeat_penalty",
            "frequency_penalty",
            "presence_penalty",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "skip_special_token",
            "is_async",
            "img_start",
            "img_end",
            "img_content",
            "extend_param",
        ],
    )
    dump_layout("RKLLMInput", RKLLMInput, ["input_type", "prompt_input"])
    dump_layout("RKLLMInferParam", RKLLMInferParam, ["mode", "lora_params", "prompt_cache_params"])
    dump_layout("RKLLMResultLastHiddenLayer", RKLLMResultLastHiddenLayer, ["hidden_states", "embd_size", "num_tokens"])
    dump_layout("RKLLMResult", RKLLMResult, ["text", "token_id", "last_hidden_layer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
