#include <cstdint>
#include <cstddef>
#include <cstdio>

#include "rkllm.h"

int main() {
    std::printf("sizeof(RKLLMExtendParam)=%zu\n", sizeof(RKLLMExtendParam));
    std::printf("sizeof(RKLLMParam)=%zu\n", sizeof(RKLLMParam));
    std::printf("sizeof(RKLLMInput)=%zu\n", sizeof(RKLLMInput));
    std::printf("sizeof(RKLLMInferParam)=%zu\n", sizeof(RKLLMInferParam));
    std::printf("sizeof(RKLLMResultLastHiddenLayer)=%zu\n", sizeof(RKLLMResultLastHiddenLayer));
    std::printf("sizeof(RKLLMResult)=%zu\n", sizeof(RKLLMResult));

    std::printf("offsetof(RKLLMParam, model_path)=%zu\n", offsetof(RKLLMParam, model_path));
    std::printf("offsetof(RKLLMParam, max_context_len)=%zu\n", offsetof(RKLLMParam, max_context_len));
    std::printf("offsetof(RKLLMParam, max_new_tokens)=%zu\n", offsetof(RKLLMParam, max_new_tokens));
    std::printf("offsetof(RKLLMParam, top_k)=%zu\n", offsetof(RKLLMParam, top_k));
    std::printf("offsetof(RKLLMParam, top_p)=%zu\n", offsetof(RKLLMParam, top_p));
    std::printf("offsetof(RKLLMParam, temperature)=%zu\n", offsetof(RKLLMParam, temperature));
    std::printf("offsetof(RKLLMParam, repeat_penalty)=%zu\n", offsetof(RKLLMParam, repeat_penalty));
    std::printf("offsetof(RKLLMParam, frequency_penalty)=%zu\n", offsetof(RKLLMParam, frequency_penalty));
    std::printf("offsetof(RKLLMParam, presence_penalty)=%zu\n", offsetof(RKLLMParam, presence_penalty));
    std::printf("offsetof(RKLLMParam, mirostat)=%zu\n", offsetof(RKLLMParam, mirostat));
    std::printf("offsetof(RKLLMParam, mirostat_tau)=%zu\n", offsetof(RKLLMParam, mirostat_tau));
    std::printf("offsetof(RKLLMParam, mirostat_eta)=%zu\n", offsetof(RKLLMParam, mirostat_eta));
    std::printf("offsetof(RKLLMParam, skip_special_token)=%zu\n", offsetof(RKLLMParam, skip_special_token));
    std::printf("offsetof(RKLLMParam, is_async)=%zu\n", offsetof(RKLLMParam, is_async));
    std::printf("offsetof(RKLLMParam, img_start)=%zu\n", offsetof(RKLLMParam, img_start));
    std::printf("offsetof(RKLLMParam, img_end)=%zu\n", offsetof(RKLLMParam, img_end));
    std::printf("offsetof(RKLLMParam, img_content)=%zu\n", offsetof(RKLLMParam, img_content));
    std::printf("offsetof(RKLLMParam, extend_param)=%zu\n", offsetof(RKLLMParam, extend_param));

    std::printf("offsetof(RKLLMInput, input_type)=%zu\n", offsetof(RKLLMInput, input_type));
    std::printf("offsetof(RKLLMInput, prompt_input)=%zu\n", offsetof(RKLLMInput, prompt_input));

    std::printf("offsetof(RKLLMInferParam, mode)=%zu\n", offsetof(RKLLMInferParam, mode));
    std::printf("offsetof(RKLLMInferParam, lora_params)=%zu\n", offsetof(RKLLMInferParam, lora_params));
    std::printf("offsetof(RKLLMInferParam, prompt_cache_params)=%zu\n", offsetof(RKLLMInferParam, prompt_cache_params));

    std::printf("offsetof(RKLLMResult, text)=%zu\n", offsetof(RKLLMResult, text));
    std::printf("offsetof(RKLLMResult, token_id)=%zu\n", offsetof(RKLLMResult, token_id));
    std::printf("offsetof(RKLLMResult, last_hidden_layer)=%zu\n", offsetof(RKLLMResult, last_hidden_layer));
    return 0;
}
