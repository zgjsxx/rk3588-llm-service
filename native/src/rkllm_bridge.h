#pragma once

#ifdef _WIN32
#define RKLLM_BRIDGE_EXPORT __declspec(dllexport)
#else
#define RKLLM_BRIDGE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*rkllm_bridge_stream_callback)(const char *text, int event, int error_code, void *user_data);

typedef struct rkllm_bridge_sampling_config_t {
    int max_new_tokens;
    int max_context_len;
    int top_k;
    float top_p;
    float temperature;
    float repeat_penalty;
    float frequency_penalty;
    float presence_penalty;
    int skip_special_token;
} rkllm_bridge_sampling_config_t;

enum rkllm_bridge_event_t {
    RKLLM_BRIDGE_EVENT_TOKEN = 0,
    RKLLM_BRIDGE_EVENT_FINISH = 1,
    RKLLM_BRIDGE_EVENT_ERROR = 2
};

RKLLM_BRIDGE_EXPORT void *rkllm_bridge_create();
RKLLM_BRIDGE_EXPORT int rkllm_bridge_init(void *bridge_handle, const char *model_path, const rkllm_bridge_sampling_config_t *config, char *error_buffer, int error_buffer_len);
RKLLM_BRIDGE_EXPORT int rkllm_bridge_generate(void *bridge_handle, const char *request_id, const char *prompt, rkllm_bridge_stream_callback callback, void *user_data, char *error_buffer, int error_buffer_len);
RKLLM_BRIDGE_EXPORT int rkllm_bridge_cancel(void *bridge_handle, char *error_buffer, int error_buffer_len);
RKLLM_BRIDGE_EXPORT void rkllm_bridge_destroy(void *bridge_handle);

#ifdef __cplusplus
}
#endif
