#include "rkllm_bridge.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>

#include "rkllm.h"

namespace {

struct GenerationContext {
    std::mutex mutex;
    std::condition_variable cv;
    rkllm_bridge_stream_callback callback = nullptr;
    void *user_data = nullptr;
    bool finished = false;
    bool errored = false;
    bool cancelled = false;
    std::string error_message;
};

struct BridgeState {
    LLMHandle handle = nullptr;
    std::mutex invoke_mutex;
    std::mutex state_mutex;
    std::mutex trace_mutex;
    std::atomic<bool> initialized{false};
    GenerationContext *active_generation = nullptr;
    rkllm_bridge_sampling_config_t config{};
    std::string model_path;
};

void write_error(char *buffer, int buffer_len, const std::string &message) {
    if (buffer == nullptr || buffer_len <= 0) {
        return;
    }

    std::strncpy(buffer, message.c_str(), static_cast<size_t>(buffer_len - 1));
    buffer[buffer_len - 1] = '\0';
}

std::string json_escape(const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += ch; break;
        }
    }
    return escaped;
}

void trace_native(BridgeState *bridge, const std::string &stage, const std::string &request_id, const std::string &message) {
    const char *trace_file = std::getenv("RKLLM_TRACE_FILE");
    if (trace_file == nullptr || std::strlen(trace_file) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bridge->trace_mutex);
    std::ofstream out(trace_file, std::ios::app);
    if (!out.is_open()) {
        return;
    }
    out << "{\"stage\":\"" << json_escape(stage)
        << "\",\"request_id\":\"" << json_escape(request_id)
        << "\",\"message\":\"" << json_escape(message)
        << "\"}\n";
}

void bridge_callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    auto *bridge = reinterpret_cast<BridgeState *>(userdata);
    if (bridge == nullptr) {
        return;
    }

    GenerationContext *generation = nullptr;
    {
        std::lock_guard<std::mutex> lock(bridge->state_mutex);
        generation = bridge->active_generation;
    }

    if (generation == nullptr) {
        return;
    }

    if (state == RKLLM_RUN_NORMAL) {
        if (!generation->cancelled && generation->callback != nullptr && result != nullptr && result->text != nullptr) {
            generation->callback(result->text, RKLLM_BRIDGE_EVENT_TOKEN, 0, generation->user_data);
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lock(generation->mutex);
        if (state == RKLLM_RUN_ERROR) {
            generation->errored = true;
            generation->finished = true;
            generation->error_message = "rkllm_run returned RKLLM_RUN_ERROR";
            if (generation->callback != nullptr) {
                generation->callback(generation->error_message.c_str(), RKLLM_BRIDGE_EVENT_ERROR, -1, generation->user_data);
            }
        } else if (state == RKLLM_RUN_FINISH) {
            generation->finished = true;
            if (generation->callback != nullptr && !generation->cancelled) {
                generation->callback("", RKLLM_BRIDGE_EVENT_FINISH, 0, generation->user_data);
            }
        }
    }
    generation->cv.notify_all();
}

}  // namespace

extern "C" void *rkllm_bridge_create() {
    return new BridgeState();
}

extern "C" int rkllm_bridge_init(void *bridge_handle, const char *model_path, const rkllm_bridge_sampling_config_t *config, char *error_buffer, int error_buffer_len) {
    auto *bridge = reinterpret_cast<BridgeState *>(bridge_handle);
    if (bridge == nullptr) {
        write_error(error_buffer, error_buffer_len, "bridge handle is null");
        return -1;
    }
    if (model_path == nullptr || std::strlen(model_path) == 0) {
        write_error(error_buffer, error_buffer_len, "model_path is required");
        return -1;
    }
    if (config == nullptr) {
        write_error(error_buffer, error_buffer_len, "sampling config is required");
        return -1;
    }

    std::lock_guard<std::mutex> lock(bridge->invoke_mutex);
    if (bridge->initialized.load()) {
        write_error(error_buffer, error_buffer_len, "bridge has already been initialized");
        return -1;
    }

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = const_cast<char *>(model_path);
    param.top_k = config->top_k;
    param.top_p = config->top_p;
    param.temperature = config->temperature;
    param.repeat_penalty = config->repeat_penalty;
    param.frequency_penalty = config->frequency_penalty;
    param.presence_penalty = config->presence_penalty;
    param.max_new_tokens = config->max_new_tokens;
    param.max_context_len = config->max_context_len;
    param.skip_special_token = config->skip_special_token != 0;
    param.extend_param.base_domain_id = 0;

    int ret = rkllm_init(&bridge->handle, &param, bridge_callback);
    bridge->model_path = model_path;
    trace_native(bridge, "native.init", "", std::string("model_path=") + model_path + ",ret=" + std::to_string(ret));
    if (ret != 0) {
        write_error(error_buffer, error_buffer_len, "rkllm_init failed");
        return ret;
    }

    bridge->config = *config;
    bridge->initialized.store(true);
    return 0;
}

extern "C" int rkllm_bridge_generate(void *bridge_handle, const char *request_id, const char *prompt, rkllm_bridge_stream_callback callback, void *user_data, char *error_buffer, int error_buffer_len) {
    auto *bridge = reinterpret_cast<BridgeState *>(bridge_handle);
    std::string request_id_str = request_id != nullptr ? request_id : "";
    if (bridge == nullptr) {
        write_error(error_buffer, error_buffer_len, "bridge handle is null");
        return -1;
    }
    if (!bridge->initialized.load() || bridge->handle == nullptr) {
        write_error(error_buffer, error_buffer_len, "bridge is not initialized");
        return -1;
    }
    if (prompt == nullptr || std::strlen(prompt) == 0) {
        write_error(error_buffer, error_buffer_len, "prompt is required");
        return -1;
    }

    std::lock_guard<std::mutex> lock(bridge->invoke_mutex);
    trace_native(
        bridge,
        "native.generate.start",
        request_id_str,
        std::string("prompt_length=") + std::to_string(std::strlen(prompt)) + ",model_path=" + bridge->model_path + ",prompt=" + prompt
    );
    GenerationContext generation;
    generation.callback = callback;
    generation.user_data = user_data;

    {
        std::lock_guard<std::mutex> state_lock(bridge->state_mutex);
        bridge->active_generation = &generation;
    }

    RKLLMInput input;
    std::memset(&input, 0, sizeof(RKLLMInput));
    input.input_type = RKLLM_INPUT_PROMPT;
    input.prompt_input = const_cast<char *>(prompt);

    RKLLMInferParam infer_param;
    std::memset(&infer_param, 0, sizeof(RKLLMInferParam));
    infer_param.mode = RKLLM_INFER_GENERATE;

    int ret = rkllm_run(bridge->handle, &input, &infer_param, bridge);
    trace_native(bridge, "native.generate.run_return", request_id_str, std::string("ret=") + std::to_string(ret));
    if (ret != 0) {
        {
            std::lock_guard<std::mutex> state_lock(bridge->state_mutex);
            bridge->active_generation = nullptr;
        }
        write_error(error_buffer, error_buffer_len, "rkllm_run failed");
        return ret;
    }

    std::unique_lock<std::mutex> wait_lock(generation.mutex);
    generation.cv.wait(wait_lock, [&generation]() {
        return generation.finished || generation.errored;
    });
    wait_lock.unlock();

    {
        std::lock_guard<std::mutex> state_lock(bridge->state_mutex);
        bridge->active_generation = nullptr;
    }

    if (generation.errored) {
        trace_native(bridge, "native.generate.error", request_id_str, generation.error_message);
        write_error(error_buffer, error_buffer_len, generation.error_message);
        return -1;
    }
    trace_native(bridge, "native.generate.finish", request_id_str, "ok");
    return 0;
}

extern "C" int rkllm_bridge_cancel(void *bridge_handle, char *error_buffer, int error_buffer_len) {
    auto *bridge = reinterpret_cast<BridgeState *>(bridge_handle);
    if (bridge == nullptr) {
        write_error(error_buffer, error_buffer_len, "bridge handle is null");
        return -1;
    }

    std::lock_guard<std::mutex> lock(bridge->state_mutex);
    if (bridge->active_generation != nullptr) {
        bridge->active_generation->cancelled = true;
    }
    trace_native(bridge, "native.cancel", "", "cancel requested");
    write_error(error_buffer, error_buffer_len, "cancel is not supported by the current RKLLM bridge");
    return -1;
}

extern "C" void rkllm_bridge_destroy(void *bridge_handle) {
    auto *bridge = reinterpret_cast<BridgeState *>(bridge_handle);
    if (bridge == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(bridge->invoke_mutex);
    if (bridge->handle != nullptr) {
        trace_native(bridge, "native.destroy", "", "destroy handle");
        rkllm_destroy(bridge->handle);
        bridge->handle = nullptr;
    }
    delete bridge;
}
