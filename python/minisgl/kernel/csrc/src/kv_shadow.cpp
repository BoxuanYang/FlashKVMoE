#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

struct ScatterPlan {
    std::byte* k_dst;
    std::byte* v_dst;
    const std::byte* src;
    const int32_t* indices;
    std::size_t count;
    std::size_t row_bytes;
    std::size_t capacity;
    bool delete_after_run;
};

void CUDART_CB scatter_host(void* opaque) {
    auto* plan = static_cast<ScatterPlan*>(opaque);
    const std::size_t src_stride = 2 * plan->row_bytes;
    for (std::size_t i = 0; i < plan->count; ++i) {
        const int64_t index = plan->indices[i];
        if (index < 0 || static_cast<std::size_t>(index) >= plan->capacity) {
            continue;
        }
        const auto* row = plan->src + i * src_stride;
        std::memcpy(plan->k_dst + index * plan->row_bytes, row, plan->row_bytes);
        std::memcpy(
            plan->v_dst + index * plan->row_bytes,
            row + plan->row_bytes,
            plan->row_bytes);
    }
    if (plan->delete_after_run) {
        delete plan;
    }
}

}  // namespace

class HostScatterLauncher {
public:
    void launch(
        uintptr_t stream_ptr,
        uintptr_t k_dst,
        uintptr_t v_dst,
        uintptr_t src,
        uintptr_t indices,
        std::size_t count,
        std::size_t row_bytes,
        std::size_t capacity) {
        if (!stream_ptr || !k_dst || !v_dst || !src || !indices || !count || !row_bytes) {
            throw std::invalid_argument("KV shadow scatter received an invalid pointer or size");
        }

        auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        check_cuda(cudaStreamIsCapturing(stream, &capture_status));
        const bool captured = capture_status != cudaStreamCaptureStatusNone;
        auto plan = std::make_unique<ScatterPlan>(ScatterPlan{
            reinterpret_cast<std::byte*>(k_dst),
            reinterpret_cast<std::byte*>(v_dst),
            reinterpret_cast<const std::byte*>(src),
            reinterpret_cast<const int32_t*>(indices),
            count,
            row_bytes,
            capacity,
            !captured,
        });
        auto* raw_plan = plan.get();
        check_cuda(cudaLaunchHostFunc(stream, scatter_host, raw_plan));
        if (captured) {
            // A captured host node reuses its user-data pointer on every replay.
            // Retain the plan for the lifetime of this launcher/its CUDA graphs.
            std::lock_guard<std::mutex> lock(mutex_);
            captured_plans_.push_back(std::move(plan));
        } else {
            plan.release();
        }
    }

private:
    std::mutex mutex_;
    std::vector<std::unique_ptr<ScatterPlan>> captured_plans_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<HostScatterLauncher>(module, "HostScatterLauncher")
        .def(py::init<>())
        .def("launch", &HostScatterLauncher::launch);
}
