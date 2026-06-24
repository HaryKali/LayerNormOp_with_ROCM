#include <torch/extension.h>
#include <hip/hip_runtime.h>

void launchLayerNorm(const float* input, float* output,
                     int rows, int cols, float eps);

torch::Tensor layernorm_forward(torch::Tensor input, float eps) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [rows, cols]");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);

    int rows = input.size(0);
    int cols = input.size(1);

    launchLayerNorm(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols, eps
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (HIP)");
}
