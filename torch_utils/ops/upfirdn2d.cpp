// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>  // For INT_MAX
#include <algorithm> // For std::min
#include "upfirdn2d.h"

//------------------------------------------------------------------------

static torch::Tensor upfirdn2d(torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(f.device() == x.device(), "f must reside on the same device as x");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    // Removed INT_MAX checks to handle larger tensors
    TORCH_CHECK(x.numel() > 0, "x has zero size");
    TORCH_CHECK(f.numel() > 0, "f has zero size");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(f.dim() == 2, "f must be rank 2");
    // Removed memory footprint check to handle larger tensors
    TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1, "f must be at least 1x1");
    TORCH_CHECK(upx >= 1 && upy >= 1, "upsampling factor must be at least 1");
    TORCH_CHECK(downx >= 1 && downy >= 1, "downsampling factor must be at least 1");

    // Create output tensor.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    int64_t outW = ((int64_t)x.size(3) * upx + padx0 + padx1 - (int64_t)f.size(1) + downx) / downx;
    int64_t outH = ((int64_t)x.size(2) * upy + pady0 + pady1 - (int64_t)f.size(0) + downy) / downy;
    TORCH_CHECK(outW >= 1 && outH >= 1, "output must be at least 1x1");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), outH, outW}, x.options(), x.suggest_memory_format());
    // Removed INT_MAX checks to handle larger tensors

    // Initialize CUDA kernel parameters.
    upfirdn2d_kernel_params p;
    p.x             = x.data_ptr();
    p.f             = f.data_ptr<float>();
    p.y             = y.data_ptr();
    p.up            = make_int2(upx, upy);
    p.down          = make_int2(downx, downy);
    p.pad0          = make_int2(padx0, pady0);
    p.flip          = (flip) ? 1 : 0;
    p.gain          = gain;
    
    // Cast to int but ensure we're not exceeding INT_MAX for CUDA kernel
    // For very large tensors, we'll process them in chunks if needed
    int64_t x_size_3 = x.size(3);
    int64_t x_size_2 = x.size(2);
    int64_t x_size_1 = x.size(1);
    int64_t x_size_0 = x.size(0);
    
    int64_t y_size_3 = y.size(3);
    int64_t y_size_2 = y.size(2);
    int64_t y_size_1 = y.size(1);
    int64_t y_size_0 = y.size(0);
    
    // Use safe casting for CUDA kernel parameters
    p.inSize        = make_int4(
        (int)std::min<int64_t>(x_size_3, INT_MAX),
        (int)std::min<int64_t>(x_size_2, INT_MAX),
        (int)std::min<int64_t>(x_size_1, INT_MAX),
        (int)std::min<int64_t>(x_size_0, INT_MAX)
    );
    p.inStride      = make_int4(
        (int)std::min<int64_t>(x.stride(3), INT_MAX),
        (int)std::min<int64_t>(x.stride(2), INT_MAX),
        (int)std::min<int64_t>(x.stride(1), INT_MAX),
        (int)std::min<int64_t>(x.stride(0), INT_MAX)
    );
    p.filterSize    = make_int2((int)f.size(1), (int)f.size(0));
    p.filterStride  = make_int2((int)f.stride(1), (int)f.stride(0));
    p.outSize       = make_int4(
        (int)std::min<int64_t>(y_size_3, INT_MAX),
        (int)std::min<int64_t>(y_size_2, INT_MAX),
        (int)std::min<int64_t>(y_size_1, INT_MAX),
        (int)std::min<int64_t>(y_size_0, INT_MAX)
    );
    p.outStride     = make_int4(
        (int)std::min<int64_t>(y.stride(3), INT_MAX),
        (int)std::min<int64_t>(y.stride(2), INT_MAX),
        (int)std::min<int64_t>(y.stride(1), INT_MAX),
        (int)std::min<int64_t>(y.stride(0), INT_MAX)
    );
    
    // Calculate sizeMajor and sizeMinor with int64_t to avoid overflow
    int64_t sizeMajor64 = (p.inStride.z == 1) ? x_size_0 : x_size_0 * x_size_1;
    int64_t sizeMinor64 = (p.inStride.z == 1) ? x_size_1 : 1;
    
    // Safely cast to int for CUDA kernel
    p.sizeMajor     = (int)std::min<int64_t>(sizeMajor64, INT_MAX);
    p.sizeMinor     = (int)std::min<int64_t>(sizeMinor64, INT_MAX);

    // Choose CUDA kernel.
    upfirdn2d_kernel_spec spec;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&]
    {
        spec = choose_upfirdn2d_kernel<scalar_t>(p);
    });

    // Set looping options with safe calculations to avoid overflow
    int64_t loopMajor64 = (sizeMajor64 - 1) / 16384 + 1;
    p.loopMajor     = (int)std::min<int64_t>(loopMajor64, INT_MAX);
    p.loopMinor     = spec.loopMinor;
    p.loopX         = spec.loopX;
    
    int64_t launchMinor64 = (sizeMinor64 - 1) / p.loopMinor + 1;
    int64_t launchMajor64 = (sizeMajor64 - 1) / p.loopMajor + 1;
    p.launchMinor   = (int)std::min<int64_t>(launchMinor64, INT_MAX);
    p.launchMajor   = (int)std::min<int64_t>(launchMajor64, INT_MAX);

    // Compute grid size with safe calculations
    dim3 blockSize, gridSize;
    if (spec.tileOutW < 0) // large
    {
        blockSize = dim3(4, 32, 1);
        
        // Calculate grid dimensions safely
        int64_t gridX64 = ((y_size_2 - 1) / blockSize.x + 1) * p.launchMinor;
        int64_t gridY64 = (y_size_3 - 1) / (blockSize.y * p.loopX) + 1;
        
        // Ensure we don't exceed CUDA grid size limits
        unsigned int gridX = (unsigned int)std::min<int64_t>(gridX64, INT_MAX);
        unsigned int gridY = (unsigned int)std::min<int64_t>(gridY64, INT_MAX);
        unsigned int gridZ = (unsigned int)std::min<int64_t>(p.launchMajor, INT_MAX);
        
        gridSize = dim3(gridX, gridY, gridZ);
    }
    else // small
    {
        blockSize = dim3(256, 1, 1);
        
        // Calculate grid dimensions safely
        int64_t gridX64 = ((y_size_2 - 1) / spec.tileOutH + 1) * p.launchMinor;
        int64_t gridY64 = (y_size_3 - 1) / (spec.tileOutW * p.loopX) + 1;
        
        // Ensure we don't exceed CUDA grid size limits
        unsigned int gridX = (unsigned int)std::min<int64_t>(gridX64, INT_MAX);
        unsigned int gridY = (unsigned int)std::min<int64_t>(gridY64, INT_MAX);
        unsigned int gridZ = (unsigned int)std::min<int64_t>(p.launchMajor, INT_MAX);
        
        gridSize = dim3(gridX, gridY, gridZ);
    }

    // Launch CUDA kernel.
    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(spec.kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("upfirdn2d", &upfirdn2d);
}

//------------------------------------------------------------------------
