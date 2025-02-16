#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__global__
void matSearchKernel(float *mat_d, int *result_d, int m, int n, float target) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        int offset = row * n + col;
        if (mat_d[offset] == target) {
            atomicOr(result_d, 1);
        }
    }
}

bool matSearch(float *mat, int m, int n, float target) {
    float *mat_d;
    int *result_d;
    int result_h = 0;

    size_t size = m * n * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&mat_d, size));
    CUDA_CHECK(cudaMalloc((void **)&result_d, sizeof(int)));
    CUDA_CHECK(cudaMemset(result_d, 0, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(mat_d, mat, size, cudaMemcpyHostToDevice));

    dim3 dimGrid((n + 15) / 16, (m + 15) / 16, 1); // Ensure coverage
    dim3 dimBlock(16, 16, 1);
    matSearchKernel<<<dimGrid, dimBlock>>>(mat_d, result_d, m, n, target);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel

    CUDA_CHECK(cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(mat_d);
    cudaFree(result_d);

    return (result_h != 0);
}
/** Note that we have also defined a CUDA_CHECK macro as this is the common practice used by the developers for simplifying the CUDA API errors
    and you will also use this always.
*/

int main() {
    int m = 1 << 10; // 1024 rows
    int n = 1 << 10; // 1024 columns
    float target = 25.0f;

    float *mat = (float *)malloc(m * n * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * n + j] = i + j; // i+j as floats
        }
    }
    /**
        This is we usually handle 2D matrix operation in CUDA , Most of the image ops are also handled this way only.
        There are cuda APIs as well for copying 2D and 3D matrices but still we use 1D array and make it to simulate 
        the 2D/3D array because of the performance of using the 1d array.

        Note that this is only how you will initialize you 2D or 3D arra in cuda as this is the convention.    
     */

    bool is_present = matSearch(mat, m, n, target);
    printf("Target %s\n", is_present ? "found!" : "not found.");

    free(mat);
    return 0;
}

/**
    -------------------------------------------------------IMPORTANT LEARNINGS ------------------------------------------------------------
     
     to check if GPU is allocated to you or not use this code snippet in google colab

     import pytorch
     print("GPU available : " , torch.cuda.isavailable())

     !nvcc your_filename.cu -o output -arch=sm_75
     !./output


     Now the above comands is used to compile and run the cuda code that you have , but note that
     we are also using -arch=sm_75

     what is this? So the thing is every NVIDIA GPU has a CUDA toolkit installed in it and these CUDA toolkit can only run 
     if the CUDA version if the GPU is compatible only. So if the CUDA version is newer that your GPU driver can support
     then there will be some error, what i encountered was 
     
     -> "the provided PTX was compiled with an unsupported toolchain"

    So how i fixed it is i tell the compiler about the compute capability of my GPU and it solves the issue
    So google colab provides us the NVIDIA T4 GPU access and its compute capability is 7.5, NVIDIA H100 is the most advanced
    GPU and it has the compute capability of 9.0

    What -arch=sm_75 Does ?

        ->
        -arch=sm_XX:
        sm stands for "streaming multiprocessor", which defines the GPU architecture.
        sm_75 corresponds to compute capability 7.5, the architecture of NVIDIA Tesla T4 GPUs (used in Google Colab).
        This flag tells the compiler to generate GPU machine code (SASS) and PTX (Parallel Thread Execution) intermediate code 
        specifically for GPUs with compute capability 7.5.

    Why It Fixes the Error?
    
    ->
        Original Error:
        CUDA error: the provided PTX was compiled with an unsupported toolchain
        This occurs when the CUDA toolkit compiles code for a newer architecture (e.g., sm_80 for Ampere GPUs) 
        that your GPU driver/hardware doesn’t support.

        With -arch=sm_75:
        The compiler generates code only for the T4’s compute capability (7.5). 
        This ensures compatibility with Colab’s Tesla T4 GPU and its installed driver.

    Google Colab’s Environment?

    ->
        GPU in Colab:
        Colab provides Tesla T4 GPUs (compute capability 7.5). The preinstalled NVIDIA driver typically supports CUDA 11.x,
        which aligns with sm_75.

        Default Compilation Without -arch:
        If you omit -arch, nvcc may default to a newer architecture (e.g., sm_80 for CUDA 12.x), 
        which the T4 GPU cannot execute. This causes the "unsupported toolchain" error.

    How This Ensures Compatibility ?

    ->
        PTX vs. SASS:

        PTX (Parallel Thread Execution) is a portable intermediate code.

        SASS is the actual GPU machine code.
        By specifying -arch=sm_75, the compiler generates both PTX and SASS for the T4, 
        allowing the GPU driver to directly execute the code without needing to JIT-compile PTX for an unsupported architecture.

        Avoiding Driver Mismatch:
        Colab’s driver may not support PTX code for newer architectures. Compiling directly for sm_75 bypasses this issue.
    
    Why It Didn’t Work Before?

    ->
        Default Architecture:
        Without -arch=sm_75, nvcc may have defaulted to a newer architecture (e.g., sm_80 for CUDA 12.x), 
        which the Tesla T4 (compute 7.5) cannot execute.

        Driver Limitations:
        The NVIDIA driver in Colab may not support JIT-compiling PTX for newer architectures.
 */