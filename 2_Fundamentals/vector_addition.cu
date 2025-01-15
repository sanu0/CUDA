#include<stdio.h>
#include<stdlib.h>

__global__
void vecAddKernal(float *A, float *B, float *C, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<n){
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A, float *B, float *C, int n){
    float *A_d,*B_d,*C_d;
    int size = n * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    
    vecAddKernal<<< ceil(n/256.0) , 256.0 >>>(A_d,B_d,C_d,n);
    
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    int n = 1024; // Size of vectors
    float *A, *B, *C;

    // Allocate memory for vectors on the host
    A = (float *)malloc(n * sizeof(float));
    B = (float *)malloc(n * sizeof(float));
    C = (float *)malloc(n * sizeof(float));

    // Initialize vectors A and B
    for (int i = 0; i < n; i++) {
        A[i] = i * 1.0f; // A = [0, 1, 2, ...]
        B[i] = i * 2.0f; // B = [0, 2, 4, ...]
    }

    // Call the CUDA vector addition function
    vecAdd(A, B, C, n);

    // Print the result (for demonstration, print the first 10 elements)
    printf("Result (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}