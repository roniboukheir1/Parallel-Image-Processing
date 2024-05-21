#include "header.h"
#include <cuda_runtime.h>

// Example CUDA kernel for image processing
__global__ void kernelProcessImage(png_byte *d_input, png_byte *d_output, int width, int height, int pixel_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < width * height * pixel_size)
    {
        // Simple example: copy input to output
        d_output[idx] = d_input[idx];
    }
}

void processImageCUDA(PROCESSING_JOB *job)
{
    png_byte *d_input, *d_output;
    size_t size = job->width * job->height * job->pixel_size * sizeof(png_byte);

    // Allocate device memory
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    // Define grid and block dimensions

    dim3 blockDim(256);
    dim3 gridDim((job->width * job->height * job->pixel_size + blockDim.x - 1) / blockDim.x);

    // Launch kernel
    kernelProcessImage<<<gridDim, blockDim>>>(d_input, d_output, job->width, job->height, job->pixel_size);
    cudaGetLastError();

    // Copy result back to host
    cudaMemcpy(job->dest_raw, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Completed processImageCUDA for job: %s\n", job->source_name);
}

void execute_jobs_gpu(PROCESSING_JOB **jobs)
{
    for (int i = 0; jobs[i] != NULL; i++)
    {
        processImageCUDA(jobs[i]);
    }
}

void execute_jobs_gpu_shared(PROCESSING_JOB **jobs)
{
    for (int i = 0; jobs[i] != NULL; i++)
    {
        sharedProcessImageCUDA(jobs[i]);
    }
}

__global__ void sharedProcessImage(png_byte *d_input, png_byte *d_output, int width, int height, int pixel_size)
{
    extern __shared__ png_byte shared_data[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = ty * blockDim.x + tx;

    if (index < width * height * pixel_size)
    {
        shared_data[idx] = d_input[index];
    }
    __syncthreads();

    if (index < width * height * pixel_size)
    {
        d_output[index] = shared_data[idx];
    }
}

void sharedProcessImageCUDA(PROCESSING_JOB *job)
{
    printf("Starting processImageCUDA for job: %s\n", job->source_name);

    png_byte *d_input, *d_output;
    size_t size = job->width * job->height * job->pixel_size * sizeof(png_byte);

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    cudaMemcpy(d_input, job->source_raw, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((job->width + blockDim.x - 1) / blockDim.x, (job->height + blockDim.y - 1) / blockDim.y);

    size_t sharedMemSize = blockDim.x * blockDim.y * sizeof(png_byte);

    sharedProcessImage<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, job->width, job->height, job->pixel_size);
    cudaMemcpy(job->dest_raw, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    printf("Completed processImageCUDA for job: %s\n", job->source_name);
}
