#include "header.h"
#include <cuda_runtime.h>
#include <vector>
#include <thread>

// Define the filter size and tile size
#define FILTER_SIZE 5
#define TILE_SIZE 16

#include <unordered_map>
#include <string>

__constant__ float d_filter[FILTER_SIZE * FILTER_SIZE];

void copyFilterToConstantMemory(float *h_filter)
{
    cudaMemcpyToSymbol(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
}

std::string algoToString(PROCESSING_ALGO algo)
{
    switch (algo)
    {
    case SHARPEN:
        return "SHARPEN";
    case EDGE:
        return "EDGE";
    case BLUR:
        return "BLUR";
    case UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

__global__ void applyFilterSharedv1(float *filter, png_byte *d_input, png_byte *d_output, int width, int height, int pixel_size)
{
    extern __shared__ png_byte shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int shared_width = TILE_SIZE + FILTER_SIZE - 1;
    int halo_width = FILTER_SIZE / 2;

    // Calculate the global index
    int global_idx = (y * width + x) * pixel_size;

    // Load main tile into shared memory
    for (int i = 0; i < pixel_size; i++)
    {
        int shared_idx = ((ty + halo_width) * shared_width + (tx + halo_width)) * pixel_size + i;
        if (x < width && y < height)
        {
            shared_mem[shared_idx] = d_input[global_idx + i];
        }
        else
        {
            shared_mem[shared_idx] = 0;
        }
    }

    // Load halo elements once per channel
    if (tx < halo_width && x >= halo_width)
    {
        for (int i = 0; i < pixel_size; i++)
        {
            int shared_idx = ((ty + halo_width) * shared_width + tx) * pixel_size + i;
            int halo_global_idx = ((y * width + (x - halo_width)) * pixel_size) + i;
            shared_mem[shared_idx] = (x - halo_width < 0) ? 0 : d_input[halo_global_idx];
        }
    }

    if (ty < halo_width && y >= halo_width)
    {
        for (int i = 0; i < pixel_size; i++)
        {
            int shared_idx = (ty * shared_width + (tx + halo_width)) * pixel_size + i;
            int halo_global_idx = (((y - halo_width) * width + x) * pixel_size) + i;
            shared_mem[shared_idx] = (y - halo_width < 0) ? 0 : d_input[halo_global_idx];
        }
    }

    // Apply filter and write to output
    __syncthreads();

    if (x < width && y < height)
    {
        float sum = 0.0f;
        for (int i = -halo_width; i <= halo_width; i++)
        {
            for (int j = -halo_width; j <= halo_width; j++)
            {
                int shared_idx = ((ty + halo_width + i) * shared_width + (tx + halo_width + j)) * pixel_size;
                sum += shared_mem[shared_idx] * d_filter[(i + halo_width) * FILTER_SIZE + (j + halo_width)];
            }
        }

        for (int i = 0; i < pixel_size; i++)
        {
            d_output[global_idx + i] = (png_byte)min(max(sum, 0.0f), 255.0f);
        }
    }
}

void processImageCUDAv1(PROCESSING_JOB *job)
{
    // Allocate device memory
    png_byte *d_input, *d_output;
    size_t size = job->width * job->height * job->pixel_size * sizeof(png_byte);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input image to device
    cudaMemcpy(d_input, job->source_raw, size, cudaMemcpyHostToDevice);

    // Copy filter to constant memory
    copyFilterToConstantMemory(getAlgoFilterByType(job->processing_algo));

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((job->width + TILE_SIZE - 1) / TILE_SIZE, (job->height + TILE_SIZE - 1) / TILE_SIZE);
    size_t sharedMemSize = (TILE_SIZE + FILTER_SIZE - 1) * (TILE_SIZE + FILTER_SIZE - 1) * job->pixel_size * sizeof(png_byte);

    // Launch kernel
    applyFilterSharedv1<<<gridDim, blockDim, sharedMemSize>>>(d_filter, d_input, d_output, job->width, job->height, job->pixel_size);
    cudaGetLastError();

    // Copy result back to host
    cudaMemcpy(job->dest_raw, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Completed processImageCUDAv1 for job: %s\n", job->source_name);
}

// __global__ void applyFilterShared(float *filter, png_byte *d_input, png_byte *d_output, int width, int height, int pixel_size)
// {
//     extern __shared__ png_byte shared_mem[];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int x = blockIdx.x * TILE_SIZE + tx;
//     int y = blockIdx.y * TILE_SIZE + ty;

//     int shared_width = TILE_SIZE + FILTER_SIZE - 1;
//     int halo_width = FILTER_SIZE / 2;

//     // Calculate the global index
//     int global_idx = (y * width + x) * pixel_size;

//     // Load data into shared memory
//     for (int i = 0; i < pixel_size; i++)
//     {
//         int shared_idx = ((ty + halo_width) * shared_width + (tx + halo_width)) * pixel_size + i;
//         if (x < width && y < height)
//         {
//             shared_mem[shared_idx] = d_input[global_idx + i];
//         }
//         else
//         {
//             shared_mem[shared_idx] = 0;
//         }

//         // Load halo elements
//         if (tx < halo_width && x >= halo_width)
//         {
//             shared_mem[((ty + halo_width) * shared_width + tx) * pixel_size + i] = d_input[(y * width + (x - halo_width)) * pixel_size + i];
//         }
//         if (ty < halo_width && y >= halo_width)
//         {
//             shared_mem[(ty * shared_width + (tx + halo_width)) * pixel_size + i] = d_input[((y - halo_width) * width + x) * pixel_size + i];
//         }
//         if (tx >= TILE_SIZE - halo_width && x < width - halo_width)
//         {
//             shared_mem[((ty + halo_width) * shared_width + (tx + FILTER_SIZE - halo_width)) * pixel_size + i] = d_input[(y * width + (x + halo_width)) * pixel_size + i];
//         }
//         if (ty >= TILE_SIZE - halo_width && y < height - halo_width)
//         {
//             shared_mem[((ty + FILTER_SIZE - halo_width) * shared_width + (tx + halo_width)) * pixel_size + i] = d_input[((y + halo_width) * width + x) * pixel_size + i];
//         }
//     }

//     __syncthreads();

//     // Apply the filter
//     if (x < width && y < height)
//     {
//         float r = 0.0, g = 0.0, b = 0.0;
//         for (int fy = 0; fy < FILTER_SIZE; fy++)
//         {
//             for (int fx = 0; fx < FILTER_SIZE; fx++)
//             {
//                 int shared_idx = ((ty + fy) * shared_width + (tx + fx)) * pixel_size;
//                 int f_idx = fy * FILTER_SIZE + fx;
//                 r += shared_mem[shared_idx] * filter[f_idx];
//                 if (pixel_size > 1)
//                 {
//                     g += shared_mem[shared_idx + 1] * filter[f_idx];
//                     b += shared_mem[shared_idx + 2] * filter[f_idx];
//                 }
//             }
//         }

//         int dest_idx = (y * width + x) * pixel_size;
//         d_output[dest_idx] = min(max(int(r), 0), 255);
//         if (pixel_size > 1)
//         {
//             d_output[dest_idx + 1] = min(max(int(g), 0), 255);
//             d_output[dest_idx + 2] = min(max(int(b), 0), 255);
//         }
//     }
// }

__global__ void applyFilterShared(png_byte *d_input, png_byte *d_output, int width, int height, int pixel_size)
{
    extern __shared__ png_byte shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int shared_width = TILE_SIZE + FILTER_SIZE - 1;
    int halo_width = FILTER_SIZE / 2;

    int global_idx = (y * width + x) * pixel_size;

    for (int i = 0; i < pixel_size; i++)
    {
        int shared_idx = ((ty + halo_width) * shared_width + (tx + halo_width)) * pixel_size + i;
        if (x < width && y < height)
        {
            shared_mem[shared_idx] = d_input[global_idx + i];
        }
        else
        {
            shared_mem[shared_idx] = 0;
        }
    }

    if (tx < halo_width && x >= halo_width)
    {
        for (int i = 0; i < pixel_size; i++)
        {
            shared_mem[((ty + halo_width) * shared_width + tx) * pixel_size + i] = d_input[(y * width + (x - halo_width)) * pixel_size + i];
        }
    }
    if (ty < halo_width && y >= halo_width)
    {
        for (int i = 0; i < pixel_size; i++)
        {
            shared_mem[(ty * shared_width + (tx + halo_width)) * pixel_size + i] = d_input[((y - halo_width) * width + x) * pixel_size + i];
        }
    }
    if (tx >= TILE_SIZE - halo_width && x < width - halo_width)
    {
        for (int i = 0; i < pixel_size; i++)
        {
            shared_mem[((ty + halo_width) * shared_width + (tx + FILTER_SIZE - halo_width)) * pixel_size + i] = d_input[(y * width + (x + halo_width)) * pixel_size + i];
        }
    }
    if (ty >= TILE_SIZE - halo_width && y < height - halo_width)
    {
        for (int i = 0; i < pixel_size; i++)
        {
            shared_mem[((ty + FILTER_SIZE - halo_width) * shared_width + (tx + halo_width)) * pixel_size + i] = d_input[((y + halo_width) * width + x) * pixel_size + i];
        }
    }

    __syncthreads();

    if (x < width && y < height)
    {
        float r = 0.0, g = 0.0, b = 0.0;
        for (int fy = 0; fy < FILTER_SIZE; fy++)
        {
            for (int fx = 0; fx < FILTER_SIZE; fx++)
            {
                int shared_idx = ((ty + fy) * shared_width + (tx + fx)) * pixel_size;
                int f_idx = fy * FILTER_SIZE + fx;
                r += shared_mem[shared_idx] * d_filter[f_idx];
                if (pixel_size > 1)
                {
                    g += shared_mem[shared_idx + 1] * d_filter[f_idx];
                    b += shared_mem[shared_idx + 2] * d_filter[f_idx];
                }
            }
        }

        int dest_idx = (y * width + x) * pixel_size;
        d_output[dest_idx] = min(max(int(r), 0), 255);
        if (pixel_size > 1)
        {
            d_output[dest_idx + 1] = min(max(int(g), 0), 255);
            d_output[dest_idx + 2] = min(max(int(b), 0), 255);
        }
    }
}

void processImageCUDA(PROCESSING_JOB *job)
{
    png_byte *d_input, *d_output;
    size_t size = job->width * job->height * job->pixel_size * sizeof(png_byte);
    float *d_filter;
    float *h_filter = getAlgoFilterByType(job->processing_algo);

    // Allocate device memory
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMalloc((void **)&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // Copy data to the device
    cudaMemcpy(d_input, job->source_raw, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((job->width + TILE_SIZE - 1) / TILE_SIZE, (job->height + TILE_SIZE - 1) / TILE_SIZE);
    size_t sharedMemSize = (TILE_SIZE + FILTER_SIZE - 1) * (TILE_SIZE + FILTER_SIZE - 1) * job->pixel_size * sizeof(png_byte);

    // Launch kernel
    applyFilterSharedv1<<<gridDim, blockDim, sharedMemSize>>>(d_filter, d_input, d_output, job->width, job->height, job->pixel_size);
    cudaGetLastError();

    // Copy result back to host
    cudaMemcpy(job->dest_raw, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    printf("Completed processImageCUDA for job: %s\n", job->source_name);
}
#define NUM_THREADS 4

void process_jobs_parallel(PROCESSING_JOB **jobs, int start_idx, int end_idx)
{
    for (int i = start_idx; i < end_idx; ++i)
    {
        if (jobs[i] != NULL)
        {
            processImageCUDA(jobs[i]); // Process each job using the GPU
        }
    }
}

void cleanup_shared_images()
{
    for (auto it = image_cache.begin(); it != image_cache.end(); ++it)
    {
        delete it->second->png_raw; // Free the PNG_RAW structure
        delete it->second;          // Free the SharedImageData structure
    }
    image_cache.clear();
}
void execute_jobs_gpu_parallel(PROCESSING_JOB **jobs)
{
    int num_jobs = 0;
    while (jobs[num_jobs] != NULL)
    {
        num_jobs++;
    }

    int jobs_per_thread = (num_jobs + NUM_THREADS - 1) / NUM_THREADS;
    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        int start_idx = i * jobs_per_thread;
        int end_idx = std::min(start_idx + jobs_per_thread, num_jobs);
        threads.emplace_back(process_jobs_parallel, jobs, start_idx, end_idx);
    }

    for (auto &t : threads)
    {
        t.join();
    }
}

void execute_jobs_gpu_sharedv1(PROCESSING_JOB **jobs)
{
    std::unordered_map<std::string, std::string> processed_jobs;

    for (int i = 0; jobs[i] != NULL; i++)
    {
        std::string job_key = std::string(jobs[i]->source_name) + "_" + algoToString(jobs[i]->processing_algo);

        if (processed_jobs.find(job_key) == processed_jobs.end())
        {
            processImageCUDA(jobs[i]);
            processed_jobs[job_key] = std::string(jobs[i]->dest_name);
        }
        else
        {
            // Reuse processed result
            strcpy(jobs[i]->dest_name, processed_jobs[job_key].c_str());
        }
    }
}
void execute_jobs_gpu(PROCESSING_JOB **jobs) {
    for (int i = 0; jobs[i] != NULL; i++) {
        PROCESSING_JOB *job = jobs[i];
        sharedProcessImageCUDA(job);
    }
}
void sharedProcessImageCUDA(PROCESSING_JOB *job) {
    // CUDA memory allocations and kernel calls
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    png_byte *d_In, *d_Out;
    size_t imageSize = job->width * job->height * 4 * sizeof(png_byte);
    cudaMalloc(&d_In, imageSize);
    cudaMalloc(&d_Out, imageSize);

    // Asynchronous memory copy
    cudaMemcpyAsync(d_In, job->source_raw, imageSize, cudaMemcpyHostToDevice, stream);

    // Kernel execution (assuming d_Filter is allocated and set properly)
    dim3 blockSize(16, 16);
    dim3 gridSize((job->width + blockSize.x - 1) / blockSize.x, (job->height + blockSize.y - 1) / blockSize.y);
    filterKernel<<<gridSize, blockSize, 0, stream>>>(d_In, d_Out, job->height, job->width, getAlgoFilterByType(job->processing_algo));

    // Asynchronous memory copy back to host
    cudaMemcpyAsync(job->dest_raw, d_Out, imageSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_In);
    cudaFree(d_Out);
}

// void execute_jobs_gpu_sharedv1(PROCESSING_JOB **jobs)
// {
//     std::unordered_map<std::string, std::string> processed_jobs;

//     for (int i = 0; jobs[i] != NULL; i++)
//     {
//         std::string job_key = std::string(jobs[i]->source_name) + "_" + algoToString(jobs[i]->processing_algo);

//         if (processed_jobs.find(job_key) == processed_jobs.end())
//         {
//             processImageCUDA(jobs[i]);
//             processed_jobs[job_key] = std::string(jobs[i]->dest_name);
//         }
//         else
//         {
//             // Reuse processed result
//             strcpy(jobs[i]->dest_name, processed_jobs[job_key].c_str());
//         }
//     }
// }