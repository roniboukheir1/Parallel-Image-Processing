#include "header.h"
#include <cuda_runtime.h>
#include <vector>
#include <thread>
#include <unordered_map>
#include <string>

// Define the filter size and tile size
#define FILTER_SIZE 5
#define TILE_SIZE 16
#define NUM_THREADS 4

__constant__ float d_filter[FILTER_SIZE * FILTER_SIZE];

void copyFilterToConstantMemory(float *h_filter) {
    cudaMemcpyToSymbol(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
}

std::string algoToString(PROCESSING_ALGO algo) {
    switch (algo) {
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

__global__ void applyFilterWithSharedMemory(png_byte *d_input, png_byte *d_output, int width, int height, int pixel_size) {
    extern __shared__ png_byte shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int shared_width = TILE_SIZE + FILTER_SIZE - 1;
    int halo_width = FILTER_SIZE / 2;

    int global_idx = (y * width + x) * pixel_size;

    // Load main tile into shared memory
    for (int i = 0; i < pixel_size; i++) {
        int shared_idx = ((ty + halo_width) * shared_width + (tx + halo_width)) * pixel_size + i;
        if (x < width && y < height) {
            shared_mem[shared_idx] = d_input[global_idx + i];
        } else {
            shared_mem[shared_idx] = 0;
        }
    }

    // Load halo elements into shared memory
    if (ty < halo_width) {
        for (int i = 0; i < pixel_size; i++) {
            int top_shared_idx = (ty * shared_width + (tx + halo_width)) * pixel_size + i;
            int top_global_idx = ((y - halo_width) * width + x) * pixel_size + i;
            int bottom_shared_idx = ((ty + TILE_SIZE) * shared_width + (tx + halo_width)) * pixel_size + i;
            int bottom_global_idx = ((y + TILE_SIZE) * width + x) * pixel_size + i;

            shared_mem[top_shared_idx] = (y >= halo_width) ? d_input[top_global_idx] : 0;
            shared_mem[bottom_shared_idx] = (y + TILE_SIZE < height) ? d_input[bottom_global_idx] : 0;
        }
    }

    if (tx < halo_width) {
        for (int i = 0; i < pixel_size; i++) {
            int left_shared_idx = ((ty + halo_width) * shared_width + tx) * pixel_size + i;
            int left_global_idx = (y * width + (x - halo_width)) * pixel_size + i;
            int right_shared_idx = ((ty + halo_width) * shared_width + (tx + TILE_SIZE)) * pixel_size + i;
            int right_global_idx = (y * width + (x + TILE_SIZE)) * pixel_size + i;

            shared_mem[left_shared_idx] = (x >= halo_width) ? d_input[left_global_idx] : 0;
            shared_mem[right_shared_idx] = (x + TILE_SIZE < width) ? d_input[right_global_idx] : 0;
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        for (int i = 0; i < pixel_size; i++) {
            float result = 0.0f;
            for (int ky = -halo_width; ky <= halo_width; ky++) {
                for (int kx = -halo_width; kx <= halo_width; kx++) {
                    int shared_idx = ((ty + ky + halo_width) * shared_width + (tx + kx + halo_width)) * pixel_size + i;
                    result += shared_mem[shared_idx] * d_filter[(ky + halo_width) * FILTER_SIZE + (kx + halo_width)];
                }
            }
            d_output[global_idx + i] = min(max(result, 0.0f), 255.0f);
        }
    }
}

void processImageCUDA(PROCESSING_JOB *job, png_byte *d_input, png_byte *d_output, cudaStream_t stream) {
    float *h_filter = getAlgoFilterByType(job->processing_algo);
    cudaMemcpyToSymbolAsync(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice, stream);

    int width = job->width;
    int height = job->height;
    int pixel_size = job->pixel_size;

    size_t image_size = width * height * pixel_size * sizeof(png_byte);

    cudaMemcpyAsync(d_input, job->source_raw, image_size, cudaMemcpyHostToDevice, stream);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    size_t shared_mem_size = (TILE_SIZE + FILTER_SIZE - 1) * (TILE_SIZE + FILTER_SIZE - 1) * pixel_size * sizeof(png_byte);

    applyFilterWithSharedMemory<<<gridDim, blockDim, shared_mem_size, stream>>>(d_input, d_output, width, height, pixel_size);

    cudaMemcpyAsync(job->dest_raw, d_output, image_size, cudaMemcpyDeviceToHost, stream);
}

void process_jobs_parallel(PROCESSING_JOB **jobs, int start_idx, int end_idx) {
    int width = jobs[0]->width; // Assuming all jobs have the same dimensions
    int height = jobs[0]->height;
    int pixel_size = jobs[0]->pixel_size;
    size_t image_size = width * height * pixel_size * sizeof(png_byte);

    png_byte *d_input, *d_output;
    cudaMalloc((void **)&d_input, image_size);
    cudaMalloc((void **)&d_output, image_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = start_idx; i < end_idx; i++) {
        processImageCUDA(jobs[i], d_input, d_output, stream);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
}

void execute_jobs_gpu_parallel(PROCESSING_JOB **jobs) {
    int num_jobs = 0;
    while (jobs[num_jobs] != NULL) {
        num_jobs++;
    }

    int jobs_per_thread = (num_jobs + NUM_THREADS - 1) / NUM_THREADS;
    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_THREADS; ++i) {
        int start_idx = i * jobs_per_thread;
        int end_idx = std::min(start_idx + jobs_per_thread, num_jobs);
        threads.emplace_back(process_jobs_parallel, jobs, start_idx, end_idx);
    }

    for (auto &t : threads) {
        t.join();
    }
}

void execute_jobs_gpu_sharedv1(PROCESSING_JOB **jobs) {
    std::unordered_map<std::string, std::string> processed_jobs;

    // Allocate device memory once and reuse
    int width = jobs[0]->width;
    int height = jobs[0]->height;
    int pixel_size = jobs[0]->pixel_size;
    size_t image_size = width * height * pixel_size * sizeof(png_byte);

    png_byte *d_input, *d_output;
    cudaMalloc((void **)&d_input, image_size);
    cudaMalloc((void **)&d_output, image_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; jobs[i] != NULL; i++) {
        std::string job_key = std::string(jobs[i]->source_name) + "_" + algoToString(jobs[i]->processing_algo);

        if (processed_jobs.find(job_key) == processed_jobs.end()) {
            processImageCUDA(jobs[i], d_input, d_output, stream);
            processed_jobs[job_key] = std::string(jobs[i]->dest_name);
        } else {
            strcpy(jobs[i]->dest_name, processed_jobs[job_key].c_str());
        }
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
}
