#include "header.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

extern void processImageCUDA(PROCESSING_JOB *job);

std::unordered_map<std::string, SharedImageData *> image_cache;

int count_lines(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening job file <%s>.\n", filename);
        exit(-1);
    }
    int lineCount = 0;
    char buffer[1024]; // Buffer to store each line
    while (fgets(buffer, sizeof(buffer), file) != NULL)
    {
        lineCount++;
    }
    fclose(file);
    return lineCount;
}

PROCESSING_JOB **prepare_jobs(char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return NULL;

    int job_count = count_lines(filename);
    PROCESSING_JOB **processing_job = (PROCESSING_JOB **)malloc(sizeof(PROCESSING_JOB *) * (job_count + 1));

    char input_filename[256];
    char processing_algo[256];
    char output_filename[256];

    int count = 0;
    while (fscanf(file, "%s %s %s", input_filename, processing_algo, output_filename) != EOF) {
        SharedImageData *shared_image_data;
        if (image_cache.find(input_filename) == image_cache.end()) {
            shared_image_data = (SharedImageData *)malloc(sizeof(SharedImageData));
            shared_image_data->png_raw = read_png(input_filename);
            shared_image_data->ref_count = 1;
            image_cache[input_filename] = shared_image_data;
        } else {
            shared_image_data = image_cache[input_filename];
            shared_image_data->ref_count++;
        }

        processing_job[count] = (PROCESSING_JOB *)malloc(sizeof(PROCESSING_JOB));
        processing_job[count]->source_name = strdup(input_filename);
        processing_job[count]->dest_name = strdup(output_filename);
        processing_job[count]->width = shared_image_data->png_raw->width;
        processing_job[count]->height = shared_image_data->png_raw->height;
        processing_job[count]->info_ptr = shared_image_data->png_raw->info_ptr;
        processing_job[count]->pixel_size = shared_image_data->png_raw->pixel_size;
        processing_job[count]->source_raw = shared_image_data->png_raw->buf;
        processing_job[count]->dest_raw = (png_byte *)malloc(shared_image_data->png_raw->width * shared_image_data->png_raw->height * shared_image_data->png_raw->pixel_size * sizeof(png_byte));

        clear_buf(processing_job[count]->dest_raw, shared_image_data->png_raw->height, shared_image_data->png_raw->width);
        processing_job[count]->processing_algo = getAlgoByName(processing_algo);
        count++;
    }
    processing_job[count] = NULL;
    fclose(file);

    return processing_job;
}
void write_jobs_output_files(PROCESSING_JOB **jobs)
{
    int count = 0;
    while (jobs[count] != NULL)
    {
        PNG_RAW *png_raw = (PNG_RAW *)malloc(sizeof(PNG_RAW));
        png_raw->height = jobs[count]->height;
        png_raw->width = jobs[count]->width;
        png_raw->pixel_size = jobs[count]->pixel_size;
        png_raw->buf = jobs[count]->dest_raw;
        png_raw->info_ptr = jobs[count]->info_ptr;
        printf("Writing output of job: %s -> %s\n", jobs[count]->source_name, jobs[count]->dest_name);
        count++;
    }
}
