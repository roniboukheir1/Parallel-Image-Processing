#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <string>

typedef struct
{
    int height;
    int width;
    int pixel_size;
    png_infop info_ptr;
    png_byte *buf;
} PNG_RAW;

struct SharedImageData
{
    PNG_RAW *png_raw;
    int ref_count;
};

extern std::unordered_map<std::string, SharedImageData *> image_cache;

extern int count_lines(char *filename);
extern void cleanup_shared_images();

typedef enum
{
    SHARPEN,
    EDGE,
    BLUR,
    UNKNOWN
} PROCESSING_ALGO;

typedef struct
{
    char *source_name;
    char *dest_name;
    int width;
    int height;
    int pixel_size;
    png_infop info_ptr;
    png_byte *source_raw;
    png_byte *dest_raw;
    PROCESSING_ALGO processing_algo;
} PROCESSING_JOB;

PNG_RAW *read_png(char *file_name);
void write_png(char *file_name, PNG_RAW *png_raw);
void clear_buf(png_byte *buf, int height, int width);
PROCESSING_JOB **prepare_jobs(char *filename);
void write_jobs_output_files(PROCESSING_JOB **jobs);
void execute_jobs_cpu(PROCESSING_JOB **jobs);          // jobs NULL terminated
void execute_jobs_gpu_sharedv1(PROCESSING_JOB **jobs); // jobs NULL terminated
void execute_jobs_gpu_parallel(PROCESSING_JOB **jobs); // jobs NULL terminated
void sharedProcessImageCUDA(PROCESSING_JOB *job);
PROCESSING_ALGO getAlgoByName(char *algo_name);
float *getAlgoFilterByType(PROCESSING_ALGO algo);
char *getStrAlgoFilterByType(PROCESSING_ALGO algo);
void process_jobs_parallel(PROCESSING_JOB **jobs, int start_idx, int end_idx);
float min1(float a, float b);
float max1(float a, float b);
void PictureHost_FILTER(png_byte *h_In, png_byte *h_Out, int h, int w, float *h_filt);

#endif // HEADER_H
