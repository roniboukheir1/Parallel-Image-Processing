#include "header.h"

long long timeInMilliseconds(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}

int main(int argc, char **argv)
{

    PROCESSING_JOB **jobs = prepare_jobs(argv[1]);

    //35000 ms for cpu processing
    // long long start = timeInMilliseconds();
    // execute_jobs_cpu(jobs);
    // long long end = timeInMilliseconds();
    // printf("Execution time for CPU processing: %lld ms\n", end - start);


    //400 ms
    // long long start2 = timeInMilliseconds();
    // execute_jobs_gpu_parallel(jobs);
    // long long end2 = timeInMilliseconds();
    // printf("Execution time for GPU processing: %lld ms\n", end2 - start2);


    long long start3 = timeInMilliseconds();
    execute_jobs_gpu_sharedv1(jobs);
    long long end3 = timeInMilliseconds();
    // 300ms
    printf("Execution Time of parallel processing part with parallel processing: %lld ms\n", end3 - start3);
    write_jobs_output_files(jobs);
    return 0;
}