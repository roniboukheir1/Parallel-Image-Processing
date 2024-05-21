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

    // add other processing cases to main
    // long long start = timeInMilliseconds();
    // execute_jobs_cpu(jobs);
    // long long end = timeInMilliseconds();

    // // average execution time of processing part on CPU = 22800 ms
    // printf("Execution Time of processing part: %lld ms\n", end - start);
    // long long start2 = timeInMilliseconds();
    // execute_jobs_gpu(jobs);
    // long long end2 = timeInMilliseconds();
    // // average execution time of processing part on GPU = 2000 ms improved by 11 times
    // printf("Execution Time of parallel processing part: %lld ms\n", end2 - start2);

    long long start3 = timeInMilliseconds();
    execute_jobs_gpu_shared(jobs);
    long long end3 = timeInMilliseconds();
    // average execution time of processing part on GPU with shared memory = 190 ms improved by 118 times
    printf("Execution Time of parallel processing part with shared memory: %lld ms\n", end3 - start3);

    write_jobs_output_files(jobs);
    return 0;
}