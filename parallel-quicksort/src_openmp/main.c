/*
 * A parallelized implementation of the famous quicksort algorithm.
 * 
 * Author: Leo Vainio
 * 
 * Compile: gcc -o quicksort main.c quicksort_openmp.c data.c -fopenmp -O
 * Run: ./quicksort <len>
 * 
 */

#include <omp.h>
#include <stdio.h> 
#include <stdlib.h>

#include "data.h"
#include "quick_sort_openmp.h"

// main
int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Invalid argument, usage: ./quicksort <len>\n");
        return 0;
    }
    int n = atoi(argv[1]);
    if (n < 0) {
        printf("Invalid length of array, can't have negative length\n");
        return 0;
    }

    omp_set_num_threads(NUM_THREADS);
    int* arr = generate_array(n);
    double start_time = omp_get_wtime();
    Args args = {arr, 0, n-1, 1};

#pragma omp parallel
#pragma omp single
    parallel_quicksort(&args);

    double end_time = omp_get_wtime();
    printf("The execution time is %g sec\n", end_time - start_time);

    free_array();
    return 0;
}