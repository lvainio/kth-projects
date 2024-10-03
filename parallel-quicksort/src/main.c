/*
 * A parallelized implementation of the famous quicksort algorithm.
 * 
 * Author: Leo Vainio
 * 
 * Compile: gcc -o quicksort main.c quick_sort.c data.c -pthread
 * Run: ./quicksort <len>
 * 
 */

#include <stdbool.h>
#include <stdio.h> 
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "data.h"
#include "quick_sort.h"

// timer returns the current time, taken from matrix.c given in the course ID1217.
double read_timer() {
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if(!initialized) {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

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
    int* arr = generate_array(n);
    
    double start_time = read_timer();
    Args args = {arr, 0, n-1, 1};
    parallel_quicksort(&args);
    double end_time = read_timer();
    printf("The execution time is %g sec\n", end_time - start_time);

    free_array();
    return 0;
}