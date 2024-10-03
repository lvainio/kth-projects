#include <stdio.h> 
#include <stdlib.h>

#include "quick_sort_openmp.h"

// parallel_quicksort is a parallelized implementation of quicksort. 
void parallel_quicksort(Args* args) {
    int left = args->left;
    int right = args->right;
    int level = args->level;
    
    if(left >= right) {
        return;
    }

    int pivot = partition(args);
    Args args_left = {args->arr, left, pivot-1, level+1}; 
    Args args_right = {args->arr, pivot+1, right, level+1};

    // parallel
    if (level < MAX_LEVEL) {
        #pragma omp task 
        parallel_quicksort(&args_left); 
        #pragma omp task
        parallel_quicksort(&args_right);
        #pragma omp taskwait // this waits only for the two above tasks to finish (i.e. the tasks created within this task.)
        
    // sequential
    } else {
        parallel_quicksort(&args_left);
        parallel_quicksort(&args_right);
    }
}

// partition picks the last element of the subarray as pivot and places
// lower elements to the "left" and higher elements to the "right" of the pivot.
int partition(Args* args) {
    int left = args->left;
    int right = args->right;
    int pivot = args->arr[right];
    int i = left - 1;
    for(int j = left; j < right; j++) {
        if(args->arr[j] <= pivot) {
            i++;
            swap(args->arr, i, j);
        }
    }
    swap(args->arr, i + 1, right);
    return i + 1;
}

// swap swaps two elements of the array.
void swap(int* arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}


// ----- BENCHMARKING ----- //

// OBS! Time is given in seconds and the time presented is the median time from 5 runs!

// 1 thread
// n = 100 000:     0.00749929
// n = 1 000 000:   0.0887307
// n = 10 000 000:  1.01492   
// n = 100 000 000: 11.5067

// 2 threads
// n = 100 000:     0.00472332
// n = 1 000 000:   0.0608003
// n = 10 000 000:  0.624531
// n = 100 000 000: 7.56373

// 4 threads
// n = 100 000:     0.00376742
// n = 1 000 000:   0.0460531
// n = 10 000 000:  0.447437
// n = 100 000 000: 4.06882

// 8 threads
// n = 100 000:     0.00333566
// n = 1 000 000:   0.0363227
// n = 10 000 000:  0.241507
// n = 100 000 000: 3.29835

// 16 threads
// n = 100 000:     0.00286984
// n = 1 000 000:   0.024253
// n = 10 000 000:  0.218519
// n = 100 000 000: 2.78138