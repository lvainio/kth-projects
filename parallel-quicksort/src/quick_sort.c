#include <pthread.h>
#include <stdio.h> 
#include <stdlib.h>

#include "quick_sort.h"

// parallel_quicksort is a parallelized implementation of quicksort. 
void* parallel_quicksort(void* arg) {
    Args *args = arg;
    int left = args->left;
    int right = args->right;
    int level = args->level;
    
    if(left >= right) {
        return NULL;
    }

    int pivot = partition(args);
    Args args_left = {args->arr, left, pivot-1, level+1}; 
    Args args_right = {args->arr, pivot+1, right, level+1};

    if (level < MAX_LEVEL) {
        pthread_t t_left;
        pthread_t t_right;

        if(pthread_create(&t_left, NULL, parallel_quicksort, &args_left) != 0)
            fprintf(stderr, "Creating thread failed");
        if(pthread_create(&t_right, NULL, parallel_quicksort, &args_right) != 0)
            fprintf(stderr, "Creating thread failed");

        if(pthread_join(t_left, NULL) != 0)
            fprintf(stderr, "Joining thread failed");
        if(pthread_join(t_right, NULL) != 0)
            fprintf(stderr, "Joining thread failed");

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

// ----- BENCHMARKS (time given in seconds) ----- //

// fully sequential (MAX_LEVEL = 1):
// n = 100 000:         0.011607, 0.011925, 0.011705 
// n = 1 000 000:       0.143727, 0.145697, 0.178966 
// n = 10 000 000:      2.01987, 2.12878, 2.20362
// n = 100 000 000:     23.065, 23.2824, 22.9335

// parallel implementation (MAX_LEVEL = 2):
// n = 100 000:         0.008313, 0.008726, 0.008704
// n = 1 000 000:       0.158295, 0.09582, 0.173933
// n = 10 000 000:      1.15495, 1.62315, 1.72717
// n = 100 000 000:     17.589, 16.6441, 20.2575

// parallel implementation (MAX_LEVEL = 4):
// n = 100 000:         0.00794, 0.007987, 0.004754
// n = 1 000 000:       0.060891, 0.099068, 0.098563
// n = 10 000 000:      0.826437, 1.36594, 0.968509
// n = 100 000 000:     14.9867, 8.54499, 6.3723

// parallel implementation (MAX_LEVEL = 8):
// n = 100 000:         0.008933, 0.009584, 0.009171    
// n = 1 000 000:       0.043686, 0.047184, 0.037065
// n = 10 000 000:      0.388953, 0.560401, 0.414454
// n = 100 000 000:     7.84743, 6.04248, 7.15438

// parallel implementation (MAX_LEVEL = 12):
// n = 100 000:         0.104786, 0.080181, 0.090653
// n = 1 000 000:       0.130178, 0.116954, 0.126627
// n = 10 000 000:      0.689073, 0.457411, 0.472586
// n = 100 000 000:     4.8749, 4.44355, 4.59349

// parallel implementation (MAX_LEVEL = 16):
// n = 100 000:         0.561164, 0.599422, 0.572329
// n = 1 000 000:       1.18521, 1.08798, 1.09963
// n = 10 000 000:      1.62332, 1.53492, 1.80322
// n = 100 000 000:     5.14809, 5.08067, 5.22745