#ifndef QUICK_SORT_H
#define QUICK_SORT_H

#define MAX_LEVEL 8
#define NUM_THREADS 16

typedef struct {
    int* arr;
    int left;
    int right;
    int level;
} Args;

void parallel_quicksort(Args*);
void quick_sort(Args*);
int partition(Args*);
void swap(int*, int, int);

#endif