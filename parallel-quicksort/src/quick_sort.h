#ifndef QUICK_SORT_H
#define QUICK_SORT_H

#define MAX_LEVEL 16

typedef struct {
    int* arr;
    int left;
    int right;
    int level;
} Args;

void* parallel_quicksort(void*);
void quick_sort(Args*);
int partition(Args*);
void swap(int*, int, int);

#endif