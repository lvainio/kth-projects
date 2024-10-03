#include <stdio.h> 
#include <stdlib.h>
#include <time.h>

int* arr;
int* arr_copy;
int n;

// generate_array fills nums with random integers ranging from 0 to RAND_MAX.
int* generate_array(int len) {
    n = len;
    srand(time(NULL));
    arr = (int*) malloc(n * sizeof(int));
    arr_copy = (int*) malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) {
        int num = rand();
        arr[i] = num;
        arr_copy[i] = num;
    }
    return arr;
}

// free_array frees the allocated memory.
void free_array() {
    free(arr);
    free(arr_copy);
}

// print_array prints out the contents of nums.
void print_array() {
    printf("arr: {");
    for (size_t i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i != n-1) {
            printf(", ");
        }
    }
    printf("}\n");
}

// print_array prints out the contents of nums.
void print_array_copy() {
    printf("arr_copy: {");
    for (size_t i = 0; i < n; i++) {
        printf("%d", arr_copy[i]);
        if (i != n-1) {
            printf(", ");
        }
    }
    printf("}\n");
}

// compare function for qsort().
int compare(const void * num1, const void * num2) {
    return (*(int*)num1 - *(int*)num2);
}

// test if array is sorted and contains the same elements as initial array.
void test() {
    qsort(arr_copy, n, sizeof(int), compare);
    if(n == 0) {
        printf("success\n");
        return;
    }
    for (int i = 1; i < n; i++) {
        if (arr[i-1] > arr[i]) {
            printf("fail\n");
            return;
        }
    }
    for (int i = 0; i < n; i++) {
        if(arr[i] != arr_copy[i]) {
            printf("fail\n");
            return;
        }
    }
    printf("success\n");
}