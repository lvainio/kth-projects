/* 
  matrix summation using OpenMP

  usage with gcc:
    gcc -O -fopenmp -o out matrix_sum_openmp.c 
    ./out <size> <num_workers>
*/

#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SIZE 25000  // maximum matrix size 
#define MAX_WORKERS 16  // maximum number of workers
#define MAX_VALUE 1000  // maximum value of element

int matrix[MAX_SIZE][MAX_SIZE];
int size;

// init_matrix fills the matrix with random values.
void init_matrix() {
  srand(time(NULL));
  for (int row = 0; row < size; row++)
	  for (int col = 0; col < size; col++)
      matrix[row][col] = rand() % MAX_VALUE;
}

// print_matrix print the contents of matrix to stdout.
void print_matrix() {
  printf("matrix:\n");
  for (int row = 0; row < size; row++) {
	  for (int col = 0; col < size; col++) {
      printf("%4d ", matrix[row][col]);
	  }
    printf("\n");
  }
  printf("\n");
}

// test calculates the sum sequentially (just for testing purpose).
void test() {
  long total = 0;
  for (int row = 0; row < size; row++)
	  for (int col = 0; col < size; col++)
      total += matrix[row][col];
  printf("sum (test): %ld\n", total);
}

typedef struct {
  int value;
  int row;
  int col;  
} Elem;

// main. read command line, initialize, and create threads.
int main(int argc, char *argv[]) {
  // command line args
  size = (argc > 1) ? atoi(argv[1]) : MAX_SIZE;
  int num_workers = (argc > 2) ? atoi(argv[2]) : MAX_WORKERS;
  if (size > MAX_SIZE) {
    size = MAX_SIZE;
  }
  if (num_workers > MAX_WORKERS) {
    num_workers = MAX_WORKERS;
  }

  // init
  init_matrix();

  omp_set_num_threads(num_workers);
  double start_time = omp_get_wtime();

  // sum, min, max
  long total = 0;
  int row, col;

  // omp_in is the partial result local to each thread, omp_out references final value after reduction. omp_priv is the initial value of private.
#pragma omp declare reduction(mini : Elem : omp_out = omp_in.value < omp_out.value ? omp_in : omp_out) initializer(omp_priv = {INT_MAX, -1, -1})
  Elem min = {INT_MAX, -1, -1};
#pragma omp declare reduction(maxi : Elem : omp_out = omp_in.value > omp_out.value ? omp_in : omp_out) initializer(omp_priv = {INT_MIN, -1, -1})
  Elem max = {INT_MIN, -1, -1};

#pragma omp parallel for reduction(+:total) reduction(mini:min) reduction(maxi:max) private(col)
  for (row = 0; row < size; row++) {
    for (col = 0; col < size; col++){
      total += matrix[row][col];
      if (matrix[row][col] < min.value) {
        min.value = matrix[row][col];
        min.row = row;
        min.col = col;
      }
      if (matrix[row][col] > max.value) {
        max.value = matrix[row][col];
        max.row = row;
        max.col = col;
      }
    }
  }

  // result
  double end_time = omp_get_wtime();
  printf("total sum: %ld\n", total);
  printf("min value: %d, row: %d, col: %d\n", min.value, min.row, min.col);
  printf("max value: %d, row: %d, col: %d\n", max.value, max.row, max.col);
  printf("it took: %g seconds\n", end_time - start_time);

  return 0;
}