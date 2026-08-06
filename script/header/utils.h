#ifndef UTILS_H
#define UTILS_H

void generateRandomMatrix(float *matrix, int rows, int cols, float max, bool whole);

void printMatrix(const float *matrix, int rows, int cols, int maxElements);

bool compare_matrix(const float *mat1, const float *mat2, int rows, int cols);

int read_matrix(char * filename, float* dest_matrix, int x, int y, int c);

int write_matrix(char * filename, float* src_matrix, int x, int y, int c);
#endif