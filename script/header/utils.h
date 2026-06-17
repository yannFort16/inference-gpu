#ifndef UTILS_H
#define UTILS_H

void generateRandomMatrix(float *matrix, int rows, int cols, float max, bool whole);

void printMatrix(const float *matrix, int rows, int cols, int maxElements);

bool compare_matrix(const float *mat1, const float *mat2, int rows, int cols);

#endif