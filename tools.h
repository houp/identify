#ifndef TOOLS_H_
#define TOOLS_H_

#include "main.h"

double tool_get_time();
double tool_cut(double x);
void print_array(double c[], int l);
void print_array_b(bool c[], int l);
void print_array_2d(bool **c, int xlen, int ylen);
void print_array_2d_d(double **c, int xlen, int ylen);
void tool_free_array_d(double **a, int xlen);
void tool_free_array_b(bool **a, int xlen);
bool **tool_alloc_array_b(int xlen, int ylen);
double **tool_alloc_array_d(int xlen, int ylen);
void tool_array_bool_to_double(double *res, bool *src, int len);
void tool_round(double **a, int xlen, int ylen);
double tool_rand();
void tool_fill_random(double **a, int xlen, int ylen, bool deterministic);
double tool_sum(double **a, int xlen, int ylen);
int tool_foldedSize(int s, int r);
void tool_fold(bool *f_res, bool *u_src, int u_len, int r);
void tool_unfold(bool u_res[], bool f_src[], int f_len, int r);
void tool_add(double **res, bool **src, int xlen, int ylen);
void tool_mult(double **a, double c, int xlen, int ylen);
void tool_abs_subst(double **res, bool **a, double **b, int xlen, int ylen);
double tool_1d_max(double *a, int len);
void print_array_i(int c[], int l);
int tool_differences(bool **a, bool **b, int xlen, int ylen);
void tool_fill_random_b(bool **a, int xlen, int ylen);
void tool_init_random();
void tool_fill_random1(double *a, int xlen, bool deterministic);
int tool_rand_i(int min, int max);
void tool_fill_random1ui(unsigned int *a, int xlen, int min, int max);
void tool_fold_d(double *f_res, double *u_src, int u_len, int r);
void tool_unfold_d(double u_res[], double f_src[], int f_len, int r);
int tool_cap(int a, int min, int max);
void tool_fill_random_ui(unsigned int **a, int xlen, int ylen, int min, int max);
unsigned int **tool_alloc_array_ui(int xlen, int ylen);
void tool_free_array_ui(unsigned int **a, int xlen);
void print_array_ui(unsigned int c[], int l);
double tool_rand_gauss(double sigma);
void tool_range(int *result, int start, int len);
int *tool_alloc_range(int start, int len);
int *tool_sort_fitness_index(int *index, double *fitness, int count);
void tool_fill_random_d(double **a, int xlen, int ylen);
void tool_fake_random();
bool tool_row_cmp(bool *r1, bool *r2, int len);
double tool_avg(double *a, int len);

double tool_max(double *a, int len);
int tool_max_i(int *a, int len);

double tool_min(double *a, int len);
int tool_min_i(int *a, int len);

double tool_max_ix(double *a, int len, int *id);
double tool_min_ix(double *a, int len, int *id);

double tool_1d_sum(double *a, int len);
bool tool_compar_b(bool *a, bool *b, int len);
bool tool_compar_d(double *a, double *b, int len);

void tool_fill_random1i(int *a, int xlen, int min, int max);
void tool_fill_random1b(bool *a, int xlen);
int *tool_sort_index_int(int *index, int *array, int count);

#endif
