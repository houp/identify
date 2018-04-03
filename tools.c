#include "tools.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

gsl_rng **_random = NULL;

double tool_get_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

double tool_cut(double x)
{
    if (x <= 0.0)
        return 0.0;
    if (x >= 1.0)
        return 1.0;
    return x;
}

void print_array(double c[], int l)
{
    for (int i = 0; i < l - 1; i++)
    {
        printf("%.3f ", c[i]);
    }
    printf("%.3f\n", c[l - 1]);
}

void print_array_b(bool c[], int l)
{
    for (int i = 0; i < l - 1; i++)
    {
        printf("%d,", c[i] ? 1 : 0);
    }
    printf("%d\n", c[l - 1] ? 1 : 0);
}

void print_array_i(int c[], int l)
{
    for (int i = 0; i < l - 1; i++)
    {
        printf("%d,", c[i]);
    }
    printf("%d\n", c[l - 1]);
}

void print_array_ui(unsigned int c[], int l)
{
    for (int i = 0; i < l - 1; i++)
    {
        printf("%u,", c[i]);
    }
    printf("%u\n", c[l - 1]);
}

void print_array_2d(bool **c, int xlen, int ylen)
{
    for (int x = 0; x < xlen; x++)
    {
        for (int y = 0; y < ylen; y++)
        {
            printf("%d,", c[x][y]);
        }
        puts("");
    }
}

void print_array_2d_d(double **c, int xlen, int ylen)
{
    for (int x = 0; x < xlen; x++)
    {
        for (int y = 0; y < ylen; y++)
        {
            printf("%.2f,", c[x][y]);
        }
        puts("");
    }
}

void tool_free_array_d(double **a, int xlen)
{
    for (int x = 0; x < xlen; x++)
    {
        free(a[x]);
    }
    free(a);
}

void tool_free_array_b(bool **a, int xlen)
{
    for (int x = 0; x < xlen; x++)
    {
        free(a[x]);
    }
    free(a);
}

void tool_free_array_ui(unsigned int **a, int xlen)
{
    for (int x = 0; x < xlen; x++)
    {
        free(a[x]);
    }
    free(a);
}

bool **tool_alloc_array_b(int xlen, int ylen)
{

    bool **a = (bool **)malloc(xlen * sizeof(bool *));
    for (int x = 0; x < xlen; x++)
    {
        a[x] = (bool *)calloc((size_t)ylen, sizeof(bool));
    }
    return a;
}

unsigned int **tool_alloc_array_ui(int xlen, int ylen)
{

    unsigned int **a = (unsigned int **)malloc(xlen * sizeof(unsigned int *));
    for (int x = 0; x < xlen; x++)
    {
        a[x] = (unsigned int *)calloc((size_t)ylen, sizeof(unsigned int));
    }
    return a;
}

double **tool_alloc_array_d(int xlen, int ylen)
{
    double **a = (double **)malloc(xlen * sizeof(double *));
    for (int x = 0; x < xlen; x++)
    {
        a[x] = (double *)calloc((size_t)ylen, sizeof(double));
    }
    return a;
}

void tool_array_bool_to_double(double *res, bool *src, int len)
{
    for (int i = 0; i < len; i++)
    {
        res[i] = src[i] ? 1.0 : 0.0;
    }
}

void tool_round(double **a, int xlen, int ylen)
{
    for (int x = 0; x < xlen; x++)
        for (int y = 0; y < ylen; y++)
            a[x][y] = round(a[x][y]);
}

static int tool_current_thread()
{
#ifdef _USE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

double tool_rand()
{
    return gsl_rng_uniform(_random[tool_current_thread()]);
}

/*
 * Returns random int from [min, max-1]
 */
int tool_rand_i(int min, int max)
{
    return min + gsl_rng_uniform_int(_random[tool_current_thread()], max - min);
}

void tool_fill_random1(double *a, int xlen, bool deterministic)
{
    for (int x = 0; x < xlen; x++)
    {
        a[x] = tool_rand();
        if (deterministic)
        {
            a[x] = a[x] > 0.5 ? 1 : 0;
        }
    }
}

void tool_fill_random1ui(unsigned int *a, int xlen, int min, int max)
{
    for (int x = 0; x < xlen; x++)
    {
        a[x] = (unsigned int)tool_rand_i(min, max);
    }
}

void tool_fill_random1i(int *a, int xlen, int min, int max)
{
    for (int x = 0; x < xlen; x++)
    {
        a[x] = tool_rand_i(min, max);
    }
}

void tool_fill_random1b(bool *a, int xlen)
{
    for (int x = 0; x < xlen; x++)
    {
        a[x] = tool_rand() < 0.5;
    }
}

void tool_fill_random(double **a, int xlen, int ylen, bool deterministic)
{
    for (int x = 0; x < xlen; x++)
    {
        for (int y = 0; y < ylen; y++)
        {
            a[x][y] = tool_rand();
            if (deterministic)
            {
                a[x][y] = a[x][y] > 0.5 ? 1 : 0;
            }
        }
    }
}

void tool_fill_random_d(double **a, int xlen, int ylen)
{
    for (int x = 0; x < xlen; x++)
    {
        for (int y = 0; y < ylen; y++)
        {
            a[x][y] = tool_rand(); // < 0.5 ? 1 : 0;
        }
    }
}

void tool_fill_random_b(bool **a, int xlen, int ylen)
{
    for (int x = 0; x < xlen; x++)
    {
        double p = tool_rand();
        for (int y = 0; y < ylen; y++)
        {
            a[x][y] = tool_rand() <= p;
        }
    }
}

void tool_fill_random_ui(unsigned int **a, int xlen, int ylen, int min, int max)
{
    for (int x = 0; x < xlen; x++)
    {
        for (int y = 0; y < ylen; y++)
        {
            a[x][y] = (unsigned int)tool_rand_i(min, max);
        }
    }
}

double tool_sum(double **a, int xlen, int ylen)
{
    double res = 0;
    for (int x = 0; x < xlen; x++)
        for (int y = 0; y < ylen; y++)
            res += a[x][y];
    return res;
}

int tool_foldedSize(int s, int r)
{
    return s + 2 * r;
}

void tool_fold(bool *f_res, bool *u_src, int u_len, int r)
{
    memcpy(f_res + r, u_src, (size_t)u_len * sizeof(bool));
    memcpy(f_res, u_src + u_len - r, (size_t)r * sizeof(bool));
    memcpy(f_res + r + u_len, u_src, (size_t)r * sizeof(bool));
}

void tool_fold_d(double *f_res, double *u_src, int u_len, int r)
{
    memcpy(f_res + r, u_src, (size_t)u_len * sizeof(double));
    memcpy(f_res, u_src + u_len - r, (size_t)r * sizeof(double));
    memcpy(f_res + r + u_len, u_src, (size_t)r * sizeof(double));
}

void tool_unfold(bool u_res[], bool f_src[], int f_len, int r)
{
    memcpy(u_res, f_src + r, (size_t)(f_len - 2 * r) * sizeof(bool));
}

void tool_unfold_d(double u_res[], double f_src[], int f_len, int r)
{
    memcpy(u_res, f_src + r, (size_t)(f_len - 2 * r) * sizeof(double));
}

void tool_add(double **res, bool **src, int xlen, int ylen)
{
    for (int x = 0; x < xlen; x++)
        for (int y = 0; y < ylen; y++)
            res[x][y] += src[x][y] ? 1.0 : 0.0;
}

void tool_mult(double **a, double c, int xlen, int ylen)
{
    for (int x = 0; x < xlen; x++)
        for (int y = 0; y < ylen; y++)
            a[x][y] *= c;
}

void tool_abs_subst(double **res, bool **a, double **b, int xlen, int ylen)
{
    for (int x = 0; x < xlen; x++)
        for (int y = 0; y < ylen; y++)
        {
            res[x][y] = fabs((a[x][y] ? 1.0 : 0.0) - b[x][y]);
        }
}

double tool_1d_max(double *a, int len)
{
    double max = 0.0;
    for (int i = 0; i < len; i++)
    {
        if (a[i] > max)
            max = a[i];
    }
    return max;
}

int tool_differences(bool **a, bool **b, int xlen, int ylen)
{
    int result = 0;
    for (int x = 0; x < xlen; x++)
    {
        for (int y = 0; y < ylen; y++)
        {
            if (a[x][y] && !b[x][y])
                result++;
            if (!a[x][y] && b[x][y])
                result++;
        }
    }
    return result;
}

static int tool_num_threads()
{
#ifdef _USE_OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

void tool_fake_random()
{
    int threads = 0;
#ifdef _USE_OPENMP
#pragma omp parallel
#pragma omp master
#endif
    {
        threads = tool_num_threads();
        const gsl_rng_type *T;
        gsl_rng_env_setup();
        T = gsl_rng_default;
        _random = (gsl_rng **)malloc(threads * sizeof(gsl_rng *));
        for (int i = 0; i < threads; i++)
        {
            _random[i] = gsl_rng_alloc(T);
            gsl_rng_set(_random[i], 1234567890);
        }
    }
}

void tool_init_random()
{
    int threads = 0;
#ifdef _USE_OPENMP
#pragma omp parallel
#pragma omp master
#endif
    {
        if (_random == NULL)
        {
            threads = tool_num_threads();

#ifndef _USE_RAND
#ifdef __APPLE__
            srandomdev();
#else
            srandom((unsigned int)tool_get_time());
#endif
#else
            srand((unsigned int)tool_get_time());
#endif

            const gsl_rng_type *T;

            gsl_rng_env_setup();

            T = gsl_rng_default;

            _random = (gsl_rng **)malloc(threads * sizeof(gsl_rng *));
            for (int i = 0; i < threads; i++)
            {
                _random[i] = gsl_rng_alloc(T);
                unsigned long int seed;
#ifdef _USE_RAND
                seed = (unsigned long int)rand();
#else
                seed = (unsigned long int)random();
#endif
                gsl_rng_set(_random[i], seed);
            }
        }
    }
}

double tool_rand_gauss(double sigma)
{
    return gsl_ran_gaussian_ziggurat(_random[tool_current_thread()], sigma);
}

int tool_cap(int a, int min, int max)
{
    if (a > max)
        return max;
    if (a < min)
        return min;
    return a;
}

void tool_range(int *result, int start, int len)
{
    for (int i = start; i < start + len; i++)
    {
        result[i - start] = i;
    }
}

int *tool_alloc_range(int start, int len)
{
    int *a = (int *)malloc(sizeof(int) * len);
    tool_range(a, start, len);
    return a;
}

#ifdef __APPLE__
int _compr(void *f, const void *a, const void *b)
{
#else
int _compr(const void *a, const void *b, void *f)
{
#endif
    double *fitness = (double *)f;
    int i1 = (int)(*(int *)a);
    int i2 = (int)(*(int *)b);
    double diff = fitness[i1] - fitness[i2];
    if (diff == 0.0)
        return 0;
    if (diff > 0.0)
        return 1;
    return -1;
}

int *tool_sort_fitness_index(int *index, double *fitness, int count)
{
#ifdef __APPLE__
    qsort_r(index, count, sizeof(int), fitness, _compr);
#else
    qsort_r(index, count, sizeof(int), _compr, fitness);
#endif
    return index;
}

#ifdef __APPLE__
int _compr_int(void *f, const void *a, const void *b)
{
#else
int _compr_int(const void *a, const void *b, void *f)
{
#endif
    int *fitness = (int *)f;
    int i1 = (int)(*(int *)a);
    int i2 = (int)(*(int *)b);
    int diff = fitness[i1] - fitness[i2];
    if (diff == 0)
        return 0;
    if (diff > 0)
        return 1;
    return -1;
}

int *tool_sort_index_int(int *index, int *array, int count)
{
#ifdef __APPLE__
    qsort_r(index, count, sizeof(int), array, _compr_int);
#else
    qsort_r(index, count, sizeof(int), _compr_int, array);
#endif
    return index;
}

bool tool_row_cmp(bool *r1, bool *r2, int len)
{
    for (int i = 0; i < len; i++)
    {
        if (r1[i] != r2[i])
            return false;
    }
    return true;
}

double tool_avg(double *a, int len)
{
    double result = 0;
    for (int i = 0; i < len; i++)
        result += a[i];
    return result / len;
}

double tool_max(double *a, int len)
{
    double result = a[0];
    for (int i = 1; i < len; i++)
    {
        if (result < a[i])
            result = a[i];
    }
    return result;
}

int tool_max_i(int *a, int len)
{
    int result = a[0];
    for (int i = 1; i < len; i++)
    {
        if (result < a[i])
            result = a[i];
    }
    return result;
}

double tool_min(double *a, int len)
{
    double result = a[0];
    for (int i = 1; i < len; i++)
    {
        if (result > a[i])
            result = a[i];
    }
    return result;
}

int tool_min_i(int *a, int len)
{
    int result = a[0];
    for (int i = 1; i < len; i++)
    {
        if (result > a[i])
            result = a[i];
    }
    return result;
}

double tool_max_ix(double *a, int len, int *id)
{
    double result = a[0];
    (*id) = 0;
    for (int i = 1; i < len; i++)
    {
        if (result < a[i])
        {
            result = a[i];
            (*id) = i;
        }
    }
    return result;
}

double tool_min_ix(double *a, int len, int *id)
{
    double result = a[0];
    (*id) = 0;
    for (int i = 1; i < len; i++)
    {
        if (result > a[i])
        {
            result = a[i];
            (*id) = i;
        }
    }
    return result;
}

double tool_1d_sum(double *a, int len)
{
    double result = 0;
    for (int i = 0; i < len; i++)
    {
        result += a[i];
    }
    return result;
}

bool tool_compar_b(bool *a, bool *b, int len)
{
    for (int i = 0; i < len; i++)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

bool tool_compar_d(double *a, double *b, int len)
{
    for (int i = 0; i < len; i++)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}
