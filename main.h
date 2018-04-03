#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define _USE_OPENMP

#ifdef _USE_OPENMP
#include <omp.h>
#else
#warning Not using OpenMP! If you need OpenMP please define _USE_OPENMP macro in main.h or with -D
#endif

#include <math.h>
#include <sys/time.h>

#ifdef __gnu_linux__
#define _USE_RAND
#include <strings.h>
#endif

#endif
