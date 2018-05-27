#ifndef _FITNESS_LIB_H
#define _FITNESS_LIB_H

#include "spatial.h"

t_ca* ca_new_lut(_LUT_ENTRY lut[], int radius);
extern void lib_init(_LUT_ENTRY lut[], int radius, int img_count, double prob);
extern double lib_find_fitness(_LUT_ENTRY lut[], int radius);
extern void lib_echo();

#endif
