
#ifndef SPATIAL_H_
#define SPATIAL_H_

#include "main.h"

//#define _LOG

// minimal radius
#define MIN_RAD 2

// maximal radius of rules
#define MAX_RAD 5

// maximal time gap size
#define MAX_GAP 10

// max iterations of GA
#define MAX_GA_ITER 100000

// if this is set to false, MAX_GA_ITER is not used and the algorithm can run forever!
#define LIMIT_ITER true

// the length of longest possible LUT
#define MAX_LEN (1 << (2 * MAX_RAD + 1))

// number of cells in a configuration
#define CELL_COUNT 69

// probability of applying a fix
#define _FIX_PROB 1.0

// multiplier for allowed fixes
#define _FIXES_MULT 1.0

// type to store cell state
#define _CONF_ENTRY int

// type of a configuration
#define _CONF _CONF_ENTRY *

// function that can get the minimum of a table with _CONF_ENTRY type
#define _CONF_MIN tool_min_i

#define _PRINT_ARRAY print_array_b

// number of rows in a test image (we assume it is constant)
#define TIME_STEPS 69

// number of complete cells in configuration below (<=) which we ignore a given row in observation
#define PARTIAL_TRESHOLD 0

// type to store LUT entries
#define _LUT_ENTRY bool
#define _DIST_ENTRY int

// type representing a CA, given by radius and the LUT
// note that the LUT is always initialized to hold the longest case
// for high radius values, this might be not optimal
typedef struct t_ca
{
    int radius;
    long num;
    _LUT_ENTRY lut[MAX_LEN];
} t_ca;

// probability of downscale mutation
#define _P_DOWN 0.001

// probability of upscale mutation
#define _P_UP 0.001

// probability of bit flip mutation
#define _P_FLIP 0.001

// population count
#define POP_COUNT (24 * 8)

// population elite count
#define ELITE_COUNT 24

#define MAX_BEST_AGE 250

#define MAX_NON_ELITE 5

#define SHIFT_MASK_FREQ 1

// statistics of an array
typedef struct t_stat
{
    double min;     // minimum value
    int min_idx;    // index of an element with minimum value
    double avg;     // average (mean) of values
    double max;     // maximum value
    int max_idx;    // index of an element with maximum value
    int min_errors; // minimum number of errors (obtained from the individual with max fit)
} t_stat;

// population of the GA
typedef struct t_pop
{
    t_ca rules[POP_COUNT];     // contents of population (CA rules)
    double fitness[POP_COUNT]; // fitness corresponding to population individuals
    int gaps[POP_COUNT][TIME_STEPS - 1];
    double dist[POP_COUNT];
    t_stat stats;
    int rank[POP_COUNT];
    int radius_count[MAX_RAD];
} t_pop;

typedef struct t_img
{
    _CONF_ENTRY data[TIME_STEPS][CELL_COUNT];
    bool mask[TIME_STEPS - 1];
    bool is_complete[TIME_STEPS];
    int mask_rows;
    int complete_cells; // number of complete cells in non-masked rows expect for IC
    int partial_cells;
} t_img;

typedef struct t_case
{
    t_img *img;
    int **gaps; // optional
    int count;
    bool *mask;
    int mask_count;
    t_ca *original_rule;
    int total_cells;
    int masked_cells;
    int partial_cells;
    int allowed_fixes; // PER ONE OBSERVATION
} t_case;

typedef enum {
    RANDOM = 1,
    CONSTANT_PER_IMAGE = 2,
    CONSTANT_PER_PROBLEM = 3
} e_generator_mode;

typedef enum {
    ALL,
    EVEN,
    ODD
} e_generator_gap_type;

typedef enum {
    OBSERVATION,
    SYSTEM,
    NONE
} e_generator_noise_type;

typedef struct t_generator_setup
{
    e_generator_mode mode;
    e_generator_gap_type type;
    double noise;
    e_generator_noise_type noise_type;
} t_generator_setup;

#endif
