#include "spatial.h"
#include "tools.h"

#include <sys/stat.h>
#include <errno.h>

/**
 * Returns the effective length of the LUT for given radius
 */
int ca_get_lut_len(t_ca *rule)
{
    return 1 << (2 * rule->radius + 1);
}

/*
 * Converts rule number to LUT
 * rule - target CA
 * ruleNum - number to convert
 * lutLen - length of the target LUT (should be equal to ca_get_lut_let(rule))
 */
void ca_decodeRule(t_ca *rule, long ruleNum, int lutLen)
{
    for (int i = 0; i < lutLen; i++)
    {
        rule->lut[i] = ((ruleNum >> i) & 1) == 1;
    }
}

/**
 * Copy src CA to res. Both pointers need to be non-null!
 */
void ca_copy(t_ca *res, t_ca *src)
{
    res->radius = src->radius;
    memcpy(res->lut, src->lut, MAX_LEN * sizeof(_LUT_ENTRY));
}

/**
 * Initializes new CA structure
 */
t_ca *ca_new(int radius)
{
    t_ca *result = calloc(1, sizeof(t_ca));
    result->radius = radius;
    return result;
}

/**
 * Frees a CA structure
 */
void ca_free(t_ca *ca)
{
    if (ca != NULL)
    {
        free(ca);
    }
}

/**
 * Fills the LUT with random values. Good for testing :)
 */
void ca_init_random(t_ca *rule)
{
    int l = ca_get_lut_len(rule);
    for (int i = 0; i < l; i++)
    {
        rule->lut[i] = tool_rand() > 0.5;
    }
}

/**
 * Downscale the radius of the LUT by one
 * If check == true, downscale will work only if rule->radius > MIN_RAD
 * If check == false, downscale will work if rule->radius > 1
 */
t_ca *ca_downscale_lut_check(t_ca *rule, bool check)
{
    if ((check && rule->radius > MIN_RAD) || (!check && rule->radius > 1))
    {
        int lenBig = ca_get_lut_len(rule);
        rule->radius = rule->radius - 1;
        int l = ca_get_lut_len(rule);

        int blackCount[l];
        memset(blackCount, 0, sizeof(int) * l);

        for (int i = 0; i < lenBig; i++)
        {
            if (rule->lut[i])
            {
                blackCount[(i / 2) % l]++;
            }
        }

        for (int i = 0; i < l; i++)
        {
            if (blackCount[i] > 2)
            {
                rule->lut[i] = true;
            }
            else if (blackCount[i] == 2)
            {
                rule->lut[i] = tool_rand() > 0.5;
            }
            else
            {
                rule->lut[i] = false;
            }
        }
    }
    return rule;
}

/**
 * Shorthand for ca_downscale_lut_check(rule, true)
 */
t_ca *ca_downscale_lut(t_ca *rule)
{
    return ca_downscale_lut_check(rule, true);
}

/**
 * Upscale the radius of the LUT by one if radius < MAX_RAD
 */
t_ca *ca_upscale_lut(t_ca *rule)
{
    if (rule->radius < MAX_RAD)
    {
        int len = ca_get_lut_len(rule);
        rule->radius = rule->radius + 1;
        int nlen = ca_get_lut_len(rule);

        _LUT_ENTRY *tmpLut = calloc(len, sizeof(_LUT_ENTRY));
        memcpy(tmpLut, rule->lut, len * sizeof(_LUT_ENTRY));

        int off = nlen / 2;
        for (int i = 0; i < len; i++)
        {
            rule->lut[2 * i] = tmpLut[i];
            rule->lut[2 * i + 1] = tmpLut[i];
            rule->lut[off + 2 * i] = tmpLut[i];
            rule->lut[off + 2 * i + 1] = tmpLut[i];
        }

        free(tmpLut);
    }
    return rule;
}

/**
 * Re-scale the LUT to the new radius.
 * If given radius does not belong 1..MAX_RAD it will be trimmed.
 */
t_ca *ca_set_radius_check(t_ca *rule, int radius, bool check)
{
    if (rule->radius == radius)
    {
        return rule;
    }

    radius = radius < 1 ? 1 : radius;
    radius = radius > MAX_RAD ? MAX_RAD : radius;

    int diff = rule->radius - radius;
    if (diff == 0)
        return rule;
    if (diff < 0)
    {
        for (int i = diff; i < 0; i++)
        {
            ca_upscale_lut(rule);
        }
    }
    else
    {
        for (int i = 0; i < diff; i++)
        {
            ca_downscale_lut_check(rule, check);
        }
    }

    return rule;
}

t_ca *ca_set_radius(t_ca *rule, int radius)
{
    return ca_set_radius_check(rule, radius, true);
}

/**
 * Checks if CAs c1 and c2 are equal, meaning that the radius is the same and LUTs are equal.
 * In general this could be improved to support different radius, and LUTs defining the same rule
 */
bool ca_equal(t_ca *c1, t_ca *c2)
{
    if (c1->radius != c2->radius)
    {
        return false;
    }

    int l = ca_get_lut_len(c1);

    for (int i = 0; i < l; i++)
    {
        if (c1->lut[i] != c2->lut[i])
        {
            return false;
        }
    }

    return true;
}

/**
 * Calculates the distance between LUTs of CAs c1 and c2
 * If radius of c1 is different from c2, the result is -1
 */
_DIST_ENTRY ca_dist(t_ca *c1, t_ca *c2)
{
    if (c1->radius != c2->radius)
    {
        return -1;
    }
    int l = ca_get_lut_len(c1);
    _DIST_ENTRY res = 0;
    for (int i = 0; i < l; i++)
    {
        _LUT_ENTRY d = c1->lut[i] - c2->lut[i];
        res += abs(d);
    }
    return res;
}

_DIST_ENTRY ca_dist_upscale(t_ca *c1, t_ca *c2)
{
    if (c1->radius != c2->radius)
    {
        int r = c1->radius > c2->radius ? c1->radius : c2->radius;
        t_ca *a = ca_new(c1->radius);
        t_ca *b = ca_new(c2->radius);
        ca_copy(a, c1);
        ca_copy(b, c2);
        ca_set_radius(a, r);
        ca_set_radius(b, r);
        _DIST_ENTRY res = ca_dist(a, b);
        ca_free(a);
        ca_free(b);
        return res;
    }
    else
        return ca_dist(c1, c2);
}

/**
 * Returns the number of the neighborhood with configuration conf of length l
 * Example: If l = 3  and conf = 101, then the results is 5.
 */
int ca_num_index(_CONF space, int l)
{
    int r = 0;
    for (int i = 0; i < l; i++)
    {
        r += space[i] * (1 << (l - i - 1));
    }
    return r;
}

/**
 * Checks if given configuration is complete (i.e. no cells with state -1)
 */
bool conf_is_complete(_CONF a)
{
    for (int i = 0; i < CELL_COUNT; i++)
    {
        if (a[i] < 0.0)
            return false;
    }
    return true;
}

/**
 * Utility function to simplify distance calculations.
 */
_CONF_ENTRY conf_pow(_CONF_ENTRY a)
{
    return a * a;
}

/**
 * Return the distance between configurations
 */
_CONF_ENTRY conf_diff(_CONF r1, _CONF r2)
{
    _CONF_ENTRY result = 0;

    for (int i = 0; i < CELL_COUNT; i++)
    {
        if (r1[i] >= 0 && r2[i] >= 0)
        {
            result += conf_pow(r1[i] - r2[i]);
        }
    }

    return result;
}

/**
 * Constructs new, uninitialized configuration.
 */
_CONF conf_new()
{
    return (_CONF)malloc(CELL_COUNT * sizeof(_CONF_ENTRY));
}

/**
 * Copy src configuration to res. It is assumed that both configurations are initialized.
 */
_CONF conf_copy(_CONF res, _CONF src)
{
    memcpy(res, src, CELL_COUNT * sizeof(_CONF_ENTRY));
    return res;
}

/**
 * Initialize a new configuration and copy the src into it.
 */
_CONF conf_new_copy(_CONF src)
{
    _CONF result = conf_new();
    return conf_copy(result, src);
}

/**
 * Completes configuration src - for each i where src[i] < 0 we use value comp[i] instead. The result is stored in res configuration.
 * All configurations are assumed to be initialized.
 */
_CONF conf_complete(_CONF res, _CONF src, _CONF comp)
{
    conf_copy(res, src);
    for (int i = 0; i < CELL_COUNT; i++)
    {
        if (res[i] < 0)
        {
            res[i] = comp[i];
        }
    }
    return res;
}

/**
 * Initialize new configuration, copy src into it, and the complete entries with < 0 values using values in comp
 */
_CONF conf_new_copy_complete(_CONF src, _CONF comp)
{
    _CONF result = conf_new();
    return conf_complete(result, src, comp);
}

/**
 * How many cells in c are completely observed?
 */
int conf_count_complete(_CONF c)
{
    int result = 0;
    for (int i = 0; i < CELL_COUNT; i++)
    {
        result += c[i] >= 0 ? 1 : 0;
    }
    return result;
}

/**
 * Deallocate configuration (free the memory).
 */
void conf_free(_CONF c)
{
    free(c);
}

/**
 * Same as ca_num_index, but the configuration is divided into two parts
 * conf = [space_left, space_right]
 * l = l_left + l_right
 */
int ca_num_index_2(_CONF space_left, int l_left, _CONF space_right, int l_right)
{
    int r = 0;
    for (int i = 0; i < l_left; i++)
    {
        r += space_left[i] * (1 << ((l_left + l_right) - i - 1));
    }

    for (int i = l_left; i < l_left + l_right; i++)
    {
        r += space_right[i - l_left] * (1 << ((l_left + l_right) - i - 1));
    }
    return r;
}

/**
 * Apply the rule on given input, and store the result in output.
 * Only complete inputs are allowed, i.e. for all i, input[i] >= 0 AND THIS CONDITION IS NOT VALIDATED!
 * If input is incomplete, some strange things will happen
 */
void ca_apply_rule_bool(t_ca *rule, _CONF input, _CONF output)
{
    int R = 2 * rule->radius;

    for (int i = rule->radius; i < CELL_COUNT - rule->radius; i++)
    {
        output[i] = rule->lut[ca_num_index(input - rule->radius + i, R + 1)];
    }

    // the two loops below are responsible for supporting periodic boundary conditions

    for (int i = 0; i < rule->radius; i++)
    {
        output[CELL_COUNT - rule->radius + i] = rule->lut[ca_num_index_2(input + (CELL_COUNT - R + i), R - i, input, i + 1)];
    }

    for (int i = 0; i < rule->radius; i++)
    {
        output[i] = rule->lut[ca_num_index_2(input + (CELL_COUNT - rule->radius + i), rule->radius - i, input, i + 1 + rule->radius)];
    }
}

/**
 * Apply the CA rule on input, and store results in output. All of the parameters need to be initialized (allocated).
 */
void ca_apply_rule(t_ca *rule, _CONF input, _CONF output)
{
    ca_apply_rule_bool(rule, input, output);
}

/**
 * Calculate the one step error of transition from input to output using given rule. The configuration target_output is the desired configuration.
 * The result of applying rule in input is returned in rule_output.
 */
_CONF_ENTRY id_error_step(t_ca *rule, _CONF input, _CONF target_output, _CONF rule_output)
{
    _CONF out = rule_output;
    bool clear = false;
    if (out == NULL)
    {
        out = conf_new();
        clear = true;
    }
    ca_apply_rule(rule, input, out);
    _CONF_ENTRY diff = conf_diff(target_output, out);

    if (clear)
    {
        conf_free(out);
    }
    return diff;
}

/**
 * Calculate the one step error with possible time gap of size 1,...,MAX_GAP, of transition from input to output using given rule.
 * The configuration target_output is the desired configuration.
 * The result of applying rule in input is returned in rule_output, and the resulting time gap size is written to gap.
 * Note that gap might be NULL - in such case time gaps are not returned
 */
_CONF_ENTRY id_error_step_gap(t_ca *rule, _CONF orginal_input, _CONF target_output, _CONF rule_output, int gap_multiply, int *gap)
{

    int len_gap = gap_multiply * MAX_GAP;
    int idx_min = gap_multiply - 1;

    _CONF *resultImage = malloc((len_gap) * sizeof(_CONF));
    for (int i = 0; i < len_gap; i++)
    {
        resultImage[i] = conf_new();
    }

    _CONF_ENTRY errors[len_gap];

    _CONF_ENTRY result = id_error_step(rule, orginal_input, target_output, resultImage[0]);

    errors[0] = result;
    int min_count = 1;

    for (int t = 1; t < len_gap; t++)
    {
        _CONF_ENTRY err = id_error_step(rule, resultImage[t - 1], target_output, resultImage[t]);
        errors[t] = err;

        if (t == idx_min || err < result)
        {
            result = err;
            min_count = 1;
        }
        else if (err == result)
        {
            min_count++;
        }
    }

    int idx = 0;

    if (conf_is_complete(target_output) == false)
    {
        idx = tool_rand_i(0, min_count);
    }

    int g = idx_min;
    for (; g < len_gap; g++)
    {
        if (errors[g] == result)
        {
            if (idx == 0)
                break;
            idx--;
        }
    }

    conf_copy(rule_output, resultImage[g]);

    if (gap != NULL)
    {
        (*gap) = g;
    }

    for (int i = 0; i < len_gap; i++)
    {
        conf_free(resultImage[i]);
    }

    free(resultImage);

    return result;
}

/**
 * Calculated the cumulative error of the rule on given image.
 * If mask[i] == false for some i, it means that transition image[i] -> image[i+1] is not verified.
 * Time gaps are assumed the be 1..MAX_GAP, and are returned in gap (i.e. gap[i] is the time gap between image[i] and image[i+1]).
 * If gap == NULL, time gaps are not returned
 */
_CONF_ENTRY id_error(t_ca *rule, t_img *img, int *gap)
{
    _CONF_ENTRY result = 0;

    _CONF input = conf_new_copy(img->data[0]);
    _CONF output = conf_new();
    _CONF tmp = conf_new();

    int skip = 0;

    for (int t = 0; t < TIME_STEPS - 1; t++)
    {
        output = conf_copy(output, img->data[t + 1]); // take a row from observation

        if (conf_count_complete(output) <= PARTIAL_TRESHOLD)
        { // check if we should ignore this row?
            if (gap != NULL)
            {
                (*(gap + t)) = -1;
            }
            skip++;
        }
        else
        { // if we don't ignore - find the error ...
            skip = 0;
            if (img->mask[t])
            { // ... if this row is not masked due to duplication :)
                _CONF_ENTRY r = id_error_step_gap(rule, input, output, tmp, skip + 1, gap != NULL ? gap + t : NULL);

                result += r;
                conf_complete(input, output, tmp);
            }
            else
            {
                if (gap != NULL)
                {
                    (*(gap + t)) = -1;
                }
                conf_copy(input, output);
            }
        }
    }

    conf_free(input);
    conf_free(output);
    conf_free(tmp);

    return result;
}

/**
 * Mutation operator for the GA
 */
t_ca *evolve_mutate(t_ca *ca, double pm)
{

    // upscale mutation
    if (MIN_RAD < MAX_RAD && tool_rand() < pm)
    {
        ca_upscale_lut(ca);
    }

    int l = ca_get_lut_len(ca);
    int c = tool_rand_i(0, l);
    for (int j = 0; j < c; j++)
    {
        if (tool_rand() < pm)
        {
            int i = tool_rand_i(0, l);
            ca->lut[i] = 1 - ca->lut[i];
        }
    }

    // downscale mutation
    if (MIN_RAD < MAX_RAD && tool_rand() < pm)
    {
        ca_downscale_lut(ca);
    }

    return ca;
}

/**
 * Crossover operator
 */
t_ca *evolve_cross(t_ca *res, t_ca *p1, t_ca *p2)
{
    if (p1->radius == p2->radius)
    {
        res->radius = p1->radius;
        int l = ca_get_lut_len(res);

        for (int i = 0; i < l; i++)
        {
            res->lut[i] = tool_rand() < 0.5 ? p1->lut[i] : p2->lut[i];
        }
    }
    else
    {
        t_ca *t1 = ca_new(p1->radius);
        ca_copy(t1, p1);

        t_ca *t2 = ca_new(p2->radius);
        ca_copy(t2, p2);

        res->radius = p1->radius > p2->radius ? tool_rand_i(p2->radius, 1 + p1->radius) : tool_rand_i(p1->radius, 1 + p2->radius);

        ca_set_radius(t1, res->radius);
        ca_set_radius(t2, res->radius);
        evolve_cross(res, t1, t2);

        ca_free(t1);
        ca_free(t2);
    }

    return res;
}

/**
 * Calculate fitenss for given rule
 */
double evolve_fitness(t_ca *rule, t_case *problem, bool use_mask)
{
    _CONF_ENTRY f = 0;
    for (int j = 0; j < problem->count; j++)
    {
        if (!use_mask || problem->mask[j])
        {
            f += id_error(rule, problem->img + j, NULL);
        }
    }

    int x = (use_mask ? problem->masked_cells : problem->total_cells);
    double res = (x - f);
    return res / x;
}

/**
 * Calculates fitness values for the population pop, and a problem setting problem.
 * The use_mask flag tells if we should use problem->mask to determine which images to include in the fitness (use_mask == true),
 * or use all images (use_mask == false) to get an exact result
 */
void evolve_recalc_fitenss(t_pop *pop, t_case *problem, bool use_mask)
{
#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < POP_COUNT; i++)
    {
        pop->fitness[i] = evolve_fitness(pop->rules + i, problem, use_mask);
    }
}

void problem_count_cells(t_case *problem)
{
    problem->masked_cells = 0;
    for (int j = 0; j < problem->count; j++)
    {
        if (problem->mask[j])
            problem->masked_cells += problem->img[j].complete_cells;
    }
}

/**
 * Slighlty change the image mask, replacing one of the images with a different one.
 * Originally this was a deterministic shift, i.e. if mask was [1,2,3] then after shift it was [2,3,4].
 * Now a random replacement is used.
 */
void evolve_shift_mask(t_case *problem)
{
    if (problem->count > problem->mask_count)
    {
        int i;
        do
        {
            i = tool_rand_i(0, problem->count);
        } while (problem->mask[i] == false);

        int j;
        do
        {
            j = tool_rand_i(0, problem->count);
        } while (problem->mask[j] == true);

        problem->mask[i] = false;
        problem->mask[j] = true;
    }
}

/**
 * Calculates fitness values and updates the statistics of a population
 */
void evolve_recalc_fitness_and_stats(t_pop *pop, t_case *problem, bool use_mask)
{
    evolve_recalc_fitenss(pop, problem, use_mask);
#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < POP_COUNT; i++)
    {
        pop->rank[i] = i;
    }

    tool_sort_fitness_index(pop->rank, pop->fitness, POP_COUNT);

    pop->dist[0] = pop->fitness[0];
    for (int i = 1; i < POP_COUNT; i++)
    {
        pop->dist[i] = pop->dist[i - 1] + pop->fitness[i];
    }

    double d = pop->dist[POP_COUNT - 1];

    pop->stats.avg = d / POP_COUNT;

    for (int i = 0; i < POP_COUNT; i++)
    {
        pop->dist[i] /= d;
    }

    pop->stats.min = pop->fitness[pop->rank[0]];
    pop->stats.max = pop->fitness[pop->rank[POP_COUNT - 1]];
    pop->stats.min_idx = pop->rank[0];
    pop->stats.max_idx = pop->rank[POP_COUNT - 1];

    pop->stats.min_errors = 0;

    // calculate the min_error which is used for the stopping criteria of the GA
    // note that, we are NOT using the problem image mask here - we always use all of the images
    int max_idx = pop->stats.max_idx;
    t_ca *best = &pop->rules[max_idx];
    int min_error = 0;

    int err[problem->count];
    int idx[problem->count];

    int total[problem->count];

    tool_range(idx, 0, problem->count);

    for (int i = 0; i < problem->count; i++)
    {
        err[i] = id_error(best, problem->img + i, NULL);
        total[i] = problem->img[i].complete_cells;
        min_error += err[i];
    }

    tool_sort_index_int(idx, err, problem->count);

    pop->stats.min_errors = min_error;

    int i = idx[problem->count - 1];

    if (problem->mask[i] == true)
    {
        evolve_shift_mask(problem);
    }

    if (problem->mask[i] == false)
    {
        int j;
        do
        {
            j = tool_rand_i(0, problem->count);
        } while (i == j || problem->mask[j] == false);
        problem->mask[j] = false;
        problem->mask[i] = true;
    }

    memset(pop->radius_count, 0, sizeof(int) * MAX_RAD);

    for (int k = 0; k < POP_COUNT; k++)
    {
        pop->radius_count[pop->rules[k].radius - 1]++;
    }

    problem_count_cells(problem);
}

/**
 * Initializes the population with random rules in given population p.
 */
t_pop *evolve_init_pop_a(t_pop *p)
{
#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < POP_COUNT; i++)
    {
        p->rules[i].radius = tool_rand_i(MIN_RAD, MAX_RAD + 1);
        ca_init_random(p->rules + i);
    }
    return p;
}

/**
 * Initializes the population with random rules. Allocates new memory.
 */
t_pop *evolve_init_pop()
{
    return evolve_init_pop_a(calloc(1, sizeof(t_pop)));
}

/**
 * Selection operator - picks a rule from population, with selection probability proportional to fitness.
 * It is assumed that fitness values are already updated for the population!
 */
t_ca *evolve_pick_parent(t_pop *p)
{
    double r = tool_rand();
    int i = 0;
    while (p->dist[i] <= r && i < POP_COUNT)
    {
        i++;
    }
    if (i > POP_COUNT)
        i = POP_COUNT - 1; // this is probably not needed :)
    return p->rules + i;
}

/**
 * Apply the genetic operators to build young population based on the old population
 * pm is the current value of mutation probability (in some variants of the algorithm we considered variable mutation probability,
 * although now a constant value of _P_FLIP is used)
 */
void evolve_new_pop(t_pop *young, t_pop *old, double pm)
{
    for (int i = 0; i < POP_COUNT; i++)
    {
        t_ca *p1 = evolve_pick_parent(old);
        t_ca *p2 = evolve_pick_parent(old);
        t_ca *res = young->rules + i;
        evolve_cross(res, p1, p2);
        evolve_mutate(young->rules + i, pm);
    }
}

/**
 * Function to print out current statistics of the population
 * Produces output if LOG is defined
 */
void evolve_print_stats(int iter, t_pop *p, t_case *problem, double best_fit, int best_age, double pm)
{
#ifdef _LOG
    printf("t=%d best=%.3f (%d) pm=%.3f min_e=%d max=%.3f avg=%.3f min=%.3f ", iter + 1, 100 * best_fit, best_age, pm, p->stats.min_errors, 100 * p->stats.max, 100 * p->stats.avg, 100 * p->stats.min);

    for (int i = 0; i < MAX_RAD; i++)
    {
        printf("r[%d]=%.2f ", i + 1, (double)p->radius_count[i] / POP_COUNT);
    }

    printf("img=");
    for (int i = 0; i < problem->count; i++)
    {
        if (problem->mask[i])
        {
            printf("%d, ", i);
        }
    }

    printf("\n");
    fflush(stdout);
#endif
}

/**
 * Copy elite from old to young.
 */
void evolve_perserve_elite(t_pop *young, t_pop *old)
{
    int idx[ELITE_COUNT];
    memset(idx, 0, ELITE_COUNT * sizeof(int));
    for (int i = 0; i < ELITE_COUNT; i++)
    {
        int j;
        bool ok;
        do
        {
            j = tool_rand_i(0, POP_COUNT);
            ok = true;
            for (int k = 0; k < i; k++)
            {
                if (idx[k] == j)
                {
                    ok = false;
                    break;
                }
            }
            idx[i] = j;
        } while (!ok);

        int I = POP_COUNT - 1 - i;
        memcpy(young->rules[j].lut, old->rules[old->rank[I]].lut, sizeof(_LUT_ENTRY) * MAX_LEN);
        young->rules[j].radius = old->rules[old->rank[I]].radius;
    }
}

/**
 * (Re)initialize the image mask for the problem with random selections.
 */
void evolve_reinit_mask(t_case *problem)
{
    memset(problem->mask, false, sizeof(bool) * problem->count);
    for (int i = 0; i < problem->mask_count; i++)
    {
        int idx;
        do
        {
            idx = tool_rand_i(0, problem->count);
        } while (problem->mask[idx] == true);
        problem->mask[idx] = true;
    }

    problem_count_cells(problem);
}

/**
 * The main GA algorithm implementation.
 * The resulting "best" rule is returned in best, which is assumed to be allocated (non-null).
 * The error (number of wrong states) of the best rule is returned in min_error (non-null).
 * The number of iterations used to find best is returned in iter (non-null).
 * The problem to solve is given by problem (non-null).
 */
void evolve_algorithm(t_ca *best, int *min_error, int *iter, t_case *problem)
{
    t_pop *young = calloc(1, sizeof(t_pop)); // this is empty, to hold the results of first GA epoch
    t_pop *old = evolve_init_pop();          // this is the initial, random population

    evolve_recalc_fitness_and_stats(old, problem, false);

    int best_age = 0; // the "age" of current best fitness -- how long the current best fitness is best
    double best_fit = old->fitness[old->stats.max_idx];

    bool elite = true;     // determines if we are in elite mode
    int elite_reboot = -1; // if we are not in elite mode, here we store the iteration on which we will be returning to elite mode

    int i = 0; // iteration number
    for (; old->stats.min_errors > 0 && (!LIMIT_ITER || i < MAX_GA_ITER); i++)
    {
        double pm;
        if (elite)
        {
            pm = exp(-0.025 * (MAX_BEST_AGE - best_age));
            if (pm > 1)
                pm = 1;
            if (pm < _P_FLIP)
                pm = _P_FLIP;
        }
        else
        {
            pm = _P_FLIP;
        }

        // elite reboot - i.e. going back to elite
        if (!elite && elite_reboot == i)
        {
            elite = true;
            elite_reboot = -1;
            best_fit = old->fitness[old->stats.max_idx];
            best_age = 0;
#ifdef _LOG
            printf("Elite: on\n");
#endif
        }

        evolve_print_stats(i, old, problem, best_fit, best_age, pm);

        evolve_new_pop(young, old, pm);

        if (elite)
        {
            evolve_perserve_elite(young, old);
        }

        // swap young and old
        t_pop *tmp_swap = old;
        old = young;
        young = tmp_swap;

        evolve_recalc_fitness_and_stats(old, problem, true);

        // calculate best_age and turn elite on if a new best is found
        if (old->fitness[old->stats.max_idx] > best_fit)
        {
            best_fit = old->fitness[old->stats.max_idx];
            best_age = 0;
            if (!elite)
            {
                elite = true;
                elite_reboot = -1;
#ifdef _LOG
                printf("Elite: on\n");
#endif
            }
        }
        else
        {
            best_age++;
        }

        // reboot
        if (best_age > MAX_BEST_AGE && elite)
        {
            best_age = 0;

            elite_reboot = i + 1;
            evolve_init_pop_a(old);
            evolve_reinit_mask(problem);
            evolve_recalc_fitness_and_stats(old, problem, true);
            best_fit = old->fitness[old->stats.max_idx];
        }
    }

    // print the final stats
    evolve_print_stats(i, old, problem, best_fit, best_age, 0);

    // return results
    ca_copy(best, &old->rules[old->stats.max_idx]);
    (*min_error) = old->stats.min_errors;
    (*iter) = i;

    free(young);
    free(old);
}

/**
 * Checks if two configurations are equal in terms of the configuration distance (which is not a strict "metric" distance if configurations are not complete)
 */
bool conf_equal(_CONF c1, _CONF c2)
{
    _CONF_ENTRY dist = conf_diff(c1, c2);
    return dist == 0;
}

/**
 * Calculates the image row mask - this is to remove duplicated pairs of complete rows in a image,
 * since they don't bring any additional information. This is especially helpfull when considering simple rules like ECA 0.
 */
void img_calculate_masks(t_img *img)
{
#ifdef _USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < TIME_STEPS; i++)
    {
        img->is_complete[i] = conf_is_complete(img->data[i]);
    }

    memset(img->mask, true, sizeof(bool) * (TIME_STEPS - 1));

    img->mask_rows = 0;
    for (int i = 0; i < TIME_STEPS - 1; i++)
    {
        if (img->mask[i] == true)
        {
            img->mask_rows++;
        }
    }

    img->complete_cells = 0;

    for (int i = 1; i < TIME_STEPS; i++)
    {
        if (img->mask[i - 1] == true)
        {
            for (int j = 0; j < CELL_COUNT; j++)
            {
                img->complete_cells += img->data[i][j] > -1 ? 1 : 0;
            }
        }
    }
}

/**
 * Utility method to select a random time gap lenght, according to type
 */
int test_generate_gap(e_generator_gap_type type)
{
    int max = type == ALL ? MAX_GAP : MAX_GAP / 2;
    int min = type == EVEN ? 1 : 0;
    int res = tool_rand_i(min, max);
    res = type == ALL ? res : (type == EVEN ? 2 * res : (2 * res + 1 > MAX_GAP ? 2 * res - 1 : 2 * res + 1));
    return res;
}

/**
 * Generates noise in the given configuration
 * Returns number of cells broken with noise.
 */
int test_gen_noise(_CONF conf, double noise)
{
    int noise_count = 0;
    if (noise > 0)
    {
        for (int i = 0; i < CELL_COUNT; i++)
        {
            if (tool_rand() < noise)
            {
                conf[i] = 1 - conf[i];
                noise_count++;
            }
        }
    }

    return noise_count;
}

/**
 * Generate test image using CA ca. It is assumed that img[0] holds an initialized initial condition which is complete!
 * prob is the probability for cell-wise spatial incompleteness, i.e. after building a row, for each cell decision is made if
 * the calculated value is stored, or if the spatial partiality (incompleteness) is introduced by value -1.
 * If prob == 0.0, a complete image is built. If prob == 1.0, the image has only the initial condition filled and -1 in all other cells.
 * Note that the initial condition is always complete and not changed by this function.
 */
int test_gen_image(t_ca *ca, double prob, _CONF_ENTRY img[][CELL_COUNT], int *out_gaps, int gap, t_generator_setup *setup)
{
#ifdef _LOG
    printf("Generating image\n");
#endif
    _CONF tmp1 = conf_new_copy(img[0]);
    _CONF tmp2 = conf_new();
    int noise_count = 0;

    for (int i = 1; i < TIME_STEPS; i++)
    {
        int steps = gap == -1 ? test_generate_gap(setup->type) : gap;

        if (out_gaps != NULL)
        {
            out_gaps[i - 1] = steps;
        }

        for (int j = 0; j <= steps; j++)
        {
            ca_apply_rule(ca, tmp1, tmp2);
            _CONF t = tmp1;
            tmp1 = tmp2;
            tmp2 = t;
        }

        if (setup->noise_type == SYSTEM)
        {
            noise_count += test_gen_noise(tmp1, setup->noise);
        }

        conf_copy(img[i], tmp1);

        if (setup->noise_type == OBSERVATION)
        {
            noise_count += test_gen_noise(img[i], setup->noise);
        }

        if (prob > 0)
        {
            for (int j = 0; j < CELL_COUNT; j++)
            {
                // introduce spatial incompleteness in observation
                if (tool_rand() < prob)
                {
                    img[i][j] = -1;
                }
            }
        }
    }

#ifdef _LOG
    printf("total noise count: %d\n", noise_count);
#endif

    conf_free(tmp1);
    conf_free(tmp2);
    return noise_count;
}

// in this include file we store a pre-defined set of initial conditions
// so that experiments are reproducible / comparable
// you are free to experiment with this set (generate a new one)
#include "initial_condition.h"

/**
 * Generates a set of observations with given properties:
 * ca - generator of observations
 * image_count - number of observations to generate
 * image_mult - number of variants of each observation (see code for details - not used in the article) - set this to 1 to disable!!!
 * prob - probability of cell-wise spatiall incompletness introduction
 * setup - additional parameters of generator (for noise introduction - NOT used in the article)
 */
t_case *test_prepare_ca(t_ca *ca, int image_count, int image_mult, double prob, t_generator_setup *setup)
{
    t_case *problem = calloc(1, sizeof(t_case));

    problem->count = image_count; // image count

    if (problem->count > 100)
    {
        problem->count = 100;
    }

    problem->mask_count = 8;
    if (problem->mask_count <= 0)
    {
        problem->mask_count = 1;
    }

    if (problem->mask_count > problem->count)
    {
        problem->mask_count = problem->count;
    }

    problem->img = calloc(problem->count, sizeof(t_img));
    problem->mask = calloc(problem->count, sizeof(bool));

    problem->gaps = calloc(problem->count, sizeof(int *));
    for (int i = 0; i < problem->count; i++)
    {
        problem->gaps[i] = calloc(TIME_STEPS - 1, sizeof(int));
    }

    evolve_reinit_mask(problem);
    problem->original_rule = calloc(1, sizeof(t_ca));
    ca_copy(problem->original_rule, ca);

    problem->total_cells = 0;

    int gap = -1;

#ifdef _LOG
    printf("Time gap generator mode = %d\n", setup->mode);
#endif

    if (setup->mode == CONSTANT_PER_PROBLEM)
    {
        gap = test_generate_gap(setup->type);

#ifdef _LOG
        printf("Global time gap: %d\n", gap);
#endif
    }
    int noise_count = 0;
    for (int i = 0; i < problem->count / image_mult; i++)
    {
        int l = strlen(random_init[i]);
        for (int k = 0; k < image_mult; k++)
        {
            for (int j = 0; j < CELL_COUNT; j++)
            {
                problem->img[image_mult * i + k].data[0][j] = random_init[i][j % l] == '1';
            }
            problem->img[image_mult * i + k].data[0][k] = 1 - problem->img[image_mult * i + k].data[0][k];
            if (setup->mode == CONSTANT_PER_IMAGE)
            {
                gap = test_generate_gap(setup->type);

#ifdef _LOG
                printf("Gap for image %d: %d\n", image_mult * i + k, gap);
#endif
            }
            noise_count += test_gen_image(problem->original_rule, prob, problem->img[image_mult * i + k].data, problem->gaps[i], gap, setup);
            img_calculate_masks(problem->img + image_mult * i + k);
            problem->total_cells += (problem->img + image_mult * i + k)->complete_cells;
        }
    }

#ifdef _LOG
    printf("noise_count = %d\n", noise_count);
#endif

    return problem;
}

t_case *test_prepare_problem(int rule_num, int image_count, int image_mult, double prob, t_generator_setup *setup)
{
    t_ca *ca = ca_new(1);
    ca_decodeRule(ca, rule_num, 8);
    t_case *res = test_prepare_ca(ca, image_count, image_mult, prob, setup);
    ca_free(ca);
    return res;
}

static int test_problem_fitness(int rule_num, int image_count, int image_mult, double prob)
{
    t_generator_setup setup = {.mode = RANDOM, .type = ALL, .noise = 0, .noise_type = SYSTEM};

    t_case *problem = test_prepare_problem(rule_num, image_count, image_mult, prob, &setup);

    t_ca solution;
    int min_error = 0;
    int iter = 0;

    t_ca *tmp = calloc(1, sizeof(t_ca));

    int rule = rule_num;
    tmp->radius = 1;
    ca_decodeRule(tmp, rule, 8);
    ca_set_radius(tmp, 2);
    _DIST_ENTRY dist = 0;
    if (problem->original_rule != NULL)
    {
        dist = ca_dist(problem->original_rule, tmp);
    }

    _CONF_ENTRY err = 0;
    bool errs[image_count];
    for (int i = 0; i < problem->count; i++)
    {
        _CONF_ENTRY e = id_error(tmp, problem->img + i, NULL);
        errs[i] = e > 0;
        err += e;
    }

#ifdef _LOG
    print_array_b(errs, image_count);
    printf("%d %d %d\n", rule, dist, err);
#endif

    ca_free(problem->original_rule);
    ca_free(tmp);
    free(problem->mask);
    free(problem->img);
    free(problem);
    return EXIT_SUCCESS;
}

static int test_evolution_problem(t_case *problem)
{
    t_ca *ca = problem->original_rule;

    t_ca solution, solution_scale;
    int min_error = 0;
    int iter = 0;

#ifdef _LOG
    printf("Original radius: %d\n", ca->radius);
    _PRINT_ARRAY(ca->lut, ca_get_lut_len(ca));
#endif

    evolve_algorithm(&solution, &min_error, &iter, problem);

#ifdef _LOG
    printf("Solutions radius: %d\n", solution.radius);
    _PRINT_ARRAY(solution.lut, ca_get_lut_len(&solution));

    printf("Solution casted to radius: %d\n", ca->radius);
    ca_copy(&solution_scale, &solution);
    if (solution_scale.radius != ca->radius)
    {
        ca_set_radius_check(&solution_scale, ca->radius, false);
    }
    _PRINT_ARRAY(solution_scale.lut, ca_get_lut_len(&solution_scale));

    printf("Original radius: %d\n", ca->radius);
    _PRINT_ARRAY(ca->lut, ca_get_lut_len(ca));
#endif

    if (ca->radius != solution.radius)
    {
        int r = ca->radius > solution.radius ? ca->radius : solution.radius;
        ca_set_radius(ca, r);
        ca_set_radius(&solution, r);
    }

    _DIST_ENTRY dist = ca_dist_upscale(ca, &solution);

    // this short output is useful when LOG is not defined
    if (min_error == 0)
    {
        printf("RES: OK iter=%d dist=%d\n", iter, dist);
    }
    else
    {
        printf("RES: ERR min_err=%d iter=%d dist=%d\n", min_error, iter, dist);
    }

    int err = 0;
    for (int i = 0; i < problem->count; i++)
    {
        err += id_error(problem->original_rule, problem->img + i, NULL);
    }

    double fit_f = evolve_fitness(problem->original_rule, problem, false);
    double fit_t = evolve_fitness(problem->original_rule, problem, true);

    printf("fitness unmasker = %f; fitness masked = %f; errors count = %d\n", fit_f, fit_t, err);

    return EXIT_SUCCESS;
}

/**
 * This prepares the problem and executes the algorithm, based on a specified ECA rule with given rule_num.
 */
static int test_evolution(int rule_num, int image_count, int image_mult, double prob, double noise)
{
    t_generator_setup setup = {.mode = RANDOM, .type = ALL, .noise = noise};

    t_case *problem = test_prepare_problem(rule_num, image_count, image_mult, prob, &setup);

    int res = test_evolution_problem(problem);

    ca_free(problem->original_rule);
    free(problem->mask);
    free(problem->img);
    free(problem);
    return EXIT_SUCCESS;
}

/**
 * This prepares the problem and executes the algorithm, based on a specified ECA rule with given rule_num.
 */
static int test_evolution_random_rule(int radius, int image_count, int image_mult, double prob)
{
    t_ca *ca = ca_new(radius);
    ca_init_random(ca);
    t_generator_setup setup = {.mode = RANDOM, .type = ALL, .noise = 0.0};
    t_case *problem = test_prepare_ca(ca, image_count, image_mult, prob, &setup);

    int res = test_evolution_problem(problem);

    ca_free(problem->original_rule);
    free(problem->mask);
    free(problem->img);
    free(problem);
    return res;
}

/**
 * Loads saved problem (observation set) from a given directory
 */
t_case *problem_load(char *name, int image_cnt)
{
    t_case *problem = calloc(1, sizeof(t_case));
    problem->count = image_cnt;
    problem->img = calloc(problem->count, sizeof(t_img));
    problem->mask_count = 8 < image_cnt ? 8 : image_cnt;
    problem->mask = calloc(problem->count, sizeof(bool));
    problem->original_rule = NULL;
    problem->total_cells = 0;

    int l = strlen(name);
    for (int i = 0; i < image_cnt; i++)
    {
        char fname[l + 10];
        sprintf(fname, "%s/%d", name, i);
        FILE *fp = fopen(fname, "r");
        int m = 0;
        if (fp != NULL)
        {
            while (!feof(fp) && m < TIME_STEPS)
            {
                char row[CELL_COUNT + 2];
                if (fgets(row, CELL_COUNT + 2, fp) != NULL)
                {
                    for (int n = 0; n < CELL_COUNT; n++)
                    {
                        int val = 0;
                        if (row[n] == '1')
                            val = 1;
                        else if (row[n] == '.')
                            val = -1;
                        problem->img[i].data[m][n] = val;
                    }
                    m++;
                }
            }
            img_calculate_masks(problem->img + i);
            problem->total_cells += problem->img[i].complete_cells;
            fclose(fp);
        }
        else
        {
            printf("Error loading file: %s\n", fname);
            exit(EXIT_FAILURE);
        }
    }

    evolve_reinit_mask(problem);
    return problem;
}

/**
 * Loads a pre-prepared problem from files and runs algorithm on it
 */
static int test_evolution_saved(char *problem_name, int image_cnt)
{

    double start = tool_get_time();

    t_case *problem = problem_load(problem_name, image_cnt);

    t_ca solution;
    int min_error = 0;
    int iter = 0;
    evolve_algorithm(&solution, &min_error, &iter, problem);

#ifdef _LOG
    printf("Solutions radius: %d\n", solution.radius);
    _PRINT_ARRAY(solution.lut, ca_get_lut_len(&solution));
#endif

    // this short output is useful when LOG is not defined
    if (min_error == 0)
    {
        printf("RES: OK iter=%d\n", iter);
    }
    else
    {
        printf("RES: ERR min_err=%d iter=%d\n", min_error, iter);
    }

    free(problem->mask);
    free(problem->img);
    free(problem);

    printf("Iter / sec: %f\n", iter / (tool_get_time() - start));
    return EXIT_SUCCESS;
}

/**
 * Test for mask shifting - can be ignored ;)
 */
static int test_mask_shift()
{
    t_case *problem = calloc(1, sizeof(t_case));

    problem->count = 20;
    problem->mask_count = 7;
    problem->mask = calloc(problem->count, sizeof(bool));
    for (int i = 0; i < problem->mask_count; i++)
    {
        problem->mask[i] = true;
    }

    for (int i = 0; i < 30; i++)
    {
        print_array_b(problem->mask, problem->count);
        evolve_shift_mask(problem);
    }

    return EXIT_SUCCESS;
}

/**
 * Test for fitness - can be ignored :)
 */
static int test_missing_data_fitness(int argc, char **argv)
{

    t_case *problem = calloc(1, sizeof(t_case));

    problem->count = atoi(argv[2]); // image count

    if (problem->count > 100)
    {
        problem->count = 100;
    }

    problem->mask_count = problem->count / 25;
    if (problem->mask_count <= 0)
    {
        problem->mask_count = 1;
    }

    problem->img = calloc(problem->count, sizeof(t_img));
    problem->mask = calloc(problem->count, sizeof(bool));

    evolve_reinit_mask(problem);

    t_ca *ca = calloc(1, sizeof(t_ca));
    t_ca *ca2 = calloc(1, sizeof(t_ca));

    ca->radius = 1;
    ca2->radius = 1;

    double prob = atof(argv[3]);

    ca_decodeRule(ca, atoi(argv[1]), ca_get_lut_len(ca));
    ca_decodeRule(ca2, atoi(argv[1]), ca_get_lut_len(ca2));
    ca_upscale_lut(ca2);

    for (int i = 0; i < problem->count; i++)
    {
        for (int j = 0; j < CELL_COUNT; j++)
        {
            int l = strlen(random_init[i]);
            problem->img[i].data[0][j] = random_init[i][j % l] == '1';
        }
        test_gen_image(ca2, prob, problem->img[i].data, NULL, -1, ALL);
        img_calculate_masks(problem->img + i);
    }

    for (int rule = atoi(argv[1]); rule < atoi(argv[1]) + 1; rule++)
    {
        ca->radius = 1;
        ca_decodeRule(ca, rule, 8);
        ca2->radius = 1;
        ca_decodeRule(ca2, rule, 8);
        ca_upscale_lut(ca2);
        for (int i = 0; i < 100; i++)
        {

            int err = id_error(ca, problem->img + i, NULL);
            int err2 = id_error(ca2, problem->img + i, NULL);

            printf("%d %d %d %d\n", rule, i, err, err2);
        }
    }
    return EXIT_SUCCESS;
}

/**
 * "Unit test" for CA evalutaion and down/up scaling - produces data for drawing images with gnuplot
 */
static int test_create_diff_plot(int argc, char **argv)
{

    t_ca *ca = calloc(1, sizeof(t_ca));
    t_ca *ca2 = calloc(1, sizeof(t_ca));

    ca->radius = 1;
    ca2->radius = 1;

    ca_decodeRule(ca, atoi(argv[1]), ca_get_lut_len(ca));
    ca_decodeRule(ca2, atoi(argv[1]), ca_get_lut_len(ca2));
    ca_upscale_lut(ca2);
    ca_upscale_lut(ca2);

    t_case *problem = calloc(1, sizeof(t_case));
    problem->count = 2;
    problem->img = calloc(problem->count, sizeof(t_img));

    for (int j = 0; j < CELL_COUNT; j++)
    {
        int l = strlen(random_init[0]);
        problem->img[0].data[0][j] = random_init[0][j % l] == '1';
        problem->img[1].data[0][j] = random_init[0][j % l] == '1';
    }
    test_gen_image(ca, 0, problem->img[0].data, NULL, -1, ALL);
    test_gen_image(ca2, 0, problem->img[1].data, NULL, -1, ALL);

    for (int i = 0; i < TIME_STEPS; i++)
    {
        for (int j = 0; j < CELL_COUNT; j++)
        {
            printf("%d %d %d\n", i, j, problem->img[0].data[i][j] - problem->img[1].data[i][j]);
        }
    }

    return EXIT_SUCCESS;
}

/**
 * Saves a generated problem (observation set) to a folder (each observation is a file in this folder)
 */
void problem_save(char *name, t_case *problem)
{
    int s = mkdir(name, 0700);
    if (s == 0 || (s < 0 && errno == EEXIST))
    {
        int l = strlen(name);
        for (int i = 0; i < problem->count; i++)
        {
            char fname[l + 10];
            sprintf(fname, "%s/%d", name, i);
            FILE *fp = fopen(fname, "w");

            if (problem->gaps != NULL)
            {
                char fname_gaps[strlen(fname) + 6];
                sprintf(fname_gaps, "%s.gaps", fname);
                FILE *fp_gaps = fopen(fname_gaps, "w");

                if (fp_gaps != NULL)
                {
                    for (int t = 0; t < TIME_STEPS - 1; t++)
                    {
                        fprintf(fp_gaps, "%d\n", problem->gaps[i][t]);
                    }
                    fclose(fp_gaps);
                }
                else
                {
                    printf("Error: can't create file - %s\n", fname_gaps);
                }
            }

            if (fp != NULL)
            {
                for (int t = 0; t < TIME_STEPS; t++)
                {
                    for (int c = 0; c < CELL_COUNT; c++)
                    {
                        char val = '0';
                        if (problem->img[i].data[t][c] == 1)
                            val = '1';
                        else if (problem->img[i].data[t][c] < 0)
                            val = '.';
                        fprintf(fp, "%c", val);
                    }
                    if (t != TIME_STEPS - 1)
                        fprintf(fp, "\n");
                }
                fclose(fp);
            }
            else
            {
                printf("Error opening file: %s\n", fname);
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        printf("Error creating dir: %s\n", name);
    }
}

/**
 * Generates a set of observations and saves them to a folder with files
 */
static void test_problem_generator(char *name, int rule_num, int image_cnt, int image_mult, t_generator_setup *setup)
{
    t_case *problem = test_prepare_problem(rule_num, image_cnt, image_mult, 0, setup);
    problem_save(name, problem);
}

/**
 * Entry point
 * FIXME: separate this function to functionality specific files
 */
int main(int argc, char **argv)
{
    // NOTE: THIS IS REQUIRED FOR tool_rand_* to work!
    // IF THIS IS NOT EXECTUED STRANGE THINGS MIGHT HAPPEN (probably memory corruption)
    tool_init_random();

#ifdef _BUILD_GENERATOR
    int arg_needed = 6;
    if (argc == arg_needed)
    {
        e_generator_mode mode;
        //printf("%c\n", argv[5][0]);
        switch (argv[5][0])
        {
        case 'i':
            mode = CONSTANT_PER_IMAGE;
            break;
        case 'p':
            mode = CONSTANT_PER_PROBLEM;
            break;
        case 'r':
        default:
            mode = RANDOM;
            break;
        }

        e_generator_gap_type gap_type;

        switch (argv[5][0])
        {
        case 'e':
            gap_type = EVEN;
            break;
        case 'o':
            gap_type = ODD;
            break;
        case 'a':
        default:
            gap_type = ALL;
            break;
        }
        printf("mode= %d\ntype=%d\n", mode, gap_type);

        t_generator_setup setup;
        setup.mode = mode;
        setup.type = gap_type;
        setup.noise = 0.0;

        test_problem_generator(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &setup);
    }
    else
    {
        printf("Problem Generator (complete observations)\n");
        printf("Usage: %s name rule_num img_cnt img_mult gap_mode=(i|p|r) gap_type=(e|o|a)\n", argv[0]);
        printf("\t gap_mode: i = constant gap per image, p = constant gap per problem, r = random gap per each line\n");
        printf("\t gap_type: e = only even gaps, o = only odd gaps, a = all gaps allowed\n");
        return EXIT_FAILURE;
    }

#elif _BUILD_AUTO_ALG
    int arg_needed = 5;
    if (argc == arg_needed)
    {
        return test_evolution(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atof(argv[4]), 0.0);
        //return test_problem_fitness(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atof(argv[4]));
        //test_problem_generator(argv[1], atoi(argv[2]), atoi(argv[3]));
        //test_evolution_saved(argv[1], atoi(argv[2]));
    }
    else
    {
        printf("Evolutionary algorithm - generated problem\n");
        printf("Usage: %s rule_num image_count image_mult prob\n", argv[0]);
        return EXIT_FAILURE;
    }
#elif _BUILD_RANDOM
    int arg_needed = 5;
    if (argc == arg_needed)
    {
        return test_evolution_random_rule(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atof(argv[4]));
    }
    else
    {
        printf("Evolutionary algorithm - generated problem\n");
        printf("Usage: %s radius image_count image_mult prob\n", argv[0]);
        return EXIT_FAILURE;
    }
#else
    int arg_needed = 3;
    if (argc == arg_needed)
    {
        test_evolution_saved(argv[1], atoi(argv[2]));
    }
    else
    {
        printf("Evolutionary algorithm - read pre-generated problem\n");
        printf("Usage: %s problem_name image_count\n", argv[0]);
        return EXIT_FAILURE;
    }
#endif
}
