# Identification of Cellular Automata based on partial observations with bounded time gaps

In this repo you will find an implementation of an identification algorithm described
in the paper "Identification of Cellular Automata based on partial observations with bounded time gaps"
by W. Bołt, J.M. Baetens and B. De Baets (to be published in 2018).

## Building

The main code (see: *spatial.c* file) is written in C language (C99 standard) and it is known to work on Linux and OS X platforms. There are known issues with running it on Windows (yet it should be relatively easy do port it).

To build the source code you need to have a recent version of GCC, Clang or Intel C/C++ (icc) compiler. In case of GCC you want to have a version with OpenMP support. In case of Clang, note that OpenMP is not supported and thus performance will be lower!

In addition you need to have recent version of GNU Scientific Library (GSL) installed. Official site: [http://www.gnu.org/software/gsl/](http://www.gnu.org/software/gsl/), and binaries can be easily found in common repositories (apt-get, yum, brew, *etc.*).

For convenience simple building scripts are provided for GCC and icc. See: *build.sh* for GCC and *build_icc.sh* for icc. Note that *build_gcc.sh* should not be used in most cases.

The supplied build scripts require one parameter to be provided - the build identifier, which is used as a base name for the binaries generated. For example:

```bash
./build_icc.sh demo
```

This will result in:
* creation of a subdirectory called *builds* (if it does not exists)
* creation of four binaries: *demo*-auto, *demo*-gen, *demo*-rand and *demo*-saved

Note that by default the build scripts will try to build a OpenMP variant of the code, which requires OpenMP support in the compiler, and will result in a multi-threaded version which is much faster on multi-core / multi-processor machines.

## Running

The application can be executed in one of four modes. For performance reasons, although each mode is expressed in one source code - the result is stored in four separate binaries. The modes of execution are following:

* **auto generated problem** - in this setup, the program generates a set of observations based on the selected ECA and tries to solve the identification problem for this set. This mode is good for quick experiments and demonstrations. It is not good for real study since at every execution a new problem is generated and thus it is hard to compare results.

* **problem generator mode** - in this setup, the program generates a problem (*i.e.* an observation set) and stores it under a given name for future usage. This is useful for multiple executions of the algorithm on the same data.

* **saved problem solution** - in this setup, a pre-generated problem is being used for the identification. You need to have a generated problem to use this mode, which can be built using the problem generator mode.

* **random rule mode** - this setup is similar to *auto generated problem* mode, with one key difference - here we only specify the radius of the CA that we want to have, and the specific rule is selected randomly. This mode was created exclusively for Experiment 4 from the paper and probably will not be very useful for other cases.

### Auto generated problem

This mode is stored in binary *builds/[id]-auto* where *[id]* is the build id given to the build script. The program requires following parameters:

```
Usage: ./demo-auto rule_num image_count image_mult prob
```

* *rule_num* - Wolfram code of an ECA to use
* *image_count* - Number of different initial conditions to use
* *image_mult* - Number of variations of each of the initial conditions to make (see next chapter for details)
* *prob* - cell-wise probability of spatial partiality

Example:

```
./demo-auto 150 20 1 0
```

This will generate 20 observations, each starting from different initial configuration, by using ECA 150. The observations will be spatially complete.


### Problem generator

Usage:
```
Usage: ./demo-gen name rule_num img_cnt img_mult gap_mode=(i|p|r) gap_type=(e|o|a)
	 gap_mode: i = constant gap per image, p = constant gap per problem, r = random gap per each line
	 gap_type: e = only even gaps, o = only odd gaps, a = all gaps allowed
```
where:

* *name* - name for the problem (technically name of a directory - can contain slashes for sub-directories, for ex. "problems/150/1" is a valid problem name)
* *rule_nume* - same as in auto mode
* *img_cnt* - same as in auto mode
* *img_mult* - same as in auto mode
* *gap_mode* - how to distribute time gaps ("i" = for each observation we randomly select the time gap length and use it for every row, "p" = one, randomly selected time gap length is used for the entire problem, "r" = random length, selected independently for every row in every observation) - "r" is the default for *auto* mode
* *gap_type* - what gaps do we allow ("e" - only even, "o" - only odd, "a" - all)

Note that currently generator does not support generating spatially partial observations! A Python script was used in the experiments, to modify pre-generated observations to add partiality.

Example:
```
./demo-gen test-problem/1 150 20 1 r a
```
This will try to create a directory `./test-problem/1` and store the problem there. It will consist of 20 observations of the ECA 150 with random time gaps of all types.

Note that the total number of images generated is equal to *img_cnt* x *img_mult*.

### Saved problem
Usage:
```
./demo-saved problem_name image_count
```
where:

* *problem_name* - name of the problem to load
* *image_count* - number of observations from the problem to load (starting from observation 1) - this is useful for big problems... we can try to algorithm on a subset first - note that this argument is required even if all observations shall be used! Note that this is especially important if *img_mult* > 1 was set in the generator.


### Random rule_nume

Usage:
```
./demo-rand radius image_count image_mult prob
```

In contrast to other modes, here we can use not only ECAs but also other 1D, binary CAs, therefore a radius needs to be given. Other options are straightforward.

## Output specification

In all of the modes other than the generator, we might expect the following output or similar:

```
./demo-auto 0 20 1 0
Time gap generator mode = 1
Original radius: 1
0,0,0,0,0,0,0,0
t=1 best=99.890 (0) pm=0.002 min_e=103 max=99.890 avg=76.114 min=0.337 r[1]=0.00 r[2]=0.27 r[3]=0.24 r[4]=0.25 r[5]=0.24 img=2, 3, 5, 8, 10, 11, 15, 18,
t=2 best=99.890 (1) pm=0.002 min_e=103 max=99.875 avg=93.072 min=0.717 r[1]=0.00 r[2]=0.25 r[3]=0.30 r[4]=0.27 r[5]=0.18 img=1, 2, 3, 8, 10, 11, 15, 18,
t=3 best=99.890 (2) pm=0.002 min_e=103 max=99.877 avg=94.451 min=0.458 r[1]=0.00 r[2]=0.27 r[3]=0.35 r[4]=0.28 r[5]=0.11 img=1, 2, 8, 10, 11, 13, 15, 18,
t=4 best=100.000 (0) pm=0.000 min_e=0 max=100.000 avg=96.984 min=0.565 r[1]=0.00 r[2]=0.33 r[3]=0.38 r[4]=0.21 r[5]=0.08 img=0, 1, 2, 8, 10, 11, 13, 18,
Solutions radius: 2
0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0
RES: OK iter=3 dist=10
1.000000 1.000000 0
```

Let us discuss the meaning of the lines observed. First the initial diagnostic lines:
```
Time gap generator mode = 1
Original radius: 1
0,0,0,0,0,0,0,0
```
In most cases useless. We see an info on the time gap generator mode, which is meaningless if you don't see the source code, the radius of the rule used for generation and the LUT of the rule.

After that the results lines are placed. The most interesting are lines starting with `t=` like this:

```
t=3 best=99.890 (2) pm=0.002 min_e=103 max=99.877 avg=94.451 min=0.458 r[1]=0.00 r[2]=0.27 r[3]=0.35 r[4]=0.28 r[5]=0.11 img=1, 2, 8, 10, 11, 13, 15, 18,
```

The meaning of the elements is following:
* `t=3` - this is the third GA iteration
* `best=99.890` - the best rule found so far correctly fits 99.89% of the cells in observations, we aim to get 100% here
* `(2)` - the age of the current best is two GA iterations
* `pm=0.002` - the current probability for bit-flip mutation (cell wise) is 0.002
* `min_e=103` - the number of incorrectly classified (filled) cells by the best rule found so far - we aim to get 0 here
* `max=99.877` - the current (in this iteration) best score in percentage of matching cells (see the fit and error definition in the paper)
* `avg=94.451` - similar to above, but averaged over the entire population
* `min=0.458` - similar to above, but minimum from the current population
* `r[1]=0.00 r[2]=0.27 r[3]=0.35 r[4]=0.28 r[5]=0.11` - the ratio of rules in the current population, belonging to the given radius, for exaple we have 35% of the population in radius 3 and so on
* `img=1, 2, 8, 10, 11, 13, 15, 18,` - ids of observations used for the subset for fitness approximation (note that max, avg, min and best are calculated using approximation, only min_e is exact (?))

Finally we have the closing lines of the output:
```
Solutions radius: 2
0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0
RES: OK iter=3 dist=10
fitness unmasker = 1.000000; fitness masked = 1.000000; errors count = 0
```

This means that a solution with radius 2 was found, the LUT is given then, and we have a line starting with `RES:` which says that a solution was found (`OK` says it) in 3 iterations (`iter=3`) and the LUT of solutions that was found differs on 10 entires (`dist=10`) from the original rule (upscaled to the radius of the solution that was found). If `dist=0` we know that we found the same rule that was used for generation. Note that when running the algorithm in a pre-save problem mode, the `dist` measure is not available, since we don't know the rule - we only have the observation set.

Although the output is not CSV or anything structured it is pretty straight forward to parse it with sed and grep in order to have something ready for gnuplot, excel or anything similar. When analyzing multiple runs of the algorithm it is useful to grep over lines with `RES:` and when analyzing the behavior of a single run of the algorithm you typically want to grep over `t=`.

## Parameter tweaking

Almost all of the parameters are hardcoded. This is for the performance reasons. In most cases the parameters are expressed as preprocessor macro instructions like `#define`. Due to this the compiler is able to make use of specific values to pre-allocate the fixed memory ranges, and thus dynamic allocation is relatively rare in the code.

Most of the parameters are to be found in *spatial.h* file. The most important parameters are:

*  `#define MIN_RAD 2` - the minimal radius allowed in the population
* `#define MAX_RAD 5` - the maximal radius allowed in the population
* `#define MAX_GAP 10` - size of the maximal time gap
* `#define MAX_GA_ITER 1000` - maximum number of GA iterations
* `#define LIMIT_ITER true` - should we use the MAX_GA_ITER limit? if false, the algorithm may run forever
* `#define CELL_COUNT 69` - number of cells in each observation (NOTICE: see below for how the initial conditions are created - this number can not be too big for now)
* `#define TIME_STEPS 69` - number of rows in each observation
* `#define PARTIAL_TRESHOLD 0` - if a row of an observation contains more missing cells (question marks) than this number, it is ignored
* `#define _P_DOWN 0.001` - down-scale mutation probability
* `#define _P_UP 0.001` - up-scale mutation probability
* `#define _P_FLIP 0.001` - used as a base for dynamic bit-flip mutation probability (see the code for exact formula)
* `#define POP_COUNT 512` - population size
* `#define ELITE_COUNT 128` - elite size
* `#define MAX_BEST_AGE 250` - we reset the population if the best age is bigger then this
* `#define _LOG` - if this is set, we get the nice, verbose logging of every GA iteration (the log lines starting with `t=`) - we might not set this, then we only get the lines with `RES:` ... not setting this can increase performance, since if we are not logging, we can also skip some computation
* `#define BOOL_CA` - the code base has some initial support for identification of so-called Affine Continuous Cellular Automata, if this is NOT set and if you set `CONT_CA` instead, you get this experimental variant - do *not* use it for now!

There is also one important parameter in *main.h* file, because why not :)

* `#define _USE_OPENMP` - if you don't set this, you get single-thread version which is slow, to use OpenMP you need to have a compiler that supports is. For example clang (LLVM) is known not to support it, and many GCC versions also might not support it (although newer versions should have it by default) - also note that in most cases you need to put a special compiler flag to turn OpenMP support on (`-fopenmp` for GCC and `-openmp` for icc)

## Initial configurations

For some important reasons the set of initial configurations is hardcoded. Two of them are: performance and reproducibility.

The initial configurations are stored in a separate header file: *initial_condition.h*. Note that there are exactly *100* initial configurations defined there and the length of each of these vectors is *200* elements. It means that the number of different observations which is defined by the *image_count* / *img_cnt* start parameter is limited by 100, and the cell count defined by a constant *CELL_COUNT* is limited by 200. To make things easier :) this is never checked - so if you ran out of the limits, strange things might happen.

The file *initial_condition.h* can be recreated with the following Python (2.x) script:
```python
import random

print "const char *random_init[] = {"
print ','.join(['"'+''.join([str(random.randint(0,1)) for x in range(0,200)])+'"\n' for y in range(0,100)])
print "}"
```
