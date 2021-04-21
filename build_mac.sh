#!/bin/bash

if [ ! -d ./builds ] ; then
  mkdir builds
fi;

SRC="spatial.c tools.c"
FLAGS="-lgsl -std=gnu2x -flto -march=native -O3 -lomp -DLOG -Wall -D_USE_OPENMP"
CC="clang -Xclang -fopenmp"


# echo Building: $1-saved
# rm -f ./builds/$1-saved
# $CC $SRC -o builds/$1-saved $FLAGS
# strip ./builds/$1-saved

# echo Building: $1-gen
# rm -rf ./buids/$1-gen
# $CC $SRC -o builds/$1-gen -D_BUILD_GENERATOR $FLAGS
# strip ./builds/$1-gen

echo Building: $1-auto
rm -rf ./builds/$1-auto
$CC $SRC -o builds/$1-auto -D_BUILD_AUTO_ALG $FLAGS
strip ./builds/$1-auto

# echo Building: $1-rand
# rm -rf ./builds/$1-rand
# $CC $SRC -o builds/$1-rand -D_BUILD_RANDOM $FLAGS
# strip ./builds/$1-rand

echo Building: $1-nn
rm -rf ./builds/$1-nn
$CC $SRC -o builds/$1-nn -D_BUILD_NN_TRAINER $FLAGS
strip ./builds/$1-nn
