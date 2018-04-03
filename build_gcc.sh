#!/bin/bash

if [ ! -d ./builds ] ; then
  mkdir builds
fi;

SRC="spatial.c tools.c"
FLAGS="-D _GNU_SOURCE -lgsl -lm --std=gnu99 -fopenmp -Wall -O3 -march=native"

echo Building: $1-saved
rm -f ./builds/$1-saved
gcc $SRC -o builds/$1-saved $FLAGS
strip ./builds/$1-saved

echo Building: $1-gen
rm -rf ./buids/$1-gen
gcc $SRC -o builds/$1-gen -D_BUILD_GENERATOR $FLAGS
strip ./builds/$1-gen

echo Building: $1-auto
rm -rf ./builds/$1-auto
gcc $SRC -o builds/$1-auto -D_BUILD_AUTO_ALG $FLAGS
strip ./builds/$1-auto

echo Building: $1-rand
rm -rf ./builds/$1-rand
gcc $SRC -o builds/$1-rand -D_BUILD_RANDOM $FLAGS
strip ./builds/$1-rand
