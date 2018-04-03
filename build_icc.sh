#!/bin/bash

if [ ! -d ./builds ] ; then
  mkdir builds
fi;

SRC="spatial.c tools.c"
FLAGS="-lgsl -std=c99 -DLOG -qopenmp -Wall -fast"

echo Building: $1-saved
rm -f ./builds/$1-saved
icc $SRC -o builds/$1-saved $FLAGS
strip ./builds/$1-saved

echo Building: $1-gen
rm -rf ./buids/$1-gen
icc $SRC -o builds/$1-gen -D_BUILD_GENERATOR $FLAGS
strip ./builds/$1-gen

echo Building: $1-auto
rm -rf ./builds/$1-auto
icc $SRC -o builds/$1-auto -D_BUILD_AUTO_ALG $FLAGS
strip ./builds/$1-auto

echo Building: $1-rand
rm -rf ./builds/$1-rand
icc $SRC -o builds/$1-rand -D_BUILD_RANDOM $FLAGS
strip ./builds/$1-rand
