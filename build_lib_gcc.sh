#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   ext='so'
elif [[ "$unamestr" == 'Darwin' ]]; then
   ext='dylib'
else
    ext='dll'
fi


gcc spatial.c tools.c -o $1.$ext -shared -lgsl -std=gnu99 -DLOG -fopenmp -Wall -O3 -fPIC -march=native -D_BUILD_LIB -D_GNU_SOURCE
strip $1.$ext