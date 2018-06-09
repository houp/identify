#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   ext='so'
elif [[ "$unamestr" == 'Darwin' ]]; then
   ext='dylib'
else
    ext='dll'
fi

gcc spatial.c tools.c -o python/libspatial.$ext -shared -lgsl -std=gnu99 -DLOG -fopenmp -Wall -O3 -march=native -D_BUILD_LIB
