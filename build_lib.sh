#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   ext='so'
elif [[ "$unamestr" == 'Darwin' ]]; then
   ext='dylib'
else
    ext='dll'
fi

icc spatial.c tools.c -o python/libspatial.$ext -dynamiclib -lgsl -std=c99 -DLOG -qopenmp -Wall -fast -D_BUILD_LIB
