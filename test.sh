#!/bin/bash

# This script is used to test the functionality of the script.sh file.
make -j;
./ex0
./ex1
./ex2
./ex3
mpirun -np 8 ./ex4 -rule 2 -a0 0.1 -ar 2 # test with 0.1*2^iter
# ./ex5
