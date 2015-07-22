#!/bin/bash
#PBS -N NUMER_INTEGRATION
#PBS -o out
#PBS -e error
#PBS -j oe
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -m abe
#PBS -l walltime=00:10:00

workdir=$PBS_O_WORKDIR
cd $PBS_O_WORKDIR
./INT_TESTER_CUDA.o
