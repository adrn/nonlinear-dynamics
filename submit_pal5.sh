#!/bin/sh

# Directives
#PBS -N chaos
#PBS -W group_list=yetiastro
#PBS -l nodes=8:ppn=16,walltime=4:00:00,mem=24gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

# print date and time to file
date

#Command to execute Python program
mpiexec -n 128 /vega/astro/users/amp2217/anaconda/bin/python /vega/astro/users/amp2217/projects/nonlinear-dynamics/scripts/pal5.py -v --xparam q1 15 --yparam qz 15 --nsteps=10000 --dt=5. --prefix=/vega/astro/users/amp2217/projects/nonlinear-dynamics/output/pal5 --plot-orbits --plot-indicators --mpi

date

#End of script