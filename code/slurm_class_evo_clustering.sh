#!/bin/bash

# Job name
#SBATCH --job-name=clustering

# Define the files which will contain the Standard and Error output
#SBATCH --output=clusters_outputs/M_%A_%a.out
#SBATCH --error=clusters_outputs/M_%A_%a.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

# Advanced use
# #SBATCH --cpus-per-task=20
# #SBATCH --threads-per-core=2
# #SBATCH --ntasks-per-core=2

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=20GB

# Select one partition
## ML-CPU // Cola de trabajos en CPUs con AVX-512 y (VNNI Vector Neural Network Instructions)
# KAT // Cola de trabajos en la GPU 
## GENERIC //Trabajos genericos que no requieran TF2 o PyTorch

#Uso de ML-CPU
#SBATCH --partition=ML-CPU

#Uso de GENERIC
# #SBATCH --partition=GENERIC

#Uso de gpu (KAT) (Si hay trabajos multi-gpu el numero puede variar de 1 a 4)
# #SBATCH --partition=KAT
# #SBATCH --gres=gpu:1

# If you are using arrays, specify the number of tasks in the array
# #SBATCH --array=1-XX

#Ejemplo: En el caso de lanzar algo con python hay que incluir priscilla exec.
#         En el caso de binarios es necesario. 
  
# priscilla exec python3 main_classification.py 111 1 1 50 10 50 3 10
# Assigning arguments to variables for better readability
echo   "priscilla exec python3 $1 $2 $3 $4 $5 $6 $7 $8 8 8 $9 ${10} ${11} ${12}> prints/${10}_clustering_best_arch_$3_$4_$5_$6_$7_$8_$9_$2_${11}_${12}.dat"      
        priscilla exec python3 $1 $2 $3 $4 $5 $6 $7 $8 8 8 $9 ${10} ${11} ${12}> prints/${10}_clustering_best_arch_$3_$4_$5_$6_$7_$8_$9_$2_${11}_${12}.dat
