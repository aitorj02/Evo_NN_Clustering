#!/usr/bin/env python3 


import sys
import numpy as np
import os
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
  
  batch_size = 10
  n_epochs = 15
  n_generations = 20
  pop_size = 50 
  type_activation = 0
  type_repre = 0
  datasets = ['Circles_original', 'Circles_overlapped', 'Squares', 'Lines', 'Concentric']
  type_sorting = 0
  clusters = 2
  for seed in np.arange(121,152):
    for dataset in datasets:
      for loss_f in [0,1]:
        A = "sbatch slurm_class_evo_clustering.sh evolve_Clustering.py " +str(seed)+" "+str(type_sorting)+" "+str(type_repre)+" "+str(batch_size)+" "+str(n_epochs)+" "+str(n_generations)+" "+str(pop_size)+" "+str(type_activation)+" "+str(dataset)+" "+str(loss_f)+ " "+str(clusters)
        print(A)
         


        

      
       
