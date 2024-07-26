import numpy as np
#import pandas as pd
import time 
import pickle
import sys
import copy
sys.path.append('deaft_dev/deatf/')
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from deatf_dev.deatf.network_all import MLP, MLPDescriptor
from deatf_dev.deatf.evolution import Evolving
from MLP_for_Clustering        import *



#################################################
#################################################
######### MAIN FUNCTION TO EXECUTE ##############
#################################################
#################################################


if __name__ == '__main__':   
    seed = int(sys.argv[1])  
    type_sorting= int(sys.argv[2])   #  Index of axis for sorting (2d)
                                     #  0: x axis
                                     #  1: y axis
                                  
    type_repre = int(sys.argv[3])    # 0: normal  1: PCA
    batch_size = int(sys.argv[4])    # Batch size for NN training
    n_epochs = int(sys.argv[5])      # Number of epochs for NN training
    n_generations = int(sys.argv[6]) # Number of the EA generations
    population_size = int(sys.argv[7]) # Population size for EA
    max_layers = int(sys.argv[8])      # Maximum number of layers for evolved NNs
    max_neurons=int(sys.argv[9])     # Maximum number of neurons for evolved NNs
    type_activation=int(sys.argv[10]) # Type of activation function
    dataset = sys.argv[11]
    type_objective_function = int(sys.argv[12])   # Type of objective function (0: mse between preds and truth 1: min distance)     1
    n_clusters = int(sys.argv[13])       #number of clusters expected to be in the data
    # python3 evolve_Clustering.py 111 0 0 5 5 10 20 8 8 0 Circles_original 2
                                    
                                      
    if n_clusters<2:
        n_clusters=2
                                 
    
    np.random.seed(seed)    
    verbose=2
    shuffle=False
    if type_repre==0:
        prefix = "new"
    elif type_repre==1:
        prefix = "pca"    
    
    try:
        data = np.load('../datasets/' + prefix + "_clusters_data_train_sorting_" + str(dataset) + '_' + str(type_sorting) + '_' + str(n_clusters) + ".npy")
    except FileNotFoundError:
        print("The dataset does not exist, maybe you want to create one.")
        sys.exit(1)
    n_comp = data.shape[1]-(n_clusters*2)
    X_train = data[:,:n_comp]
    y_train = data[:,n_comp:]

    
    data = np.load('../datasets/'+prefix+"_clusters_data_test_sorting_"+str(dataset)+'_'+str(type_sorting)+'_'+str(n_clusters)+".npy")
    X_test = data[:,:n_comp]
    y_test = data[:,n_comp:]
    
    X_train_A, X_val, y_train_A, y_val = train_test_split(X_train, y_train, test_size=0.40)

    #n_subjects,n_features = X.shape

   
    print("population_size",population_size, "n_generations", n_generations, "n_epochs", n_epochs, "batch_size", batch_size,"dataset", dataset,"n_clusters", n_clusters)
 
#    best_architectures, log_book, hall_of  = Evolve_MLP_Architectures(seed,n_generations,population_size,n_epochs,batch_size,
#                                                                      max_layers, max_neurons,X_train_A, y_train_A,
#                                                                      X_val, y_val)
    if type_objective_function==0:
        best_architectures, log_book, hall_of  = Evolve_MLP_Architectures(seed,type_activation,n_generations,population_size,n_epochs,batch_size,
                                                                        max_layers, max_neurons,X_train_A, y_train_A,
                                                                        X_val, y_val)
    elif type_objective_function==1:
        best_architectures, log_book, hall_of  = Evolve_MLP_Architectures_closest(seed,type_activation,n_generations,population_size,n_epochs,batch_size,
                                                                        max_layers, max_neurons,X_train_A, y_train_A,
                                                                        X_val, y_val)
    elif type_objective_function==2:
        best_architectures, log_book, hall_of  = Evolve_MLP_Architectures_closest_customloss(seed,type_activation,n_generations,population_size,n_epochs,batch_size,
                                                                        max_layers, max_neurons,X_train_A, y_train_A,
                                                                        X_val, y_val)
        
        

    
        
    loogbook_fname = prefix+'_class_logbook_'+"_"+str(type_sorting)+"_"+str(type_repre)+"_"+str(type_activation)+"_"+str(population_size)+"_"+str(n_generations)+"_"+str(max_layers)+"_"+str(max_neurons)+'.pkl'
    with open(loogbook_fname, 'wb') as pickle_file:
        pickle.dump(log_book, pickle_file)

   
    for i in range(population_size):
        mpl_descriptor = hall_of[i].desc_list['n0']
        print(mpl_descriptor)
        if type_objective_function==0:
            model, train_MSE, test_MSE, preds_train, preds_test  = Evaluate_Evolved_Network(type_activation,mpl_descriptor,
                                                                            3*n_epochs,batch_size,X_train, y_train,
                                                                            X_test, y_test)
        elif type_objective_function==1:
            model, train_MSE, test_MSE, preds_train, preds_test  = Evaluate_Evolved_Network_closest(type_activation,mpl_descriptor,
                                                                            3*n_epochs,batch_size,X_train, y_train,
                                                                            X_test, y_test)
        elif type_objective_function==2:
            model, train_MSE, test_MSE, preds_train, preds_test  = Evaluate_Evolved_Network_closest_customloss(type_activation,mpl_descriptor,
                                                                            3*n_epochs,batch_size,X_train, y_train,
                                                                            X_test, y_test)

        print("Sorting",type_sorting, "Model",i,train_MSE,test_MSE)
        if i==0:
            all_train_preds = preds_train
            all_test_preds = preds_test
            kmodel_name = '../models/k_model_'+prefix+"_"+str(seed)+"_"+str(type_sorting)+"_"+str(type_repre) +"_"+str(type_activation)+"_"+str(type_objective_function)+"_"+str(population_size)+"_"+str(n_generations)+"_"+str(max_layers)+"_"+str(max_neurons)+"_"+str(dataset)+"_"+str(type_objective_function)+"_"+str(n_clusters)
            model.save(kmodel_name)    
        else:
            all_train_preds = np.vstack((all_train_preds,preds_train))
            all_test_preds = np.vstack((all_test_preds,preds_test))
        
        print(i,all_train_preds.shape,all_test_preds.shape)     
        
        
  
    
    np.save('../notebooks/predictions/'+str(dataset)+'_'+ prefix+"_Final_Train_Preds_"+str(seed)+"_"+str(type_sorting)+"_"+str(type_repre)+"_"+str(type_activation)+"_"+str(population_size)+"_"+str(n_generations)+"_"+str(max_layers)+"_"+str(max_neurons)+"_"+str(type_objective_function)+"_"+str(n_clusters)+".npy",all_train_preds)
    np.save('../notebooks/predictions/'+str(dataset)+'_'+ prefix+"_Final_Test_Preds_"+str(seed)+"_"+str(type_sorting)+"_"+str(type_repre)+"_"+str(type_activation)+"_"+str(population_size)+"_"+str(n_generations)+"_"+str(max_layers)+"_"+str(max_neurons)+"_"+str(type_objective_function)+"_"+str(n_clusters)+".npy",all_test_preds)
    
    # EXAMPLE HOW TO CALL THE PROGRAM
    # python3 evolve_Clustering.py 111 0 0 2 2 2 4 8 8 0 Circles_original 2 2