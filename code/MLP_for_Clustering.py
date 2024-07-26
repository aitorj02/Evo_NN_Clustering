import numpy as np
#import pandas as pd
import time
import sys
sys.path.append('deaft_dev/deatf/')
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
import tensorflow as tf
import torch
#from auxiliary_functions import load_fashion

from deatf_dev.deatf.network import MLP, MLPDescriptor, RNNDescriptor
from deatf_dev.deatf.evolution import Evolving

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def mse(a, b):
    return np.sqrt(np.mean((a-b)**2))
def mse_closest(a, b):
    """
    Calculate the mean squared error (MSE) between each point in 'a' and the closest centroid in 'b'.

    Parameters:
    a (numpy.ndarray): An array of shape (n_samples,n_points*2), p.e: (160, 2000), containing the points for each sample.
    b (numpy.ndarray): An array of shape (n_samples, n_centroids*2)p.e (160,4) containing the centroids for each sample.

    Returns:
    float: The average of the minimum squared distances between each point and the closest centroid.
    """

    n_samples = a.shape[0]
    n_points = a.shape[1] // 2  # 2000 coordinates, 1000 points (x, y)
    n_centroids = b.shape[1] // 2  # 4 coordinates, 2 centroids (x, y)

    # Initialize an array to hold the minimum distances
    min_distances = np.zeros((n_samples, n_points))

    # Reshape a to shape (160, 1000, 2) and b to shape (160, 2, 2)
    a_reshaped = a.reshape(n_samples, n_points, 2)
    b_reshaped = b.reshape(n_samples, n_centroids, 2)

    for i in range(n_samples):
        points = a_reshaped[i]
        centroids = b_reshaped[i]

        # Calculate the squared distances from each point to each centroid
        squared_distances = np.sum((points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)

        # Find the minimum squared distance for each point
        min_squared_distances = np.min(squared_distances, axis=1)

        # Store the minimum distances
        min_distances[i] = min_squared_distances

    # Calculate the average of the minimum squared distances for all points in all samples
    mse_val = np.mean(min_distances)

    return mse_val


def eval_mlp(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    The model is created with one RNN, but it needs a dense layer with a softmax activation function.
    That is needed because they are probability distributions and they have to be between 0 and 1.
    Finally accuracy error is used to measuare the performance of the model.

    :param nets: Dictionary with the networks that will be used to build the
                 final network and that represent the individuals to be
                 evaluated in the genetic algorithm.
    :param train_inputs: Input data for training, this data will only be used to
                         give it to the created networks and train them.
    :param train_outputs: Output data for training, it will be used to compare
                          the returned values by the networks and see their performance.
    :param batch_size: Number of samples per batch are used during training process.
    :param iters: Number of iterations that each network will be trained.
    :param test_inputs: Input data for testing, this data will only be used to
                        give it to the created networks and test them. It can not be used during
                        training in order to get a real feedback.
    :param test_outputs: Output data for testing, it will be used to compare
                         the returned values by the networks and see their real performance.
    :param hypers: Hyperparameters that are being evolved and used in the process.
    :return: Accuracy error obtained with the test data that evaluates the true
             performance of the network.
    """


    n_output = train_outputs["o0"].shape[1]
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    #out = Dense(n_output, activation=None)(out)
    model = Model(inputs=inp, outputs=out)

    #model.summary()

    mlp_opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=mlp_opt, metrics=[])


    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters,
              batch_size=batch_size, verbose=0)


    preds = model.predict(test_inputs["i0"])

    #res = tf.nn.softmax(preds)
    #res = tf.nn.softmax(preds)
    mse_val_closest = mse_closest(test_inputs['i0'], preds)
    mse_val = mse(preds, test_outputs["o0"])
    #print("truth",test_outputs["o0"][:5,:],"preds",preds[:5,:],"MSE",mse_val)
    print("normal_MSE",mse_val )
    print("closest_MSE",mse_val_closest )
    return mse_val,



# Evolve the architectures of the networks for a number of generations
def Evolve_MLP_Architectures(init_seed,type_activation,n_generations,population_size,n_iterations,batch_size,
                             max_layers,max_neurons,
                             x_train, y_train, x_val, y_val):



    n_features = x_train.shape[1]
    n_targets =  y_train.shape[1]
    custom_mutations = dict()
    custom_mutations['MLPDescriptor'] =['mut_add_layer','mut_remove_layer','mut_dimension']
    e = Evolving(evaluation=eval_mlp, desc_list=[MLPDescriptor],compl=False,
        x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val],
        n_inputs=[[n_features]], n_outputs=[[n_targets]], batch_size=batch_size,
        iters=n_iterations,
        population=population_size, generations=n_generations,
        max_num_layers=max_layers, max_num_neurons=max_neurons,
        seed=init_seed,
        dropout=True, batch_norm=True,
        evol_alg='mu_plus_lambda', evol_kwargs={'mu':population_size, 'lambda_':population_size, 'cxpb':0.1, "mutpb": 0.9},
                 #evol_alg='mu_plus_lambda', evol_kwargs={'mu':10, 'lambda_':15, 'cxpb':0.1, "mutpb": 0.9},
        hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]},
        custom_mutations = custom_mutations,
        sel = 'best',sel_kwargs={})
    best_networks, log_book, hall_of = e.evolve()


    #return best_networks, log_book
    return best_networks, log_book, hall_of




def Evaluate_Evolved_Network(type_activation,mpl_descriptor,n_iterations,batch_size,x_train, y_train, x_val, y_val):



        n_features = x_train.shape[1]
        n_targets =  y_train.shape[1]
        nfeatures = x_val.shape[1]

        net = MLP(mpl_descriptor)

        inp = Input(shape=n_features)
        out = Flatten()(inp)
        out = net.building(out)
        model = Model(inputs=inp, outputs=out)

        model.compile(loss="MSE" , optimizer="adam", metrics=["mean_squared_error"])

        history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    batch_size=batch_size,
                    epochs=n_iterations,
                    verbose=0)

        # Model training
        #history = training_MLP(model,train_X,train_y, test_X, test_y,
        #               batch_size, epochs, verbose, shuffle)


        # Visualize training history
        #Visualize_Training_History(history)


        preds_train = model.predict(x_train)
        preds_val = model.predict(x_val)


        mse_train = mse(preds_train, y_train)
        mse_val = mse(preds_val, y_val)

        return model, mse_train, mse_val, preds_train, preds_val



#Change the objetive function:
def eval_mlp_closest(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    The model is created with one RNN, but it needs a dense layer with a softmax activation function.
    That is needed because they are probability distributions and they have to be between 0 and 1.
    Finally accuracy error is used to measuare the performance of the model.

    :param nets: Dictionary with the networks that will be used to build the
                 final network and that represent the individuals to be
                 evaluated in the genetic algorithm.
    :param train_inputs: Input data for training, this data will only be used to
                         give it to the created networks and train them.
    :param train_outputs: Output data for training, it will be used to compare
                          the returned values by the networks and see their performance.
    :param batch_size: Number of samples per batch are used during training process.
    :param iters: Number of iterations that each network will be trained.
    :param test_inputs: Input data for testing, this data will only be used to
                        give it to the created networks and test them. It can not be used during
                        training in order to get a real feedback.
    :param test_outputs: Output data for testing, it will be used to compare
                         the returned values by the networks and see their real performance.
    :param hypers: Hyperparameters that are being evolved and used in the process.
    :return: Accuracy error obtained with the test data that evaluates the true
             performance of the network.
    """
    n_output = train_outputs["o0"].shape[1]
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    #out = Dense(n_output, activation=None)(out)
    model = Model(inputs=inp, outputs=out)
    mlp_opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=mlp_opt, metrics=[])
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters,
              batch_size=batch_size, verbose=0)
    preds = model.predict(test_inputs["i0"])
    #print(preds)

    #res = tf.nn.softmax(preds)
    mse_val_closest = mse_closest(test_inputs['i0'], preds)
    # #probar si aprende tambien aunque no sea su funcion
    mse_val = mse(preds, test_outputs["o0"])

    print("normal_MSE",mse_val )
    print("closest_MSE",mse_val_closest )

    # pop = population_size_val
    # mse_extra.append(mse_val)
    # if len(mse_extra)==n_gene+pop:
    #     mse_extra[n_gene] = np.mean(mse_extra[n_gene:n_gene+pop])
    #     del mse_extra[n_gene+1:]
    #     print("MSE_normal optimizando MSE_closest",mse_extra)
    #     n_gene = n_gene+1
    #print("truth",test_outputs["o0"][:5,:],"preds",preds[:5,:],"MSE",mse_val)
    return mse_val_closest,


def Evolve_MLP_Architectures_closest(init_seed,type_activation,n_generations,population_size,n_iterations,batch_size,
                             max_layers,max_neurons,
                             x_train, y_train, x_val, y_val):



    n_features = x_train.shape[1]
    n_targets =  y_train.shape[1]
    custom_mutations = dict()
    custom_mutations['MLPDescriptor'] =['mut_add_layer','mut_remove_layer','mut_dimension']
    e = Evolving(evaluation=eval_mlp_closest, desc_list=[MLPDescriptor],compl=False,
        x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val],
        n_inputs=[[n_features]], n_outputs=[[n_targets]], batch_size=batch_size,
        iters=n_iterations,
        population=population_size, generations=n_generations,
        max_num_layers=max_layers, max_num_neurons=max_neurons,
        seed=init_seed,
        dropout=True, batch_norm=True,
        evol_alg='mu_plus_lambda', evol_kwargs={'mu':population_size, 'lambda_':population_size, 'cxpb':0.1, "mutpb": 0.9},
                 #evol_alg='mu_plus_lambda', evol_kwargs={'mu':10, 'lambda_':15, 'cxpb':0.1, "mutpb": 0.9},
        hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]},
        custom_mutations = custom_mutations,
        sel = 'best',sel_kwargs={})
    best_networks, log_book, hall_of = e.evolve()


    #return best_networks, log_book
    return best_networks, log_book, hall_of




def Evaluate_Evolved_Network_closest(type_activation,mpl_descriptor,n_iterations,batch_size,x_train, y_train, x_val, y_val):


        n_features = x_train.shape[1]
        n_targets =  y_train.shape[1]
        nfeatures = x_val.shape[1]

        net = MLP(mpl_descriptor)

        inp = Input(shape=(n_features,))
        out = Flatten()(inp)
        out = net.building(out)
        model = Model(inputs=inp, outputs=out)

        model.compile(loss="MSE" , optimizer="adam", metrics=["mean_squared_error"])

        history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    batch_size=batch_size,
                    epochs=n_iterations,
                    verbose=0)

        # Model training
        #history = training_MLP(model,train_X,train_y, test_X, test_y,
        #               batch_size, epochs, verbose, shuffle)


        # Visualize training history
        #Visualize_Training_History(history)


        preds_train = model.predict(x_train)
        preds_val = model.predict(x_val)

        mse_train = mse_closest(x_train, preds_train)
        mse_val = mse_closest(x_val, preds_val)
        print("MSE for train",mse_train)
        print("MSE for val",mse_val)

        return  model, mse_train, mse_val, preds_train, preds_val

tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)
def mse_closest_torch(a, b):
    """
    Calculate the mean squared error (MSE) between each point in 'a' and the closest centroid in 'b'.

    Parameters:
    a (tf.Tensor): A tensor of shape (n_samples, n_points*2), e.g., (160, 2000), containing the points for each sample.
    b (tf.Tensor): A tensor of shape (n_samples, n_centroids*2), e.g., (160, 4), containing the centroids for each sample.

    Returns:
    float: The average of the minimum squared distances between each point and the closest centroid.
    """
    # Slicing from the fourth position (index 3) onwards along the second dimension
    aa = a[:, 4:]
    #print(a.shape)

    n_samples = aa.shape[0]         # nsamples in this case is the batch size
    n_points = aa.shape[1] // 2     # 2004 coordinates, 1002 points (x, y), [trainoutputs,traininputs]
    n_centroids = b.shape[1] // 2  # 4 coordinates, 2 centroids (x, y)
    #print(n_samples)
    # print(a[0][:4])
    # print(a[0][4:])
    # a = a[4:]



    # Initialize a tensor to hold the minimum distances
    min_distances = tf.zeros((n_samples, n_points), dtype=aa.dtype)
    # Reshape a to shape (n_samples, n_points, 2) and b to shape (n_samples, n_centroids, 2)
    a_reshaped = tf.reshape(aa, (n_samples, n_points, 2))
    b_reshaped = tf.reshape(b, (n_samples, n_centroids, 2))

    for i in range(n_samples):
        points = a_reshaped[i]
        centroids = b_reshaped[i]


        # Calculate the squared distances from each point to each centroid
        squared_distances = tf.reduce_sum((tf.expand_dims(points, 1) - tf.expand_dims(centroids, 0)) ** 2, axis=2)

        # Find the minimum squared distance for each point
        min_squared_distances = tf.reduce_min(squared_distances, axis=1)
        #print(min_squared_distances)
        # Store the minimum distances
        min_distances = tf.tensor_scatter_nd_update(min_distances, [[i]], [min_squared_distances])
    # Calculate the average of the minimum squared distances for all points in all samples
    mse_val = tf.reduce_mean(min_distances)
    #print(mse_val.numpy())

    return mse_val

def custom_loss(y_true,y_pred):
    return mse_closest_torch(y_true,y_pred)

#change the "loss_function":
def eval_mlp_closest_customloss(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    The model is created with one RNN, but it needs a dense layer with a softmax activation function.
    That is needed because they are probability distributions and they have to be between 0 and 1.
    Finally accuracy error is used to measuare the performance of the model.

    :param nets: Dictionary with the networks that will be used to build the
                 final network and that represent the individuals to be
                 evaluated in the genetic algorithm.
    :param train_inputs: Input data for training, this data will only be used to
                         give it to the created networks and train them.
    :param train_outputs: Output data for training, it will be used to compare
                          the returned values by the networks and see their performance.
    :param batch_size: Number of samples per batch are used during training process.
    :param iters: Number of iterations that each network will be trained.
    :param test_inputs: Input data for testing, this data will only be used to
                        give it to the created networks and test them. It can not be used during
                        training in order to get a real feedback.
    :param test_outputs: Output data for testing, it will be used to compare
                         the returned values by the networks and see their real performance.
    :param hypers: Hyperparameters that are being evolved and used in the process.
    :return: Accuracy error obtained with the test data that evaluates the true
             performance of the network.
    """

    # def custom_loss_wrapper(input_tensor):
    #     def custom_loss(y_true,y_pred):
    #         return mse_closest(input_tensor,y_pred)
    #     return custom_loss


    n_output = train_outputs["o0"].shape[1]
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)

    concatenated = Concatenate(axis=1)([train_outputs["o0"], train_inputs["i0"]])

    #out = Dense(n_output, activation=None)(out)
    model = Model(inputs=inp, outputs=out)
    mlp_opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=custom_loss,
                  optimizer=mlp_opt, metrics=[])
    model.fit(train_inputs['i0'], concatenated, epochs=iters,
              batch_size=batch_size, verbose=0)
    preds = model.predict(test_inputs["i0"])
    #res = tf.nn.softmax(preds)

    mse_val_closest = mse_closest(test_inputs['i0'], preds[:,:4])
    # #probar si aprende tambien aunque no sea su funcion
    mse_val = mse(preds, test_outputs["o0"])

    print("normal_MSE",mse_val )
    print("closest_MSE_custom",mse_val_closest )

    # pop = population_size_val
    # mse_extra.append(mse_val)
    # if len(mse_extra)==n_gene+pop:
    #     mse_extra[n_gene] = np.mean(mse_extra[n_gene:n_gene+pop])
    #     del mse_extra[n_gene+1:]
    #     print("MSE_normal optimizando MSE_closest",mse_extra)
    #     n_gene = n_gene+1
    #print("truth",test_outputs["o0"][:5,:],"preds",preds[:5,:],"MSE",mse_val)
    return mse_val_closest,
# Evolve the architectures of the networks for a number of generations

def Evolve_MLP_Architectures_closest_customloss(init_seed,type_activation,n_generations,population_size,n_iterations,batch_size,
                             max_layers,max_neurons,
                             x_train, y_train, x_val, y_val):



    n_features = x_train.shape[1]
    n_targets =  y_train.shape[1]
    custom_mutations = dict()
    custom_mutations['MLPDescriptor'] =['mut_add_layer','mut_remove_layer','mut_dimension']
    e = Evolving(evaluation=eval_mlp_closest_customloss, desc_list=[MLPDescriptor],compl=False,
        x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val],
        n_inputs=[[n_features]], n_outputs=[[n_targets]], batch_size=batch_size,
        iters=n_iterations,
        population=population_size, generations=n_generations,
        max_num_layers=max_layers, max_num_neurons=max_neurons,
        seed=init_seed,
        dropout=True, batch_norm=True,
        evol_alg='mu_plus_lambda', evol_kwargs={'mu':population_size, 'lambda_':population_size, 'cxpb':0.1, "mutpb": 0.9},
                 #evol_alg='mu_plus_lambda', evol_kwargs={'mu':10, 'lambda_':15, 'cxpb':0.1, "mutpb": 0.9},
        hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]},
        custom_mutations = custom_mutations,
        sel = 'best',sel_kwargs={})
    best_networks, log_book, hall_of = e.evolve()


    #return best_networks, log_book
    return best_networks, log_book, hall_of




def Evaluate_Evolved_Network_closest_customloss(type_activation,mpl_descriptor,n_iterations,batch_size,x_train, y_train, x_val, y_val):




        n_features = x_train.shape[1]
        n_targets =  y_train.shape[1]
        nfeatures = x_val.shape[1]

        net = MLP(mpl_descriptor)

        inp = Input(shape=(n_features,))
        out = Flatten()(inp)
        out = net.building(out)
        model = Model(inputs=inp, outputs=out)
        concatenated = Concatenate(axis=1)([y_train,x_train])


        model.compile(loss=custom_loss , optimizer="adam", metrics=[])

        history = model.fit(x_train,
                    concatenated,
                    
                    batch_size=batch_size,
                    epochs=n_iterations,
                    verbose=0)

        # Model training
        #history = training_MLP(model,train_X,train_y, test_X, test_y,
        #               batch_size, epochs, verbose, shuffle)


        # Visualize training history
        #Visualize_Training_History(history)


        preds_train = model.predict(x_train)
        preds_val = model.predict(x_val)

        mse_train = mse_closest(x_train, preds_train)
        mse_val = mse_closest(x_val, preds_val)
        print("MSE for train",mse_train)
        print("MSE for val",mse_val)

        return  model, mse_train, mse_val, preds_train, preds_val





