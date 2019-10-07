"""
This file is the combination of RSS data and Geomagnetic field data

"""

from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
import time
import os
import csv

from keras.utils import plot_model
from keras.layers import Input
from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Activation, Bidirectional, TimeDistributed, \
    RepeatVector, Concatenate
from keras.optimizers import Adam
from keras.models import Sequential, Model

from keras import backend as K
from keras import optimizers
from keras import regularizers
from sklearn.preprocessing import StandardScaler

l1 = regularizers.l1(0.0)
l2 = regularizers.l2(0.01)
regilization = l1

VERBOSE = 2  # 0 for turning off logging
# ------------------------------------------------------------------------
# stacked auto encoder (sae)
# ------------------------------------------------------------------------
# SAE_ACTIVATION = 'tanh'
SAE_ACTIVATION = 'relu'
SAE_BIAS = False
SAE_OPTIMIZER = 'adam'
SAE_LOSS = 'mse'
# ------------------------------------------------------------------------
# classifier
# ------------------------------------------------------------------------
CLASSIFIER_ACTIVATION = 'relu'
CLASSIFIER_BIAS = False
CLASSIFIER_OPTIMIZER = 'adam'
CLASSIFIER_LOSS = 'binary_crossentropy'
hidden_nodes = 64


def param():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help="ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="random seed",
        default=0,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 20",
        default=50,
        type=int)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 10",
        default=10,
        type=int)
    parser.add_argument(
        "-T",
        "--training_ratio",
        help="ratio of training data to overall data: default is 0.90",
        default=0.9,
        type=float)
    parser.add_argument(
        "-S",
        "--sae_hidden_layers",
        help=
        "comma-separated numbers of units in SAE hidden layers; default is '256,128,64,128,256'",
        default='256,128,' + str(hidden_nodes) + ',128,256',
        type=str)
    parser.add_argument(
        "-C",
        "--classifier_hidden_layers",
        help=
        "comma-separated numbers of units in classifier hidden layers; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "-D",
        "--dropout",
        help=
        "dropout rate before and after classifier hidden layers; default 0.0",
        default=0.4,
        type=float)
    # parser.add_argument(
    #     "--building_weight",
    #     help=
    #     "weight for building classes in classifier; default 1.0",
    #     default=1.0,
    #     type=float)
    # parser.add_argument(
    #     "--floor_weight",
    #     help=
    #     "weight for floor classes in classifier; default 1.0",
    #     default=1.0,
    #     type=float)
    parser.add_argument(
        "-N",
        "--neighbours",
        help="number of (nearest) neighbour locations to consider in positioning; default is 1",
        default=1,
        type=int)
    parser.add_argument(
        "--scaling",
        help=
        "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.0",
        default=0.0,
        type=float)
    args = parser.parse_args()
    return args


# This function is to open the file generated by "data_create_V2.py" based on delete type (D) and phone combine type (tv_mode)
# Outputs:
#   choose_mode: used to name the result csv file
#   train_data, val_data: Data contain RSS data and Geo data
#
# Inputs:
# D: Delete type , it should be 0, 1, or 2
# tv_mode: The phone combination.
def file_open(D, tv_mode):
    # 确定数据集的类型
    train_mode, val_mode = tv_mode.split('-', 1)
    # print(train_mode+'@'+val_mode)
    train_file = 'train_val_data/' + D + train_mode + '_train_set.csv'
    train_data = pd.read_csv(train_file)

    val_file = 'train_val_data/' + D + val_mode + '_val_set.csv'
    val_data = pd.read_csv(val_file)
    # print(train_file);print(val_file)
    choose_mode = {'train_set': train_mode, 'val_set': val_mode}
    return choose_mode, train_data, val_data

# This function is to separate the data and its x-y labels. It will outputs train data and val data separately
#
# The RSS and Geo data are both processed by StandardScaler() of sklearn
#
# Inputs: train_df, test_df, batch_size
# Outputs: RSS_train, y_train, RSS_val, y_val, Geo_train, Geo_val,
def data_label_seperate(train_df, test_df, batch_size):
    data_size = int(train_df.shape[1] - 5)
    len_train = int(train_df.shape[0])

    scaler_train_AP = StandardScaler()
    scaler_test_AP = StandardScaler()
    train_AP_features = scaler_train_AP.fit_transform(np.asarray(train_df.iloc[:, 0:data_size]).astype(float))
    val_AP_features = scaler_test_AP.fit_transform(np.asarray(test_df.iloc[:, 0:data_size]).astype(float))

    train_Geo_features = train_df.iloc[:, data_size + 2:]
    val_Geo_features = test_df.iloc[:, data_size + 2:]

    x_all = np.asarray(pd.get_dummies(pd.concat([train_df['Loc_x'], test_df['Loc_x']]))) # Output One-Hot label
    y_all = np.asarray(pd.get_dummies(pd.concat([train_df['Loc_y'], test_df['Loc_y']])))

    train_labels = np.concatenate((x_all, y_all), axis=1)

    RSS_train = train_AP_features
    Geo_train = train_Geo_features
    y_train = train_labels[:len_train]

    RSS_val = val_AP_features
    Geo_val = val_Geo_features
    y_val = train_labels[len_train:]

    Geo_train, Geo_val = geo_preprocess(Geo_train, Geo_val, batch_size)

    return RSS_train, y_train, RSS_val, y_val, Geo_train, Geo_val,

# This function is to process the Geo data. It applies StandardScaler() to exert preprocess
def geo_preprocess(train_data, val_data, batch_size):
    train_Mag_x = train_data['rGeoX']
    train_Mag_x = np.array(train_Mag_x).astype(float).reshape(-1, 1)
    scaler_Mag_x = StandardScaler()
    train_Mag_x = scaler_Mag_x.fit_transform(train_Mag_x)

    train_Mag_y = train_data['rGeoY']
    train_Mag_y = np.array(train_Mag_y).astype(float).reshape(-1, 1)
    scaler_Mag_y = StandardScaler()
    train_Mag_y = scaler_Mag_y.fit_transform(train_Mag_y)

    train_Mag_z = train_data['rGeoZ']
    train_Mag_z = np.array(train_Mag_z).astype(float).reshape(-1, 1)
    scaler_Mag_z = StandardScaler()
    train_Mag_z = scaler_Mag_z.fit_transform(train_Mag_z)

    val_Mag_x = val_data['rGeoX']
    val_Mag_x = np.array(val_Mag_x).astype(float).reshape(-1, 1)
    scaler_val_Mag_x = StandardScaler()
    val_Mag_x = scaler_val_Mag_x.fit_transform(val_Mag_x)

    val_Mag_y = val_data['rGeoY']
    val_Mag_y = np.array(val_Mag_y).astype(float).reshape(-1, 1)
    scaler_val_Mag_y = StandardScaler()
    val_Mag_y = scaler_val_Mag_y.fit_transform(val_Mag_y)

    val_Mag_z = val_data['rGeoZ']
    val_Mag_z = np.array(val_Mag_z).astype(float).reshape(-1, 1)
    scaler_val_Mag_z = StandardScaler()
    val_Mag_z = scaler_val_Mag_z.fit_transform(val_Mag_z)

    train_Geo = np.concatenate((train_Mag_x, train_Mag_y, train_Mag_z), axis=1)
    test_Geo = np.concatenate((val_Mag_x, val_Mag_y, val_Mag_z), axis=1)

    return train_Geo, test_Geo

# This function build and train the Auto-Encoder and delete the decoder part
def build_model(sae_hidden_layers, INPUT_DIM, SAE_ACTIVATION, SAE_BIAS, SAE_OPTIMIZER, SAE_LOSS, batch_size, epochs,
                VERBOSE, RSS_train, y_train):
    # create a model based on stacked autoencoder (SAE)
    model = Sequential()
    model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
    for units in sae_hidden_layers[1:]:
        model.add(Dense(units, activation=SAE_ACTIVATION, use_bias=SAE_BIAS, activity_regularizer=regilization, ))
    model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS, ))
    model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)

    # train the model
    model.fit(RSS_train, RSS_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

    # remove the decoder part
    num_to_remove = (len(sae_hidden_layers) + 1) // 2
    for i in range(num_to_remove):
        model.pop()
    model.add(Dropout(dropout))
    return model

#This function  build the MLP part
def build_model_MLP(INPUT_DIM, feature_size=3):
    Geo_input = Input(batch_shape=(batch_size, feature_size))
    AE_output = Input(batch_shape=(batch_size, hidden_nodes))

    MLP_input = Concatenate(axis=1)([Geo_input, AE_output])
    x = MLP_input
    for units in classifier_hidden_layers:
        x = Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=CLASSIFIER_BIAS,
                  activity_regularizer=regilization, )(x)
        x = Dropout(dropout)(x)
    output = Dense(OUTPUT_DIM, activation='sigmoid', use_bias=CLASSIFIER_BIAS, )(x)

    model = Model(inputs=[Geo_input, AE_output], outputs=output)
    # model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])
    return model

# This function conbine the Auto-Encoder and the MLP. The output of AE are combined with the Geo data.
# Therefore, shape of AE's output is [batch_size, 64+3]
# The Loss function of AE-MLP is the CLASSIFIER_LOSS.
def AE_MLP_combine(model_AE, model_MLP, feature_size=3):
    RSS_train = Input(shape=(INPUT_DIM,))
    Geo_input = Input(shape=(feature_size,))

    AE_out = model_AE(RSS_train)
    MLP_out = model_MLP(inputs=[Geo_input, AE_out])

    model_AE_MLP = Model(outputs=MLP_out, inputs=[RSS_train, Geo_input])
    model_AE_MLP.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])

    #
    return model_AE_MLP

# This function trains and saves AE-MLP. The result saves in the path based on file_acc2loss
def predict_evaluate(model, RSS_train, Geo_train, y_train, RSS_val, Geo_val, y_val, file_acc2loss):
    adam = optimizers.Adam(lr=0.001, beta_1=0.99999, beta_2=0.999, epsilon=1e-09)
    reduce_lr = ReduceLROnPlateau(optimizer=adam, monitor='val_loss', patience=4, factor=0.9, mode='min')
    history = model.fit([RSS_train, Geo_train], y_train, validation_data=([RSS_val, Geo_val], y_val),
                        batch_size=batch_size, epochs=epochs, verbose=VERBOSE, callbacks=[reduce_lr])
    train_result = pd.DataFrame(history.history)
    train_result.to_csv(file_acc2loss)
    del reduce_lr
    del adam


if __name__ == "__main__":
    args = param()
    # set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    training_ratio = args.training_ratio
    sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.classifier_hidden_layers == '':
        classifier_hidden_layers = ''
    else:
        classifier_hidden_layers = [int(i) for i in (args.classifier_hidden_layers).split(',')]
    dropout = args.dropout
    # building_weight = args.building_weight
    # floor_weight = args.floor_weight
    N = args.neighbours
    scaling = args.scaling
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# To change the delete methods, You can change the parameter  $$$ delete=['0','1','2'] $$$
# To change the phone combination, You can change the parameter $$$ all_mode $$$
#
# Parameter file_acc2loss is the path to store the output result.
    # file is a subfolder of result.
    # !!!!!
    # If you want to change "file", make sure the folder has been created
#
#  !!!!warning !!!!
    #  The delete type '2' (D='2') can only use all_mode=[a-a,b-b]
    #  The reason is I can not train the model with data in different size
    #  More information can be found in "data_create_V2.py"
    delete = ['0','1','2']
    for D in delete:
        if D != '2':
            all_mode = [
                'a-a', 'a-b', 'a-ab',
                'b-a', 'b-b', 'b-ab',
                'ab-a', 'ab-b', 'ab-ab'
            ]
        else:
            all_mode = ['a-a', 'b-b']
        for mode in all_mode:
            file = 'standard_scaler'

            choose_mode, train_df, test_df = file_open(D, mode)
            file_acc2loss = 'result/' + file + '/' + D + choose_mode['train_set'] + '-' + choose_mode[
                'val_set'] + '_Epochs_' + str(epochs) + 'Droupout_' + str(dropout) + '_acc.csv'
            print(file_acc2loss)

            RSS_train, y_train, RSS_val, y_val, Geo_train, Geo_val = data_label_seperate(train_df, test_df, batch_size)
            INPUT_DIM = RSS_train.shape[1]
            OUTPUT_DIM = y_val.shape[1]

            model_AE = build_model(sae_hidden_layers, INPUT_DIM, SAE_ACTIVATION, SAE_BIAS, SAE_OPTIMIZER, SAE_LOSS,
                                   batch_size, epochs, VERBOSE, RSS_train, y_train)
            model_MLP = build_model_MLP(INPUT_DIM)
            model_AE_MLP = AE_MLP_combine(model_AE, model_MLP)

            predict_evaluate(model_AE_MLP, RSS_train, Geo_train, y_train, RSS_val, Geo_val, y_val, file_acc2loss)

            del model_AE_MLP
            del model_AE
            del model_MLP