"""

"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input,Reshape,Activation,Attention,MaxPool1D,Dense, Conv1D, Convolution2D, GRU, LSTM, Lambda, Bidirectional, TimeDistributed,
                          Dropout, Flatten, LayerNormalization,RepeatVector, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization)
import tensorflow.keras.layers as layers
import string
from tensorflow.keras.regularizers import l1, l2
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
import os
from .file import *





class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class model_builder:

    def __init__(self,
                 input_data,
                 drop_frac=0.2,
                 layer_size=128,
                 num_ident_blocks=3,
                 l1_norm=0,
                 l1_norm_embedding=1e-3,
                 layer_steps=2,
                 embedding=16,
                 VAE=True,
                 coef=1):
        # Sets self.mean and self.std to use in the loss function;
        #       self.mean = 0
        #       self.std = 0

        # Sets the L1 norm on the decoder/encoder layers
        self.l1_norm = l1_norm

        # Sets the fraction of dropout
        self.drop_frac = drop_frac

        # saves the shape of the input data
        self.data_shape = input_data.shape

        # Sets the number of neurons in the encoder/decoder layers
        self.layer_size = layer_size

        # Sets the number of neurons in the embedding layer
        self.embedding = embedding

        # Bool to set if the model is a VAE
        self.VAE = VAE

        # Set the magnitude of the l1 regularization on the embedding layer.
        self.l1_norm_embedding = l1_norm_embedding

        # sets the number of layers between the residual layer
        self.layer_steps = layer_steps

        self.coef = coef

        # set the number of identity block
        self.num_ident_blocks = num_ident_blocks

        self.model_constructor(input_data)

    def identity_block(self, X, name,
                       block):

        # sets the name of the conv layers
        LSTM_name_base = name + '_LSTM_Res_' + block
        bn_name_base = name + '_layer_norm_' + block

        # output for the residual layer
        X_shortcut = X

        for i in range(self.layer_steps):
            # bidirectional LSTM
            X = layers.Bidirectional(LSTM(self.layer_size,
                                          return_sequences=True,
                                          dropout=self.drop_frac,
                                          activity_regularizer=l1(self.l1_norm)),
                                     input_shape=(self.data_shape[1], self.data_shape[2]))(X)

            # TODO, We could add layer norm
            X = layers.Activation('relu')(X)

        X = layers.add([X, X_shortcut])
        X = layers.LayerNormalization(axis=1, name=bn_name_base + '_res_end')(X)
        X = layers.Activation('relu')(X)

        return X

    def model_constructor(self, input_data):
        # defines the input
        encoder_input = layers.Input(shape=(self.data_shape[1:]))

        X = encoder_input

        for i in range(self.num_ident_blocks):
            X = self.identity_block(X, 'encoder', string.ascii_uppercase[i + 1])

        # This is in preparation for the embedding layer
        X = layers.Bidirectional(LSTM(self.layer_size,
                                      return_sequences=False,
                                      dropout=self.drop_frac,
                                      activity_regularizer=l1(self.l1_norm)),
                                 input_shape=(self.data_shape[1],
                                              self.data_shape[2]))(X)

        X = layers.BatchNormalization(axis=1, name='last_encode')(X)
        X = layers.Activation('relu')(X)

        if self.VAE:
            X = layers.Dense(self.embedding, name="embedding_pre")(X)
            X = layers.Activation('relu')(X)
            X = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(X)
            z_mean = layers.Dense(self.embedding, name="z_mean")(X)
            z_log_var = layers.Dense(self.embedding, name="z_log_var")(X)
            sampling = Sampling()((z_mean, z_log_var))
            # update the self.mean and self.std:
        #            self.mean = z_mean
        #            self.std = z_log_var

        self.encoder_model = Model(inputs=encoder_input, outputs=sampling, name='LSTM_encoder')

        decoder_input = layers.Input(shape=(self.embedding,), name="z_sampling")

        z = layers.Dense(self.embedding, name="embedding")(decoder_input)
        z = layers.Activation('relu')(z)
        z = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(z)

        X = layers.RepeatVector(self.data_shape[1])(z)

        X = layers.Bidirectional(LSTM(self.layer_size, return_sequences=True,
                                      dropout=self.drop_frac,
                                      activity_regularizer=l1(self.l1_norm)))(X)

        # X = layers.BatchNormalization(axis = 1, name = 'fires_decode')(X)
        X = layers.Activation('relu')(X)

        for i in range(self.num_ident_blocks):
            X = self.identity_block(X, 'decoder', string.ascii_uppercase[i + 1])

        X = layers.LayerNormalization(axis=1, name='batch_normal')(X)
        X = layers.TimeDistributed(Dense(1, activation='linear'))(X)

        self.decoder_model = Model(inputs=decoder_input, outputs=X, name='LSTM_encoder')

        outputs = self.decoder_model(sampling)

        self.vae = tf.keras.Model(inputs=encoder_input, outputs=outputs, name="vae")

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.vae.add_loss(self.coef * kl_loss)


class model_builder_combine:

    def __init__(self,
                 input_data,
                 drop_frac=0.0,
                 layer_size=128,
                 num_ident_blocks=3,
                 l1_norm=0,
                 l1_norm_embedding=1e-3,
                 layer_steps=2,
                 embedding=16,
                 VAE=True,
                 coef=1):
        # Sets self.mean and self.std to use in the loss function;
        #       self.mean = 0
        #       self.std = 0

        # Sets the L1 norm on the decoder/encoder layers
        self.l1_norm = l1_norm

        # Sets the fraction of dropout
        self.drop_frac = drop_frac

        # saves the shape of the input data
        self.data_shape = input_data.shape

        # Sets the number of neurons in the encoder/decoder layers
        self.layer_size = layer_size

        # Sets the number of neurons in the embedding layer
        self.embedding = embedding

        # Bool to set if the model is a VAE
        self.VAE = VAE

        # Set the magnitude of the l1 regularization on the embedding layer.
        self.l1_norm_embedding = l1_norm_embedding

        # sets the number of layers between the residual layer
        self.layer_steps = layer_steps

        self.coef = coef

        # set the number of identity block
        self.num_ident_blocks = num_ident_blocks

        self.model_constructor(input_data)

    def identity_block(self, X, name,
                       block):

        # sets the name of the conv layers
        LSTM_name_base = name + '_LSTM_Res_' + block
        bn_name_base = name + '_layer_norm_' + block

        # output for the residual layer
        X_shortcut = X

        for i in range(self.layer_steps):
            # bidirectional LSTM
            X = layers.Bidirectional(LSTM(self.layer_size,
                                          return_sequences=True,
                                          dropout=self.drop_frac,
                                          activity_regularizer=l1(self.l1_norm)),
                                     input_shape=(self.data_shape[1] * 2, 1))(X)

            # TODO, We could add layer norm
            X = layers.Activation('relu')(X)

        X = layers.add([X, X_shortcut])
        #    X = layers.LayerNormalization(axis = 1, name = bn_name_base + '_res_end')(X)
        X = layers.Activation('relu')(X)

        return X

    def model_constructor(self, input_data):
        # defines the input
        encoder_input = layers.Input(shape=(self.data_shape[1:]))
        X = layers.Flatten()(encoder_input)
        X = layers.RepeatVector(1)(X)
        X = layers.Permute((2, 1))(X)

        #      X = encoder_input

        for i in range(self.num_ident_blocks):
            X = self.identity_block(X, 'encoder', string.ascii_uppercase[i + 1])

        # This is in preparation for the embedding layer
        X = layers.Bidirectional(LSTM(self.layer_size,
                                      return_sequences=False,
                                      dropout=self.drop_frac,
                                      activity_regularizer=l1(self.l1_norm)),
                                 input_shape=(self.data_shape[1] * 2,
                                              1))(X)

        #     X = layers.BatchNormalization(axis=1, name='last_encode')(X)
        X = layers.Activation('relu')(X)

        #    if self.VAE:

        #    if self.VAE:
        X = layers.Dense(self.embedding, name="embedding_pre")(X)
        X = layers.Activation('relu')(X)
        Embedding_out = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(X)
        z_mean = layers.Dense(self.embedding, name="z_mean")(Embedding_out)
        z_log_var = layers.Dense(self.embedding, name="z_log_var")(Embedding_out)

        # update the self.mean and self.std:
        #            self.mean = z_mean
        #            self.std = z_log_var

        self.encoder_model = Model(inputs=encoder_input,
                                   outputs=[Embedding_out, z_mean, z_log_var], name='LSTM_encoder')

        #      decoder_input = layers.Input(shape=(self.embedding,), name="z_sampling")
        decoder_mean = layers.Input(shape=(self.embedding,), name="z_mean")
        decoder_log = layers.Input(shape=(self.embedding,), name="z_log")
        sampling = Sampling()((decoder_mean, decoder_log))

        #         self.encoder_model = Model(inputs=encoder_input,
        #                                outputs=sampling, name='LSTM_encoder')

        z = layers.Dense(self.embedding, name="embedding")(sampling)
        z = layers.Activation('relu')(z)
        z = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(z)

        X = layers.RepeatVector(self.data_shape[1])(z)

        X = layers.Bidirectional(LSTM(self.layer_size, return_sequences=True,
                                      dropout=self.drop_frac,
                                      activity_regularizer=l1(self.l1_norm)))(X)

        # X = layers.BatchNormalization(axis = 1, name = 'fires_decode')(X)
        X = layers.Activation('relu')(X)

        for i in range(self.num_ident_blocks):
            X = self.identity_block(X, 'decoder', string.ascii_uppercase[i + 1])

        #     X = layers.LayerNormalization(axis=1, name='batch_normal')(X)
        X = layers.TimeDistributed(Dense(2, activation='linear'))(X)

        self.decoder_model = Model(inputs=[decoder_mean, decoder_log], outputs=X, name='LSTM_encoder')

        outputs = self.decoder_model([z_mean, z_log_var])

        self.vae = tf.keras.Model(inputs=encoder_input, outputs=outputs, name="vae")

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.vae.add_loss(self.coef * kl_loss)


def Train(epochs,initial_epoch, epoch_per_increase, initial_beta, beta_per_increase
          ,new_data,folder_,ith_epoch=None,file_path=None,batch_size=300):
    """

    :param epochs: total epochs training
    :type epochs: int
    :param epoch_per_increase: number of epochs of each beta increase
    :type epoch_per_increase: int
    :param initial_beta: initial beta value
    :type initial_beta: float
    :param beta_per_increase: beta increase for each epoch_per_increase
    :type beta_per_increase: float
    :param new_data: input data set
    :type new_data: array
    :param folder_: folder save the weights
    :type folder_: string
    :param ith_epoch: training from the ith epoch
    :type ith_epoch: int
    :param file_path: weights dictionary from ith epoch
    :type file_path: string
    :param batch_size: batch size for training
    :type batch_size: int

    """
    best_loss = float('inf')
    iteration = (epochs // epoch_per_increase) + 1
    model = []
    # filepath =folder + '/if_appear_means_bug_happens.hdf5'
    if ith_epoch == None:
        list_= [0,iteration]
    else:
        list_=[ith_epoch,iteration]

    for i in range(list_[0],list_[1]):

        if i == iteration - 1:
            training_epochs = epochs - epoch_per_increase * (iteration - 1)
            if training_epochs <= 0:
                break
        elif i == 0:
            training_epochs = initial_epoch
        else:
            training_epochs = epoch_per_increase

        beta = initial_beta + beta_per_increase * i
        print(beta)
        del (model)
        model = model_builder(np.atleast_3d(new_data), embedding=16,
                              VAE=True, l1_norm_embedding=1e-5, coef=beta)
        beta = format(beta, '.4f')
        run_id = 'beta='+beta+'_beta_step_size='+str(beta_per_increase)+'_'+ np.str(model.embedding) + '_layer_size_' + np.str(
            model.layer_size) + '_l1_norm_' + np.str(model.l1_norm) + '_l1_norm_' + np.str(
            model.l1_norm_embedding) + '_VAE_' + np.str(model.VAE)
        folder = folder_ + '/' + run_id
        make_folder(folder)
        #
        # if i==30:
        #            filepath = 'piezoresponse+resonacnce_1/beta=0.0725__beta_step_siez=0.0025_16_layer_size_128_l1_norm_0_l1_norm_1e-05_VAE_True/triple_phase_weights2_epochs=29.hdf5'
        if i == ith_epoch:
            filepath = file_path

        if i > 0:
            print(filepath)
            model.vae.load_weights(filepath)


        # elif i == 22:
        #     training_epochs = 1000 - 79
        #     model.vae.load_weights(
        #         '/content/drive/My Drive/papers/Faster_better_v2_Training_11_06_2020/two_data_combined/piezoresponse+resonacnce/beta=0.055__beta_step_siez=0.0025_16_layer_size_128_l1_norm_0_l1_norm_1e-05_VAE_True/phase_shift_only0.055_epochs_begin_6000+22000+0079-0.03151.hdf5')
        # else:
        #     continue

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

        #        beta = beta + i*beta_per_increase
        # sets the file path
        epoch_begin = i * epoch_per_increase
        if i > 0:
            filepath = folder + '/phase_shift_only' + beta + '_epochs_begin_'+str(initial_epoch) +'+'+ np.str(
                epoch_begin) + '+{epoch:04d}' + '-{loss:.5f}.hdf5'
        else:
            filepath = folder + '/phase_shift_only' + beta + '_epochs_begin_' + np.str(
                epoch_begin) + '+{epoch:04d}' + '-{loss:.5f}.hdf5'

        # callback for saving checkpoints. Checkpoints are only saved when the model improves
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss',
                                                     verbose=0, save_best_only=True,
                                                     save_weights_only=True, mode='min')

        #         if i==0:

        #             model.vae.compile(optimizer, loss=KL_Loss(0,0,beta))
        #         else:
        #             model.vae.compile(optimizer, loss=KL_Loss(model.mean,model.std,beta))
        hist = model.vae.fit(np.atleast_3d(new_data),
                      np.atleast_3d(new_data),
                      batch_size, epochs=training_epochs, callbacks=[checkpoint])

        #        total_loss = hist.history['loss'][0]

        #        best_loss = total_loss
        min_loss = np.min(hist.history['loss'])
        user_input = folder
        directory = os.listdir(user_input)
        searchString = format(min_loss, '.5f')
        for fname in directory:  # change directory as needed
            if searchString in fname:
                f = fname
                filepath = user_input + '/' + str(f)


def get_activations(model, X=[], i=[], mode='test'):
    """
    function to get the activations of a specific layer
    this function can take either a model and compute the activations or can load previously
    generated activations saved as an numpy array

    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm

    Returns
    -------
    activation : float
        array containing the output from layer i of the network
    """
    # if a string is passed loads the activations from a file
    if isinstance(model, str):
        activation = np.load(model)
        print(f'activations {model} loaded from saved file')
    else:
        # computes the output of the ith layer
        activation = get_ith_layer_output(model, np.atleast_3d(X), i, mode)

    return activation



def get_ith_layer_output(model, X, i, mode='test'):
    """
    Computes the activations of a specific layer
    see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'


    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm
    Returns
    -------
    layer_output : float
        array containing the output from layer i of the network
    """
    # computes the output of the ith layer
#     get_ith_layer = keras.backend.function(
#         [model.layers[0].input, keras.backend.learning_phase()], [model.layers[i].output])
    get_ith_layer = tf.keras.backend.function(model.layers[0].input, model.layers[i].output)
#    layer_output = get_ith_layer([X, 0 if mode == 'test' else 1])[0]
    layer_output = get_ith_layer([X, 0 if mode == 'test' else 1])
#    print(layer_output.shape)
    return layer_output




