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

        if self.VAE:
            X = layers.Dense(self.embedding, name="embedding_pre")(X)
            X = layers.Activation('relu')(X)
            X = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(X)
            z_mean = layers.Dense(self.embedding, name="z_mean")(X)
            z_log_var = layers.Dense(self.embedding, name="z_log_var")(X)
            self.sampling = Sampling()((z_mean, z_log_var))
            # update the self.mean and self.std:
        #            self.mean = z_mean
        #            self.std = z_log_var

        self.encoder_model = Model(inputs=encoder_input, outputs=self.sampling, name='LSTM_encoder')

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

        #     X = layers.LayerNormalization(axis=1, name='batch_normal')(X)
        X = layers.TimeDistributed(Dense(2, activation='linear'))(X)

        self.decoder_model = Model(inputs=decoder_input, outputs=X, name='LSTM_encoder')

        outputs = self.decoder_model(self.sampling)

        self.vae = tf.keras.Model(inputs=encoder_input, outputs=outputs, name="vae")

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.vae.add_loss(self.coef * kl_loss)







