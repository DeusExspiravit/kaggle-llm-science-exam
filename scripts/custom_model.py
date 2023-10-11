import tensorflow as tf
import keras
from keras import *
from keras.utils import Progbar
import numpy as np
import pandas as pd

class DenseLayer(layers.Layer):
    def __init__(self, units, initialization,
                 use_bias=False, dtype=tf.float32):
        super().__init__()
        self._units = units
        self._usebias = use_bias
        self._init = initialization
        self._dtype = dtype
    def build(self, input_shape):
        if isinstance(self._init, tf.initializers.Initializer):
            w_init = self._init
        else:
            w_init = tf.initializers.random_normal()

        b_init = tf.initializers.zeros()

        w_shape = (input_shape[-1], self._units)
        b_shape = (self._units, )

        self._weight = tf.Variable(
            name= "weight",
            initial_value=w_init(shape= w_shape, dtype=self._dtype),
            trainable=True
        )

        if self._usebias:
            self._bias = tf.Variable(
                name="bias",
                initial_value=b_init(shape=b_shape, dtype=self._dtype),
                trainable=True
            )
        else:
            self._bias = None
    def call(self, inputs, *args, **kwargs):
        return tf.matmul(inputs, self._weight) + self._bias

class RetentiveLayer(layers.Layer):
    def __init__(self, prompt_units, option_units, batch_size, max_len):
        super().__init__()
        self._p_units = prompt_units
        self._o_units = option_units
        self._batch_size = batch_size
        self._max_len = max_len

    def build(self, input_shape):

        hid_init = tf.initializers.RandomUniform()

        self._hidden_state = tf.Variable(
            name="hidden_state",
            initial_value= hid_init(shape=(self._batch_size, self._max_len, self._p_units), dtype=tf.float32)
        )
        self._prompt_layer = DenseLayer(self._p_units, use_bias=True)
        self._hidden_layer = DenseLayer(self._p_units, use_bias=True)
        self._A_layer = DenseLayer(self._o_units, initialization=initializers.glorot_normal())
        self._B_layer = DenseLayer(self._o_units, initialization=initializers.glorot_normal())
        self._C_layer = DenseLayer(self._o_units, initialization=initializers.glorot_normal())
        self._D_layer = DenseLayer(self._o_units, initialization=initializers.glorot_normal())
        self._E_layer = DenseLayer(self._o_units, initialization=initializers.glorot_normal())

    def call(self, prompt, options, answer, *args, **kwargs):
        A, B, C, D, E = (options[:, i, :, :] for i in range(5))







