import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.utils import Progbar
from ncps.tf import LTC
from ncps import wirings
import pandas as pd
import numpy as np
from time import sleep
from rich.progress import track
from collections import defaultdict

train_ds = pd.read_csv("data/train.csv")


def sentence_gen(df: pd.DataFrame):
    text_ds = df.select_dtypes(object)
    for c in range(text_ds.shape[-1]):
        for t in range(text_ds.shape[0]):
            yield text_ds.iloc[t, c]


text_gen = sentence_gen(train_ds)
tokeinizer = Tokenizer(num_words=5000, oov_token="OOV", lower=False)
tokeinizer.fit_on_texts(text_gen)
len(tokeinizer.word_index)

dum = tf.ragged.constant(train_ds.prompt, dtype=tf.string)
prompt = tf.ragged.constant(tokeinizer.texts_to_sequences(train_ds.prompt)).to_tensor()
A = tf.ragged.constant(tokeinizer.texts_to_sequences(train_ds.A)).to_tensor()
prompt_pad= tf.pad(prompt, [[0,0], [0,130 - prompt.shape[-1]]],\
                   mode="CONSTANT", constant_values=0)[:, tf.newaxis, :]
A_pad= tf.pad(A, [[0,0], [0,130 - A.shape[-1]]],\
              mode="CONSTANT", constant_values=0)[:, tf.newaxis, :]
input = tf.concat([prompt_pad, A_pad, A_pad], axis=1)


class ProcessingLayer(layers.Layer):
    def __init__(self,
                 num_words=1000,
                 oov_token="OOV",
                 lower=False):
        super().__init__()
        self.tokeinizer = Tokenizer(num_words=num_words, oov_token=oov_token, lower=lower)

    def call(self, inputs, *args, **kwargs):
        prompt, options, answer = self.text2seq(inputs)
        return prompt, options, answer

    def text_gen(self, df:pd.DataFrame):
        text_ds = df.select_dtypes(object)
        for c in range(text_ds.shape[-1]):
            for t in range(text_ds.shape[0]):
                yield text_ds.iloc[t, c]

    def gather_features(self, df:pd.DataFrame):
        df = df.set_index("id")
        self.prompt_idx = [i for i, col in enumerate(df) if "prompt" in col]
        self.options_idx = [i for i, col in enumerate(df) if col in ["A", "B", "C", "D", "E"]]
        self.answer_idx = [i for i, col in enumerate(df) if "answer" in col]
        prompt = df.take(self.prompt_idx, axis=1)
        options = df.take(self.options_idx, axis=1)
        answer = df.take(self.answer_idx, axis=1)
        return prompt, options, answer

    def processing(func):
        def wrapper(*args, **kwargs):
            p, o, a = func(*args, **kwargs)
            p = pad(to_tensor(p))[:, tf.newaxis, :]
            a = to_tensor(a)
            o = tf.concat([pad(to_tensor(opt))[:, tf.newaxis, :] for opt in o], axis=1)
            return (p, o, a)

        def pad( tensor):
            padded = tf.pad(tensor=tensor,
                            paddings=[[0, 0], [0, 130 - tf.shape(tensor)[-1]]],
                            mode="CONSTANT",
                            constant_values=0)
            return padded

        def to_tensor( tensor):
            return tf.ragged.constant(tensor, dtype=tf.uint32).to_tensor()

        return wrapper
    @processing
    def text2seq(self, ds):
        self.tokeinizer.fit_on_texts(self.text_gen(ds))
        prompt, options, answer = self.gather_features(ds)
        prompt = self.tokeinizer.texts_to_sequences(prompt)
        options = [self.tokeinizer.texts_to_sequences(k) for k in options]
        answer = self.tokeinizer.texts_to_sequences(answer)
        return prompt, options, answer

sample_pl = ProcessingLayer(num_words=5000)
out_pl = sample_pl(train_ds)



