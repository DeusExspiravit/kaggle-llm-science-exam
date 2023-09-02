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

train_ds = pd.read_csv("/kaggle-llm-science-exam/data/train.csv")

class ProcessingLayer(layers.Layer):
    def __init__(self,
                 num_words=1000,
                 oov_token="OOV",
                 lower=False,
                 is_prompt=False):
        super().__init__()
        self.tokeinizer = Tokenizer(num_words=num_words, oov_token=oov_token, lower=lower)
        self._true = is_prompt

    def call(self, inputs, *args, **kwargs):
        prompt, options, answer = self.text2seq(inputs)
        if self._true:
            return (tf.squeeze(prompt), options, answer)
        else:
            return(tf.concat([prompt, options], axis=1), answer)

    def processing(func):
        def wrapper(*args, **kwargs):
            prompt, options, answer = func(*args, **kwargs)
            prompt = pad(to_tensor(prompt))[:, tf.newaxis, :]
            answer = to_tensor(answer)
            options = tf.concat([pad(to_tensor(opt))[:, tf.newaxis, :] for opt in options], axis=1)
            return (prompt, options, answer)

        def pad(tensor):
            padded = tf.pad(tensor=tensor,paddings=[[0, 0], [0, 130 - tf.shape(tensor)[-1]]],
                            mode="CONSTANT",constant_values=0)
            return padded

        def to_tensor(tensor):
            return tf.ragged.constant(tensor, dtype=tf.uint32).to_tensor()

        return wrapper

    @processing
    def text2seq(self, ds):
        text_gen = self.text_gen(ds)
        self.tokeinizer.fit_on_texts(text_gen)
        prompt, options, answer = self.gather_features(ds)
        prompt = self.tokeinizer.texts_to_sequences(prompt)
        options = [self.tokeinizer.texts_to_sequences(options[k]) for k in options.columns]
        answer = self.tokeinizer.texts_to_sequences(answer)
        return prompt, options, answer

    def gather_features(self, df:pd.DataFrame):
        # df = df.set_index("id")
        # self.prompt_idx = [i for i, col in enumerate(df) if "prompt" in col]
        self.options_idx = [i for i, col in enumerate(df) if col in ["A", "B", "C", "D", "E"]]
        # self.answer_idx = [i for i, col in enumerate(df) if "answer" in col]
        prompt = df.prompt
        options = df.take(self.options_idx, axis=1)
        answer = df.answer
        return prompt, options, answer

    def text_gen(self, df:pd.DataFrame):
        text_ds = df.select_dtypes(object)
        for c in range(text_ds.shape[-1]):
            for t in range(text_ds.shape[0]):
                yield text_ds.iloc[t, c]

sample_pl = ProcessingLayer(num_words=5000)
out_pl = sample_pl(train_ds)


