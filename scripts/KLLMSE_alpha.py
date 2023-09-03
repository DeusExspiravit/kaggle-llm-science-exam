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

train_ds = pd.read_csv("/Users/arvinprince/tensorflow-files/kaggle-llm-science-exam/data/train.csv")

class ProcessingLayer(keras.layers.Layer):
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

class PositionalEmbedding(keras.layers.Layer):
    def __init__(self,
                 vocab_size: int,
                 # max_len: int,
                 embed_dim: int,
                 # num_phrases:int,
                 dtype = tf.int32):
        super().__init__()
        self.d_model = embed_dim
        self.embedding = keras.layers.Embedding(vocab_size, embed_dim)
        self.type = dtype

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, *args, **kwargs):
        # pos_enc = self.positional_encoding(num_phrases=self.num_phrases, length=self.max_len, depth=self.vocab_size, dtype=self.type)
        # dim = tf.shape(x)[-1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=self.type))
        # x = pos_enc[tf.newaxis, :, :, :self.emb_dim] + x
        return x

    def positional_encoding(self,num_phrases, length, depth, dtype):
        depth = int(depth / 2)

        positions = np.arange(num_phrases*length).reshape((num_phrases, length))[:, :, np.newaxis]
        depths = np.arange(int(length*depth)).reshape((length, depth))[np.newaxis, :, :] / depth

        angle_rates = 1 / (1e4 ** depths)
        angle_rads = positions * angle_rates

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1,
        )
        return tf.cast(pos_encoding, dtype=dtype)

sample_pe = PositionalEmbedding(5000, 64, dtype=tf.float32)
out_pe = sample_pe(out_pl[0])

wiring = wirings.AutoNCP(32, 5)

model = keras.Sequential([
    layers.TimeDistributed(layers.Conv1D(64, 3, 2, "same", activation="relu")),
    layers.TimeDistributed(layers.Flatten()),
    layers.TimeDistributed(layers.Dense(128, activation="relu")),
    LTC(wiring, return_sequences=True),
    layers.GlobalAvgPool1D(),
    layers.Dense(5)
])

model.compile(optimizer="adam", loss=keras.losses.CategoricalCrossentropy)

model.fit(tf.cast(out_pe, tf.float32), tf.cast(out_pl[1], tf.float32), batch_size=25, epochs=1)