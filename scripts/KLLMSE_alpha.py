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
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import coremltools as ct
from timeit import default_timer as timer

train_ds = pd.read_csv("/Users/arvinprince/tensorflow-files/kaggle-llm-science-exam/data/train.csv")

class ProcessingLayer(keras.layers.Layer):
    """
    A processing layer created for the kaggle competitions "Kaggle-LLM Science exam".

    It accepts a dataframe (containing the prompt, the options, and the answer) as input
    and then outputs a tuple containing tensors: prompt and options(combined or separated) and answers.

    All output tensors are encoded and one-hot encoded or embedded and padded if needed.

    ...
        :param num_words: maximum number of words to keep in the vocabulary (default: 1000)
        :type num_words: int
        :param oov_token: token to be used if there is a word out of vocabulary
        :type oov_token: str
        :param lower: if words need to be lowered when vocabulary is being created
        :type lower: bool
        :param is_prompt: whether prompt values need to be returned separately or not
        :type is_prompt: bool
        :rtype: tuple
    """
    def __init__(self,
                 num_words=1000,
                 oov_token="OOV",
                 lower=False,
                 is_prompt=False):
        super().__init__()
        # tokenizer to convert the phrase to sequences
        self.tokeinizer = Tokenizer(num_words=num_words, oov_token=oov_token, lower=lower)
        # encoder to ordinal encode the answer columns since it contains single letters and not phrases
        self.encoder = OrdinalEncoder(dtype=np.float32)
        # parameter for determining if the output would contain a separate prompt tensor
        self._true = is_prompt

    def call(self, inputs, *args, **kwargs):
        # converting the dataset to text sequences and one-hot encoded values
        prompt, options, answer = self.text2seq(inputs)
        # a condition to check if the prompt values should be a separate tensor
        if self._true:
            return tf.squeeze(prompt), options, answer  # separate prompt tensor
        else:
            return tf.concat([prompt, options], axis=1), answer  # combined prompt and options tensor

    def processing(func):
        """
        Decorator function created to wrap text2seq function.

        Post-processing (tensor conversion, padding, expansion of dimensions, dimensionality reduction,
        one-hot encoding, and tensor concatenation) is implemented on the outputs of text2seq function

        :return: A tuple of tensors
        """
        def wrapper(*args, **kwargs):
            prompt, options, answer = func(*args, **kwargs)  # getting the outputs from the wrapped function
            # converting the prompt values from list to tensor; subsequently padding the values
            # the extra dimension is added as a safeguard if we want to concatenate the values with options values
            prompt = pad(to_tensor(prompt))[:, tf.newaxis, :]
            # one-hot encoding the answer values; later removing the extra unit dimension
            answer = tf.squeeze(tf.one_hot(answer, 5))
            # converting the options values from list to tensor; subsequently padding the values
            options = tf.concat([pad(to_tensor(opt))[:, tf.newaxis, :] for opt in options], axis=1)
            return prompt, options, answer

        def pad(tensor):
            """
            A function created to pad a tensor to the desired shape

            :param tensor: tensor that is about to be padded
            :return: A tensor
            """
            padded = tf.pad(tensor=tensor,paddings=[[0, 0], [0, 130 - tf.shape(tensor)[-1]]],
                            mode="CONSTANT",constant_values=0)  # padding the sequences to a fixed length
            return padded

        def to_tensor(list):
            """
            A function that accepts a list containing a collection of list with different lengths and then converting
            it to a tensor with uniform dimensions

            :param list: a list containing a collection of list with different lengths
            :return: A tensor
            """
            # converting the irregularly shaped lists to ragged tensors which are later converted to regular tensors
            return tf.ragged.constant(list, dtype=tf.float32).to_tensor()

        return wrapper

    @processing
    def text2seq(self, ds):
        """
        A function that creates lists of sequences from a dataset containing texts
        :param ds: dataset over which the processing would be performed
        :return: A tuple of lists
        """
        text_gen = self.text_gen(ds)
        self.tokeinizer.fit_on_texts(text_gen)  # updating the vocabulary of the tokenizer
        prompt, options, answer = self.gather_features(ds)  # acquiring the prompt, options and answer
        prompt = self.tokeinizer.texts_to_sequences(prompt)  # transforming the prompt text to sequences
        # converting individual choice from options
        options = [self.tokeinizer.texts_to_sequences(options[k]) for k in options.columns]
        answer = self.encoder.fit_transform(answer.reshape((-1,1))) # encoding the answers to numeric values
        return prompt, options, answer

    def gather_features(self, df: pd.DataFrame):
        """
        a function that gathers the required columns from the given dataset
        :param df:  which contains the question, choices and answer
        :return: DataFrame/Series/numpy array
        """
        # gathering the index of the choices in the dataframe
        self.options_idx = [i for i, col in enumerate(df) if col in ["A", "B", "C", "D", "E"]]
        prompt = df.prompt # gathering the prompt from the dataset
        options = df.take(self.options_idx, axis=1) # gathering the options from the dataset
        answer = df.answer # gathering the answer from the dataset
        return prompt, options, answer.values

    def text_gen(self, df:pd.DataFrame):
        """
        A generator that generates texts present in the provided datset
        :param df: which contains the texts
        :return: generator
        """
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
        # pos_enc = self.positional_encoding(num_phrases=self.num_phrases,
        # length=self.max_len, depth=self.vocab_size, dtype=self.type)
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

X, y=tf.cast(out_pe, tf.float32), tf.cast(out_pl[1], tf.float32)

wiring = wirings.FullyConnected(64, 10)

model = keras.Sequential([
    layers.TimeDistributed(layers.Conv1D(128, 7, 1, "same", activation="relu")),
    layers.TimeDistributed(layers.Flatten()),
    layers.TimeDistributed(layers.Dense(128, activation="gelu")),
    LTC(wiring, return_sequences=True),
    layers.GlobalAvgPool1D(),
    layers.Dense(5)
])

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.legacy.Adam(learning_rate=.015)

model.compile(optimizer=optimizer, loss=loss_fn)

history = model.fit(x=X, y=y, batch_size=100, epochs=1, validation_split=.2)

training_plots = pd.DataFrame(history.history)
training_plots.plot()
plt.show()

# coreml = ct.convert(model, convert_to="mlprogram",
#                     compute_units=ct.ComputeUnit.ALL)
#
# start = timer()
# coreml.predict({"time_distributed_input": X._numpy()[:5]})
# end = timer()
# print((end-start))
#
# start = timer()
# model.predict(X)
# end = timer()
# print((end-start))
