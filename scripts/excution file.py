import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.utils import Progbar
import pandas as pd
import numpy as np
from rich.progress import track
from collections import defaultdict
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from preprocessing import ProcessingLayer, PositionalEmbedding

train_ds = pd.read_csv("/Users/arvinprince/tensorflow-files/kaggle-llm-science-exam/data/train.csv")


sample_pl = ProcessingLayer(num_words=5000)
out_pl = sample_pl(train_ds)


sample_pe = PositionalEmbedding(5000, 64, dtype=tf.float32)
out_pe = sample_pe(out_pl[0])






