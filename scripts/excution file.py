import sys

sys.path.extend(["/Users/arvinprince/Tensorflow/kaggle-llm-science-exam/scripts"])

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
from preprocessing import ProcessingLayer, PositionalEmbedding, dataset_generator

train_ds = pd.read_csv("/Users/arvinprince/Tensorflow/kaggle-llm-science-exam/data/train.csv")


sample_pl = ProcessingLayer(num_words=5000, is_prompt=True)
out_pl = sample_pl(train_ds)


sample_pe = PositionalEmbedding(5000, 64, batch_size=5, dtype=tf.float32)
out_pe = sample_pe(out_pl[0])
out_pe_ = sample_pe(out_pl[1])


ds = dataset_generator((out_pe, out_pe), 20)








